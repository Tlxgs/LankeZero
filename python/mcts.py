"""围棋 MCTS：Numba 扁平树 + 批量神经网络评估。

本机单进程版本：推理器由 evaluator.GoEvaluator（默认）或可选的
trt_engine.TrtEvaluator（TensorRT REFIT，自动回退）提供。
每个展开的节点都保留**全部合法着法**（不做候选裁剪）：实测裁剪会漏掉
「先验低但唯一正着」，而且裁剪本身是 O(合法点²)。训练目标仍来自根访问计数。

唯一例外是**自填单眼**（_drop_self_eyes）——周围被己方完全包住的空点不展开。
规则层不改：这些点在 legal_moves/GUI 里依旧合法（整块还有别的气时确实能落），
只是搜索没必要为「白送一手」浪费模拟。
"""
import math
import os

import numpy as np
from numba import njit

import shapes
from game import PASS_MOVE, HOLE, _make_move_dispatch
from hyperparams import (C_PUCT, NUM_SIMULATIONS, TEMPERATURE,
                         DIRICHLET_ALPHA, DIRICHLET_EPSILON, VIRTUAL_LOSS,
                         BATCH_SIZE_MCTS, MCTS_CAP, OMP_NUM_THREADS, ALPHA,
                         CPUCT_LOG, CPUCT_BASE, util_score_scale,
                         UTIL_STDEV_PRIOR, UTIL_STDEV_SCALE, FPU_REDUCTION,
                         MODEL_PATH)
from rules_adapter import GO_RULES

os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_DYNAMIC"] = "FALSE"


def _utility(win_logit, score, alpha, scale):
    """把「胜率 logit + 目差（单位：目）」混成 [-1,1] 的效用。

    score 现在一律是「目」：叶子用归属头逐点求和 + 贴目（见 margin_of），
    终局用真实目差。两处口径一致，所以同一个函数就够。

    scale 是 tanh 尺度（目），按**可落子面积**取 1.6·√面积，不是固定常数：
    固定 30 会让 9 路的 α 实际是「看目差」、19 路是「看胜负」。
    """
    return ((1.0 - alpha) * math.tanh(win_logit)
            + alpha * math.tanh(score / scale))


def margin_of(value, player, komi):
    """归属头逐点求和 -> 当前行棋方视角、**含贴目**的目差。

    value = Σ tanh(own) 是 mover 视角的「己方点数 - 对方点数」，
    黑贴出 komi：黑视角目差 = value - komi，白视角 = value + komi，
    合起来就是 value - player * komi。
    """
    return float(value) - float(player) * float(komi)


@njit(cache=True)
def _is_self_eye(board, n, p, player):
    """空点 p 是否被**己方**完全包住 —— 填进去就是自填单眼。

    判定（用户定义）：
    - 四个正交邻点必须全是己方棋子（少一个都可能提子/做活，必须搜）；
    - p 四周 8 格**全在盘内、且都不是洞**时，允许至多一个**对角**不是己方
      （标准的「真眼」判据：一个对角被占不会变成假眼）；
    - 只要 8 格里有洞、或 p 在边/角（有格子出界），一律按严格规则：
      所有存在的非洞邻点（正交 + 对角）都必须是己方，不豁免对角。

    洞/出界既不算己方、也不要求是己方，等价于「墙」。
    """
    r = p // n
    c = p % n
    bad_diag = 0
    strict = not (0 < r < n - 1 and 0 < c < n - 1)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= n or cc < 0 or cc >= n:
                continue
            v = board[rr, cc]
            if v == HOLE:
                strict = True
                continue
            if dr == 0 or dc == 0:
                if v != player:
                    return False
            elif v != player:
                bad_diag += 1
    if strict:
        return bad_diag == 0
    return bad_diag <= 1


@njit(cache=True)
def _drop_self_eyes(board, n, legal, count, player, out):
    """把自填单眼从搜索候选里剔掉，按原顺序写入 out，返回保留数量。

    只动搜索候选，不动合法性：调用方拿到的 legal 本来就是
    game.get_legal_indices_and_mask() 的结果，这些点仍然合法。
    """
    total = n * n
    kept = 0
    for j in range(count):
        p = legal[j]
        if p < total and _is_self_eye(board, n, p, player):
            continue
        out[kept] = p
        kept += 1
    return kept


@njit(cache=True)
def _select_child(cmove, cnode, cstart, ccount, visits, vsum, priors, vloss,
                  node, c_puct, c_log, c_base, sd_prior, sd_scale, fpu):
    start = cstart[node]
    end = start + ccount[node]
    if start == end:
        return -1
    if visits[node] > 0:
        pv = vsum[node] / visits[node]
    else:
        pv = 0.0
    qsum = 0.0
    qsq = 0.0
    n_vis = 0
    for s in range(start, end):
        ch = cnode[s]
        vis = visits[ch]
        q = -vsum[ch] / vis if vis > 0 else pv - fpu
        if vis > 0:
            qsum += q
            qsq += q * q
            n_vis += 1
    if n_vis >= 2:
        qmean = qsum / n_vis
        qvar = qsq / n_vis - qmean * qmean
        if qvar < 0.0:
            qvar = 0.0
    else:
        qvar = sd_prior * sd_prior
    explore_c = c_puct + c_log * math.log((visits[node] + c_base) / c_base)
    explore_c *= 1.0 + sd_scale * (math.sqrt(qvar) / sd_prior - 1.0)
    if explore_c < 0.0:
        explore_c = 0.0
    sqrt_n = math.sqrt(1.0 + visits[node] + vloss[node])
    best_slot = start
    best_score = -1e18
    for s in range(start, end):
        ch = cnode[s]
        vis = visits[ch]
        q = -vsum[ch] / vis if vis > 0 else pv - fpu
        score = q + explore_c * priors[ch] * sqrt_n / (1.0 + vis + vloss[ch])
        if score > best_score:
            best_score = score
            best_slot = s
    return best_slot

# 落子规则由 (kind, win_len) 两个整数选择（见 game._make_move_dispatch）。
# 早期版本把落子函数当参数传进来，虽然 Numba 支持「一等函数」，但磁盘缓存
# 会把函数类型写进索引再 pickle，Dispatcher 不可序列化 ⇒ cache=True 时保存
# 索引抛 ReferenceError（GUI 搜索线程会静默停掉）；关缓存又要每进程重编译
# 十几秒。改成整数分支后所有 numba 函数都能正常缓存。
@njit(cache=True)
def _select_np(board, st, go, fp, cap, hist,
               parent, cstart, ccount, cmove, cnode,
               visits, vsum, priors, vloss, expanded,
               root, c_puct, vl, n, path,
               c_log, c_base, sd_prior, sd_scale, fpu, kind, win_len):
    node = root
    path_len = 0
    path[path_len] = node
    path_len += 1
    vloss[node] += vl
    while expanded[node] == 1 and ccount[node] > 0:
        s = _select_child(cmove, cnode, cstart, ccount, visits, vsum, priors, vloss,
                          node, c_puct, c_log, c_base,
                          sd_prior, sd_scale, fpu)
        if s < 0:
            for i in range(path_len):
                vloss[path[i]] -= vl
            return -1, 0, path_len
        move = cmove[s]
        child = cnode[s]
        if move == n * n:
            r = -1
            c = -1
        else:
            r = move // n
            c = move % n
        if not _make_move_dispatch(board, st, go, fp, cap, hist, r, c, kind, win_len):
            # 超级劫非法：这个子节点是在另一条历史下展开的，当前历史里落不下去。
            # 不能继续往下走 —— 否则 vloss/visits 记在一个没发生的着法上，
            # 叶值也来自错误局面。撤掉本路径上的 virtual loss 后放弃这次模拟。
            for i in range(path_len):
                vloss[path[i]] -= vl
            return -1, 0, path_len
        vloss[child] += vl
        path[path_len] = child
        path_len += 1
        node = child
    if go[0]:
        return node, 1, path_len
    return node, 0, path_len

@njit(cache=True)
def _backup_np(board, st, go, fp, cap,
               parent, cstart, ccount, cmove, cnode,
               visits, vsum, priors, vloss, expanded,
               leaf, path, path_len, value, policy, has_policy,
               legal, legal_count,
               vl, n, next_id_ptr, cpool_ptr):
    for i in range(path_len - 1, -1, -1):
        node = path[i]
        vloss[node] -= vl
        visits[node] += 1
        vsum[node] += value
        value = -value
    if has_policy == 0 or expanded[leaf] != 0:
        return
    cstart[leaf] = cpool_ptr[0]
    if legal_count <= 0:
        ccount[leaf] = 0
        expanded[leaf] = 1
        return
    total_p = 0.0
    for j in range(legal_count):
        total_p += policy[legal[j]]
    # 全部合法着法都展开，不做候选裁剪：提前丢掉低先验点会漏掉「先验低但唯一正着」，
    # 而且裁剪要找「按质量降序的前 k 个」，是 O(合法点²)；直接全展开只要 O(合法点)。
    # 这里拿到的 legal 已经是调用方 _search_indices 过滤过的（自填单眼不在其中）。
    for j in range(legal_count):
        mv = legal[j]
        new_id = next_id_ptr[0]
        next_id_ptr[0] += 1
        parent[new_id] = leaf
        cstart[new_id] = -1
        ccount[new_id] = 0
        visits[new_id] = 0
        vsum[new_id] = 0.0
        if total_p > 0.0:
            priors[new_id] = policy[mv] / total_p
        else:
            priors[new_id] = 1.0 / legal_count
        vloss[new_id] = 0
        expanded[new_id] = 0
        slot = cpool_ptr[0]
        cpool_ptr[0] += 1
        cmove[slot] = mv
        cnode[slot] = new_id
    ccount[leaf] = legal_count
    expanded[leaf] = 1

@njit(cache=True)
def _compact(parent, cstart, ccount, cmove, cnode, visits, vsum, priors, vloss, expanded,
             root, n, cap, next_id_ptr, cpool_ptr,
             scratch_id, scratch_order, scratch_stack, scratch_cmove, scratch_cnode):
    new_id = scratch_id
    new_id[:] = -1
    order = scratch_order
    stack = scratch_stack
    new_id[root] = 0
    order[0] = root
    norder = 1
    top = 0
    stack[top] = root
    top += 1
    while top > 0:
        top -= 1
        node = stack[top]
        for s in range(cstart[node], cstart[node] + ccount[node]):
            ch = cnode[s]
            if new_id[ch] == -1:
                new_id[ch] = norder
                order[norder] = ch
                norder += 1
                stack[top] = ch
                top += 1
    t_cmove = scratch_cmove
    t_cnode = scratch_cnode
    t_cstart = np.zeros(norder, dtype=np.int32)
    t_ccount = np.zeros(norder, dtype=np.int32)
    t_parent = np.empty(norder, dtype=np.int32)
    t_visits = np.empty(norder, dtype=np.int32)
    t_vsum = np.empty(norder, dtype=np.float32)
    t_priors = np.empty(norder, dtype=np.float32)
    t_vloss = np.empty(norder, dtype=np.int32)
    t_expanded = np.empty(norder, dtype=np.int8)
    pool = 0
    for i in range(norder):
        node = order[i]
        t_parent[i] = parent[node]
        t_cstart[i] = pool
        t_ccount[i] = ccount[node]
        t_visits[i] = visits[node]
        t_vsum[i] = vsum[node]
        t_priors[i] = priors[node]
        t_vloss[i] = vloss[node]
        t_expanded[i] = expanded[node]
        for s in range(cstart[node], cstart[node] + ccount[node]):
            t_cmove[pool] = cmove[s]
            t_cnode[pool] = new_id[cnode[s]]
            pool += 1
    for i in range(norder):
        parent[i] = -1 if i == 0 else new_id[t_parent[i]]
        cstart[i] = t_cstart[i]
        ccount[i] = t_ccount[i]
        visits[i] = t_visits[i]
        vsum[i] = t_vsum[i]
        priors[i] = t_priors[i]
        vloss[i] = t_vloss[i]
        expanded[i] = t_expanded[i]
    for i in range(pool):
        cmove[i] = t_cmove[i]
        cnode[i] = t_cnode[i]
    next_id_ptr[0] = norder
    cpool_ptr[0] = pool

class _NodeView:
    __slots__ = ("m", "id")
    def __init__(self, m, nid):
        self.m = m
        self.id = nid
    @property
    def visit_count(self):
        return int(self.m._visits[self.id])
    @property
    def prior_prob(self):
        return float(self.m._priors[self.id])
    def get_value(self, default=0.0):
        v = self.m._visits[self.id]
        return default if v == 0 else float(self.m._vsum[self.id] / v)

class _RootView:
    __slots__ = ("m",)
    def __init__(self, m):
        self.m = m
    @property
    def visit_count(self):
        if self.m._root is None:
            return 0
        return int(self.m._visits[self.m._root])
    @property
    def children(self):
        m = self.m
        root = m._root
        n = m.board_size
        d = {}
        for s in range(m._cstart[root], m._cstart[root] + m._ccount[root]):
            mv = m._cmove[s]
            move = PASS_MOVE if mv == n * n else (mv // n, mv % n)
            d[move] = _NodeView(m, m._cnode[s])
        return d

class MCTS:
    def __init__(self, evaluator=None, model_path=MODEL_PATH, device="cuda",
                 c_puct=C_PUCT, num_simulations=NUM_SIMULATIONS,
                 temperature=TEMPERATURE, alpha=ALPHA,
                 dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_epsilon=DIRICHLET_EPSILON,
                 board_size=None, batch_size=BATCH_SIZE_MCTS, cap=MCTS_CAP,
                 fpu_reduction=FPU_REDUCTION,
                 virtual_loss=None, rules=None):
        if evaluator is None:
            from model import PolicyValueNet
            from evaluator import GoEvaluator
            model = PolicyValueNet.load_model(model_path, device=device)
            evaluator = GoEvaluator(model, device=device, max_batch=batch_size)
        self.rules = rules or GO_RULES
        self.move_kind = int(getattr(self.rules, "move_kind", 0))
        self.win_len = int(getattr(self.rules, "win_len", 0))
        # 「自填单眼」只对围棋成立：五子棋的点被己方围住也可能连五（move_kind=1）。
        self._prune_self_eye = self.move_kind == 0
        self.evaluator = evaluator
        self.device = evaluator.device
        self.board_size = int(board_size or evaluator.n)
        self.pass_idx = self.board_size * self.board_size
        self.max_moves = int(self.rules.max_moves(self.board_size))
        # 目差 tanh 尺度按可落子面积走（实心盘 1.6·边长）；init_root 会用
        # 局面的真实形状掩码再刷新一次，那里才知道洞的面积。
        self._util_scale = util_score_scale(self.pass_idx)
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss
        self.c_puct_log = CPUCT_LOG
        # 这两个在 _select_child 里当除数（base 进 log、prior 做标准差归一化），
        # 环境变量置 0 会让整棵树的选点静默退化（inf/NaN 之后永远选槽位第一个子）。
        self.c_puct_base = max(1.0, float(CPUCT_BASE))
        self.util_stdev_prior = max(1e-6, float(UTIL_STDEV_PRIOR))
        self.util_stdev_scale = UTIL_STDEV_SCALE
        self.fpu_reduction = fpu_reduction
        self._cap = int(cap)
        self.num_simulations = int(num_simulations)
        self.temperature = temperature
        self.alpha = alpha
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.batch_size = max(1, int(batch_size))
        self._noise_pending = False
        cap = self._cap
        self._parent = np.full(cap, -1, dtype=np.int32)
        self._cstart = np.zeros(cap, dtype=np.int32)
        self._ccount = np.zeros(cap, dtype=np.int32)
        self._visits = np.zeros(cap, dtype=np.int32)
        self._vsum = np.zeros(cap, dtype=np.float32)
        self._priors = np.zeros(cap, dtype=np.float32)
        self._vloss = np.zeros(cap, dtype=np.int32)
        self._expanded = np.zeros(cap, dtype=np.int8)
        self._cmove = np.zeros(cap, dtype=np.int32)
        self._cnode = np.zeros(cap, dtype=np.int32)
        self._next = np.zeros(1, dtype=np.int32)
        self._cpool = np.zeros(1, dtype=np.int32)
        # 自填单眼过滤的临时缓冲：每次展开前把 legal 里该剔的着法挑出去。
        self._eye_buf = np.zeros(self.pass_idx + 1, dtype=np.int32)
        self._path = np.zeros(max(2 * self.board_size * self.board_size + 4,
                                  self.max_moves + 8), dtype=np.int32)
        self._s_id = np.full(cap, -1, dtype=np.int32)
        self._s_order = np.zeros(cap, dtype=np.int32)
        self._s_stack = np.zeros(cap, dtype=np.int32)
        self._s_cmove = np.zeros(cap, dtype=np.int32)
        self._s_cnode = np.zeros(cap, dtype=np.int32)
        self._root = None
        self._root_key = None
        self.root = None

    @property
    def vloss(self):
        return VIRTUAL_LOSS if self.virtual_loss is None else self.virtual_loss

    @property
    def model(self):
        return self.evaluator.model

    def _idx(self, move):
        if move == PASS_MOVE or (move[0] == -1 and move[1] == -1):
            return self.pass_idx
        return move[0] * self.board_size + move[1]

    def _game_key(self, game):
        return (game.current_player, game.board.tobytes())

    def _utility(self, win_logit, score):
        return _utility(win_logit, score, self.alpha, self._util_scale)

    def _leaf_utility(self, win_logit, value, player, komi):
        """叶子值：胜率 + 归属头逐点和（换成含贴目的目差）。"""
        return self._utility(win_logit, margin_of(value, player, komi))

    def _terminal_value(self, game):
        """终局叶子价值：具体口径交给规则（go 有 PASS 结算，五子棋没有）。"""
        return self.rules.terminal_value(game, self._utility)

    def _search_indices(self, game, indices):
        """展开前剔除「自填单眼」，返回仍要展开的扁平下标。

        规则层不动：这些点仍合法（GUI 可落子、legal_moves 照常列出），
        只是 MCTS 不为它们建子节点 —— 区域规则下自填单眼最多等于白送一手。
        全部被剔掉时回退为原样：节点没有子节点会让这次模拟被丢弃，
        宁可多搜一个坏点也不能让树在这里断掉。
        """
        cnt = int(indices.size)
        if not self._prune_self_eye or cnt <= 0:
            return indices
        kept = _drop_self_eyes(game.board, self.board_size, indices, cnt,
                               int(game.current_player), self._eye_buf)
        if kept <= 0 or kept == cnt:
            return indices
        return self._eye_buf[:kept].copy()

    def _inference(self, state, legal_mask, global_features):
        p, v, w = self.evaluator.evaluate(
            state[None], legal_mask[None], global_features[None])
        return p[0], float(v[0]), float(w[0])

    def _batch_inference(self, states, legal_masks, global_features):
        p, v, w = self.evaluator.evaluate(
            np.stack(states), np.stack(legal_masks),
            np.stack(global_features))
        return list(p), v, w

    def init_root(self, game):
        self._root_key = self._game_key(game)
        self._noise_pending = True
        # 洞不落子 ⇒ 有效面积小于边长²，目差尺度得跟着实际面积走。
        self._util_scale = util_score_scale(shapes.playable_area(game.hole_mask))
        indices, legal_mask = game.get_legal_indices_and_mask()
        if indices.size == 0:
            self._root = None
            self.root = None
            return
        state, gfeat = game.get_state_bundle(legal_mask=legal_mask)
        policy, _v, _w = self._inference(state, legal_mask, gfeat)
        indices = self._search_indices(game, indices)
        n = self.board_size
        moves = [PASS_MOVE if i == self.pass_idx else (int(i) // n, int(i) % n)
                 for i in indices]
        probs = np.array([policy[self._idx(m)] for m in moves], dtype=np.float32)
        total = float(probs.sum())
        if total <= 0.0:
            probs = np.full(len(moves), 1.0 / len(moves), dtype=np.float32)
        else:
            probs /= total
        # 根节点保留全部合法着法（含 pass），唯一边界是 _search_indices 剔除的自填单眼。
        if self.dirichlet_epsilon > 0.0:
            probs = self._mix_dirichlet(probs)
        self._noise_pending = False
        self._parent[:] = -1
        self._cstart[:] = 0
        self._ccount[:] = 0
        self._visits[:] = 0
        self._vsum[:] = 0.0
        self._priors[:] = 0.0
        self._vloss[:] = 0
        self._expanded[:] = 0
        self._next[0] = 1
        self._cpool[0] = 0
        root = 0
        self._expanded[root] = 1
        self._ccount[root] = 0
        for i, m in enumerate(moves):
            new_id = self._next[0]
            self._next[0] += 1
            self._parent[new_id] = root
            self._cstart[new_id] = -1
            self._ccount[new_id] = 0
            self._visits[new_id] = 0
            self._vsum[new_id] = 0.0
            self._priors[new_id] = probs[i]
            self._vloss[new_id] = 0
            self._expanded[new_id] = 0
            slot = self._cpool[0]
            self._cpool[0] += 1
            self._cmove[slot] = self._idx(m)
            self._cnode[slot] = new_id
        self._ccount[root] = len(moves)
        self._root = root
        self.root = _RootView(self)

    def _mix_dirichlet(self, probs):
        """把 Dirichlet 噪声混进一组先验，并重新归一化。"""
        eps = float(self.dirichlet_epsilon)
        if eps <= 0.0 or len(probs) == 0:
            return probs
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(probs))
        mixed = (1.0 - eps) * probs + eps * noise
        total = float(mixed.sum())
        if total <= 0.0:
            return np.full(len(probs), 1.0 / len(probs), dtype=np.float32)
        return mixed / total

    def _add_dirichlet_noise(self):
        """给已存在的根子节点混一次噪声（用于树复用）。

        新根的噪声已在 init_root 里对全部合法着法加过；这里只处理
        update_root 复用出来的旧子树——它的子节点集合在展开时已固定。
        """
        if self._root is None or not self._noise_pending:
            return
        self._noise_pending = False
        if self.dirichlet_epsilon <= 0.0:
            return
        root = self._root
        cnt = self._ccount[root]
        if cnt == 0:
            return
        priors = np.empty(cnt, dtype=np.float32)
        i = 0
        for s in range(self._cstart[root], self._cstart[root] + cnt):
            priors[i] = self._priors[self._cnode[s]]
            i += 1
        priors = self._mix_dirichlet(priors)
        i = 0
        for s in range(self._cstart[root], self._cstart[root] + cnt):
            self._priors[self._cnode[s]] = priors[i]
            i += 1

    def mix_root_noise(self):
        self._add_dirichlet_noise()

    def update_root(self, game, last_move):
        self._util_scale = util_score_scale(shapes.playable_area(game.hole_mask))
        if self._root is None:
            self.init_root(game)
            return
        root = self._root
        mv = self._idx(last_move)
        found = -1
        for s in range(self._cstart[root], self._cstart[root] + self._ccount[root]):
            if self._cmove[s] == mv:
                found = self._cnode[s]
                break
        if found >= 0:
            # 保留树复用：复用根的子节点集合是上一轮展开时定下的（现在是全部合法着法），
            # 噪声只能在现有子节点上重分配；要重新生成根必须每手重建整棵树，代价太大。
            self._root = found
            self._root_key = self._game_key(game)
            self._noise_pending = True
            self._compact()
        else:
            self.init_root(game)
        self.root = _RootView(self) if self._root is not None else None

    def _compact(self):
        if self._root is None:
            return
        _compact(self._parent, self._cstart, self._ccount, self._cmove, self._cnode,
                 self._visits, self._vsum, self._priors, self._vloss, self._expanded,
                 self._root, self.board_size, len(self._parent), self._next, self._cpool,
                 self._s_id, self._s_order, self._s_stack, self._s_cmove, self._s_cnode)
        self._root = 0

    def reset(self):
        self._root = None
        self.root = None
        self._root_key = None

    def simulate_batch(self, game, n):
        """跑 n 次模拟，返回**实际跑掉的模拟数**（0 = 根不存在）。

        调用方（GUI 的搜索进度、自对弈的剩余模拟数）必须按返回值记账：
        根没建起来时这里什么都不做，若调用方照加 n，搜索会「瞬间完成」
        然后拿一棵空树去选点，AI 会直接 pass。
        """
        if self._root is None:
            return 0
        remaining = n
        while remaining > 0:
            if self._next[0] > self._cap - 20000:
                self._compact()
                if self._next[0] > self._cap - 20000:
                    self.init_root(game)
                    print(f"[MCTS] 树节点达到容量上限，已重建树根（剩余 {remaining} 模拟）")
                    self._add_dirichlet_noise()
                    if self._root is None:
                        return n - remaining
            batch = min(self.batch_size, remaining)
            sims = []
            for _ in range(batch):
                gc = game.copy()
                leaf, is_term, plen = _select_np(
                    gc.board, gc._st, gc._go, gc._fp, gc._cap, gc._hist,
                    self._parent, self._cstart, self._ccount, self._cmove, self._cnode,
                    self._visits, self._vsum, self._priors, self._vloss, self._expanded,
                    self._root, self.c_puct, self.vloss, self.board_size, self._path,
                    self.c_puct_log, self.c_puct_base,
                    self.util_stdev_prior, self.util_stdev_scale, self.fpu_reduction,
                    self.move_kind, self.win_len)
                if leaf >= 0:
                    legal_mask = None
                    # 终局也必须传数组而不是 None：Numba 会为这个调用点编译
                    # _backup_np 的签名，None 会在编译期变成 getitem(none, int64)。
                    legal_idx = np.empty(0, dtype=np.int32)
                    legal_count = 0
                    if not is_term:
                        # 快速路径：Numba 已经产出扁平整数下标，直接交给
                        # _backup_np；不要再转成 tuple 列表再转回来。
                        legal_idx, legal_mask = gc.get_legal_indices_and_mask()
                        legal_idx = self._search_indices(gc, legal_idx)
                        legal_count = int(legal_idx.size)
                    sims.append((leaf, gc, is_term, self._path[:plen].copy(), plen,
                                 legal_mask, legal_idx, legal_count))
            nn = [i for i, s in enumerate(sims) if not s[2]]
            policies = [None] * len(sims)
            values = [None] * len(sims)
            wls = [None] * len(sims)
            if nn:
                states = [sims[i][1].get_canonical_state(legal_mask=sims[i][5])
                          for i in nn]
                globs = [sims[i][1].get_global_features() for i in nn]
                masks = [sims[i][5] for i in nn]
                ps, vs, xw = self._batch_inference(states, masks, globs)
                for j, i in enumerate(nn):
                    policies[i] = ps[j]
                    values[i] = vs[j]
                    wls[i] = xw[j]
            for i, (leaf, gc, is_term, path, plen, legal_mask, legal_idx, legal_count) in enumerate(sims):
                if is_term:
                    value = self._terminal_value(gc)
                    policy = np.zeros(0, dtype=np.float32)
                    has_policy = 0
                else:
                    value = self._leaf_utility(wls[i], values[i], gc.current_player, gc.komi)
                    policy = policies[i]
                    has_policy = 1
                _backup_np(
                    gc.board, gc._st, gc._go, gc._fp, gc._cap,
                    self._parent, self._cstart, self._ccount, self._cmove, self._cnode,
                    self._visits, self._vsum, self._priors, self._vloss, self._expanded,
                    leaf, path, plen, value, policy, has_policy,
                    legal_idx, legal_count,
                    self.vloss, self.board_size, self._next, self._cpool)
            remaining -= batch
        return n - remaining

    def get_move_probs(self, game, temp=None, target_visits=1):
        if temp is None:
            temp = self.temperature
        if self._root is None or self._root_key != self._game_key(game):
            self.init_root(game)
        if self._root is None:
            return {}, {}
        self._add_dirichlet_noise()
        current = int(self._visits[self._root])
        remaining = max(1, target_visits - current)
        while remaining > 0:
            b = min(self.batch_size, remaining)
            self.simulate_batch(game, b)
            remaining -= b
        return self._get_visit_probs(temp)

    def _get_visit_probs(self, temp):
        root = self._root
        n = self.board_size
        move_visits = {}
        for s in range(self._cstart[root], self._cstart[root] + self._ccount[root]):
            mv = self._cmove[s]
            move = PASS_MOVE if mv == n * n else (mv // n, mv % n)
            move_visits[move] = int(self._visits[self._cnode[s]])
        moves = list(move_visits.keys())
        if not moves:
            # 根上没有子节点（棋盘下满且不允许 pass 的极端局面）：没有点可选，
            # 返回空字典让调用方走「无着法」分支，而不是 1/len(moves) 除零。
            return {}, {}
        visits = np.array([move_visits[m] for m in moves], dtype=np.float64)
        visits = np.maximum(visits, 0.0)
        total = visits.sum()
        if total <= 0.0:
            uniform = 1.0 / len(moves)
            probs = {m: uniform for m in moves}
            return probs, dict(probs)
        policy_probs = visits / total
        policy = {m: p for m, p in zip(moves, policy_probs)}
        best = moves[int(np.argmax(policy_probs))]
        if temp <= 0.1:
            return policy, {best: 1.0}
        sharpened = visits ** (1.0 / temp)
        sharpened /= sharpened.sum()
        return policy, {m: p for m, p in zip(moves, sharpened)}
