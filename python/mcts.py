"""围棋 MCTS：Numba 扁平树 + 批量神经网络评估。

本机单进程版本：推理器由 evaluator.GoEvaluator 提供，不再走 ONNX/TensorRT。
子节点展开时按策略概率质量裁剪候选（candidate_mass + min_candidates），
减少 19 路宽分支的树节点与选择开销；根节点裁剪时额外保留合法 pass，
子节点不额外保留；训练目标仍来自根访问计数。
"""
import math
import os

import numpy as np
from numba import njit

from game import PASS_MOVE, _make_move
from hyperparams import (BOARD_SIZE, C_PUCT, NUM_SIMULATIONS, TEMPERATURE,
                         DIRICHLET_ALPHA, DIRICHLET_EPSILON, VIRTUAL_LOSS,
                         BATCH_SIZE_MCTS, MCTS_CAP, OMP_NUM_THREADS, ALPHA,
                         CPUCT_LOG, CPUCT_BASE, UTIL_SCORE_SCALE,
                         UTIL_STDEV_PRIOR, UTIL_STDEV_SCALE, FPU_REDUCTION,
                         MODEL_PATH, CANDIDATE_MASS, MIN_CANDIDATES, board_rules)

os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_DYNAMIC"] = "FALSE"


def _utility(win_logit, score, alpha):
    return ((1.0 - alpha) * math.tanh(win_logit)
            + alpha * math.tanh(score / UTIL_SCORE_SCALE))


_HAS_NUMBA = True


if _HAS_NUMBA:
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

    @njit(cache=True)
    def _select_np(board, st, go, fp, cap,
                   parent, cstart, ccount, cmove, cnode,
                   visits, vsum, priors, vloss, expanded,
                   root, c_puct, vl, n, path,
                   c_log, c_base, sd_prior, sd_scale, fpu):
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
            _make_move(board, st, go, fp, r, c, cap)
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
                   sel_moves, sel_probs, sel_used, candidate_mass, min_candidates,
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
        k = 0
        if total_p <= 0.0:
            for j in range(legal_count):
                sel_moves[j] = legal[j]
                sel_probs[j] = 1.0 / legal_count
            k = legal_count
            total_p = 1.0
        else:
            for j in range(legal_count):
                sel_used[j] = 0
            target = total_p * candidate_mass
            min_k = min_candidates
            if min_k > legal_count:
                min_k = legal_count
            if min_k < 1:
                min_k = 1
            cumulative = 0.0
            while k < legal_count and (k < min_k or cumulative < target):
                best_j = -1
                best_p = -1.0
                for j in range(legal_count):
                    if sel_used[j] == 0:
                        p = policy[legal[j]]
                        if p > best_p:
                            best_p = p
                            best_j = j
                if best_j < 0:
                    break
                sel_used[best_j] = 1
                sel_moves[k] = legal[best_j]
                sel_probs[k] = best_p
                cumulative += best_p
                k += 1
            if k <= 0:
                sel_moves[0] = legal[0]
                sel_probs[0] = 1.0 / legal_count
                cumulative = sel_probs[0]
                k = 1
            total_p = cumulative
        for j in range(k):
            mv = sel_moves[j]
            new_id = next_id_ptr[0]
            next_id_ptr[0] += 1
            parent[new_id] = leaf
            cstart[new_id] = -1
            ccount[new_id] = 0
            visits[new_id] = 0
            vsum[new_id] = 0.0
            priors[new_id] = sel_probs[j] / total_p
            vloss[new_id] = 0
            expanded[new_id] = 0
            slot = cpool_ptr[0]
            cpool_ptr[0] += 1
            cmove[slot] = mv
            cnode[slot] = new_id
        ccount[leaf] = k
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
                     virtual_loss=None, candidate_mass=CANDIDATE_MASS,
                     min_candidates=MIN_CANDIDATES):
            if evaluator is None:
                from model import PolicyValueNet
                from evaluator import GoEvaluator
                model = PolicyValueNet.load_model(model_path, device=device)
                evaluator = GoEvaluator(model, device=device, max_batch=batch_size)
            self.evaluator = evaluator
            self.device = evaluator.device
            self.board_size = int(board_size or evaluator.n)
            self.pass_idx = self.board_size * self.board_size
            self.max_moves = int(board_rules(self.board_size)["max_moves"])
            self.c_puct = c_puct
            self.virtual_loss = virtual_loss
            self.c_puct_log = CPUCT_LOG
            self.c_puct_base = CPUCT_BASE
            self.util_stdev_prior = UTIL_STDEV_PRIOR
            self.util_stdev_scale = UTIL_STDEV_SCALE
            self.fpu_reduction = fpu_reduction
            self._cap = int(cap)
            self.num_simulations = int(num_simulations)
            self.temperature = temperature
            self.alpha = alpha
            self.dirichlet_alpha = dirichlet_alpha
            self.dirichlet_epsilon = dirichlet_epsilon
            self.candidate_mass = float(candidate_mass)
            self.min_candidates = int(min_candidates)
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
            self._path = np.zeros(max(2 * self.board_size * self.board_size + 4,
                                      self.max_moves + 8), dtype=np.int32)
            self._sel_moves = np.zeros(self.board_size * self.board_size + 1, dtype=np.int32)
            self._sel_probs = np.zeros(self.board_size * self.board_size + 1, dtype=np.float32)
            self._sel_used = np.zeros(self.board_size * self.board_size + 1, dtype=np.uint8)
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
            return _utility(win_logit, score, self.alpha)

        def _inference(self, state, legal_mask):
            p, v, _w = self.evaluator.evaluate(state[None], legal_mask[None])
            return p[0], float(v[0])

        def _batch_inference(self, states, legal_masks):
            p, v, w = self.evaluator.evaluate(np.stack(states), np.stack(legal_masks))
            return list(p), v, w

        def init_root(self, game):
            self._root_key = self._game_key(game)
            self._noise_pending = True
            legal_moves, legal_mask = game.get_legal_moves_and_mask()
            if not legal_moves:
                self._root = None
                self.root = None
                return
            state = game.get_canonical_state(legal_mask=legal_mask)
            policy, _ = self._inference(state, legal_mask)
            moves = list(legal_moves)
            probs = np.array([policy[self._idx(m)] for m in moves], dtype=np.float32)
            total = float(probs.sum())
            if total <= 0.0:
                probs = np.full(len(moves), 1.0 / len(moves), dtype=np.float32)
            else:
                probs /= total
            # 先在**完整合法着法**上加根噪声，再按候选质量裁剪。
            # 反过来做会让噪声只能重分配已入选的高先验点，低先验点永远进不了候选。
            if self.dirichlet_epsilon > 0.0:
                probs = self._mix_dirichlet(probs)
            self._noise_pending = False
            # 根节点也按候选质量裁剪；否则 GUI 里会看到「候选质量 0.1 仍搜索几十个点」。
            # pass 永远保留：它是终局判断的一部分，不能被概率质量裁掉。
            if self.candidate_mass < 1.0 and len(moves) > 0:
                order = np.argsort(-probs)
                cum = np.cumsum(probs[order])
                k = int(np.searchsorted(cum, self.candidate_mass, side="left") + 1)
                k = max(self.min_candidates, min(k, len(moves)))
                keep = list(order[:k])
                for i, m in enumerate(moves):
                    if m == PASS_MOVE and i not in keep:
                        keep.append(i)
                idx = np.array(keep, dtype=np.int64)
                moves = [moves[i] for i in idx]
                probs = probs[idx]
                probs /= probs.sum()
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

            新根的完整合法着法噪声已在 init_root 里、候选裁剪之前加过；
            这里只处理 update_root 复用出来的旧子树——它的候选集在之前
            展开时已经确定，无法再引入候选集之外的点。
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
                # 保留树复用。复用根的候选集在上一轮展开时已固定，
                # 因此噪声只能在现有候选内重分配；要严格“全合法先噪声再裁剪”
                # 必须每手重建根，性能代价不可接受。
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
            if self._root is None:
                return
            remaining = n
            while remaining > 0:
                if self._next[0] > self._cap - 20000:
                    self._compact()
                    if self._next[0] > self._cap - 20000:
                        self.init_root(game)
                        print(f"[MCTS] 树节点达到容量上限，已重建树根（剩余 {remaining} 模拟）")
                        self._add_dirichlet_noise()
                        if self._root is None:
                            return
                batch = min(self.batch_size, remaining)
                sims = []
                for _ in range(batch):
                    gc = game.copy()
                    leaf, is_term, plen = _select_np(
                        gc.board, gc._st, gc._go, gc._fp, gc._cap,
                        self._parent, self._cstart, self._ccount, self._cmove, self._cnode,
                        self._visits, self._vsum, self._priors, self._vloss, self._expanded,
                        self._root, self.c_puct, self.vloss, self.board_size, self._path,
                        self.c_puct_log, self.c_puct_base,
                        self.util_stdev_prior, self.util_stdev_scale, self.fpu_reduction)
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
                            legal_count = int(legal_idx.size)
                        sims.append((leaf, gc, is_term, self._path[:plen].copy(), plen,
                                     legal_mask, legal_idx, legal_count))
                nn = [i for i, s in enumerate(sims) if not s[2]]
                policies = [None] * len(sims)
                values = [None] * len(sims)
                wls = [None] * len(sims)
                if nn:
                    states = [sims[i][1].get_canonical_state(legal_mask=sims[i][5]) for i in nn]
                    masks = [sims[i][5] for i in nn]
                    ps, vs, xw = self._batch_inference(states, masks)
                    for j, i in enumerate(nn):
                        policies[i] = ps[j]
                        values[i] = vs[j]
                        wls[i] = xw[j]
                for i, (leaf, gc, is_term, path, plen, legal_mask, legal_idx, legal_count) in enumerate(sims):
                    if is_term:
                        if gc.winner is None:
                            gc._compute_winner()
                        if gc.winner == 0:
                            value = 0.0
                        else:
                            term_score = gc.final_points * gc.current_player
                            term_wl = 6.0 if gc.winner == gc.current_player else -6.0
                            value = self._utility(term_wl, term_score)
                        policy = np.zeros(0, dtype=np.float32)
                        has_policy = 0
                    else:
                        value = values[i]
                        policy = policies[i]
                        has_policy = 1
                        score = value - gc.komi * gc.current_player
                        value = self._utility(wls[i], score)
                    _backup_np(
                        gc.board, gc._st, gc._go, gc._fp, gc._cap,
                        self._parent, self._cstart, self._ccount, self._cmove, self._cnode,
                        self._visits, self._vsum, self._priors, self._vloss, self._expanded,
                        leaf, path, plen, value, policy, has_policy,
                        legal_idx, legal_count,
                        self._sel_moves, self._sel_probs, self._sel_used,
                        self.candidate_mass, self.min_candidates,
                        self.vloss, self.board_size, self._next, self._cpool)
                remaining -= batch

        def probe_pass(self, game):
            """确保合法 pass 至少被搜索一次。

            终局附近 pass 的先验通常很低，低于其它高先验着法的探索项；
            如果从不访问 pass，Q 永远是「父均值−FPU」，搜索可能漏掉「我 pass 就赢」。
            这里临时把根节点的先验全部压到 pass 上跑 1 次模拟，拿到真实终局/叶值后
            立刻恢复原先验；后续选点仍按正常 PUCT 走。
            """
            if self._root is None:
                return
            root = self._root
            slot = -1
            child = -1
            for s in range(self._cstart[root], self._cstart[root] + self._ccount[root]):
                if self._cmove[s] == self.pass_idx:
                    slot = s
                    child = self._cnode[s]
                    break
            if child < 0 or self._visits[child] > 0:
                return
            start = self._cstart[root]
            end = start + self._ccount[root]
            saved = self._priors[start:end].copy()
            try:
                self._priors[start:end] = 0.0
                self._priors[child] = 1.0
                self.simulate_batch(game, 1)
            finally:
                self._priors[start:end] = saved

        def get_move_probs(self, game, temp=None, target_visits=1):
            if temp is None:
                temp = self.temperature
            if self._root is None or self._root_key != self._game_key(game):
                self.init_root(game)
            if self._root is None:
                return {}, {}
            self._add_dirichlet_noise()
            self.probe_pass(game)
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
