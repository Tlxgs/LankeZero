"""
蒙特卡洛树搜索
"""
import numpy as np
import math
import torch
import torch.nn.functional as F
import os
from game import PASS_MOVE
from model import PolicyValueNet

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_DYNAMIC"] = "FALSE"

C_PUCT = 1.5
NUM_SIMULATIONS = 120
TEMPERATURE = 0.0
DIRICHLET_ALPHA = 0.2
DIRICHLET_EPSILON = 0.30
VIRTUAL_LOSS = 3
BATCH_SIZE_MCTS = 16
BOARD_SIZE = 13


# ============================================================
# 推理部分公共基类（torch 加载 + 单/批量推理 + pass bonus）
# ============================================================
class _BaseMCTS:
    def __init__(self, c_puct=C_PUCT, num_simulations=NUM_SIMULATIONS,
                 temperature=TEMPERATURE, onnx_path='model.onnx', device='cpu',
                 dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_epsilon=DIRICHLET_EPSILON,
                 board_size=BOARD_SIZE, batch_size=BATCH_SIZE_MCTS):
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.board_size = board_size
        self.batch_size = batch_size
        self.pass_idx = board_size * board_size

        model_path = onnx_path.replace('.onnx', '.pt')
        if not os.path.exists(model_path):
            model_path = 'model.pt'
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = PolicyValueNet.load_model(model_path, device=self.device)
        self.model.eval()

    def _inference(self, state, legal_mask):
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            logits, value, _ = self.model(t)
            policy = self._process(logits[0], legal_mask, value[0, 0].item())
        return policy, value[0, 0].item()

    def _batch_inference(self, states, legal_masks):
        n = len(states)
        with torch.no_grad():
            batch = torch.from_numpy(np.stack(states)).to(self.device)
            mask_t = torch.from_numpy(np.stack(legal_masks)).to(self.device)
            logits, values, _ = self.model(batch)
            logits = logits.masked_fill(mask_t == 0, -1e4)
            probs = F.softmax(logits, dim=1)
            p = probs.cpu().numpy()
            v = values[:, 0].cpu().numpy()
        for i in range(n):
            self._pass_bonus(p[i], legal_masks[i], v[i])
        return list(p), v

    def _process(self, logits, legal_mask, value):
        arr = logits.cpu().numpy().copy()
        if legal_mask is not None:
            arr[legal_mask == 0] = -1e8
        arr -= arr.max()
        arr = np.exp(arr)
        arr /= arr.sum()
        self._pass_bonus(arr, legal_mask, value)
        return arr

    @staticmethod
    def _pass_bonus(policy, legal_mask, value):
        pidx = BOARD_SIZE * BOARD_SIZE
        if legal_mask is None or legal_mask[pidx] != 1:
            return
        p = policy[pidx]
        if p < 0.01:
            p += 0.01
        if value > 0.9:
            p += 2 * value - 1.8
        if value < -0.9:
            p += 0.02
        policy[pidx] = p
        policy /= policy.sum()


try:
    from numba import njit
    from game import _make_move, _legal_moves_and_mask, _HAS_NUMBA as _GAME_HAS_NUMBA
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

if _HAS_NUMBA and _GAME_HAS_NUMBA:
    _CAP = 1000000

    @njit(cache=True)
    def _select_child(cmove, cnode, cstart, ccount, visits, vsum, priors, vloss,
                      node, c_puct, move_count, n):
        start = cstart[node]
        end = start + ccount[node]
        if start == end:
            return -1
        if visits[node] > 0:
            pv = vsum[node] / visits[node]
        else:
            pv = 0.0
        EXPLORE=0.02
        optimism = EXPLORE - 0.1 * math.log(1.00001 + pv)
        sqrt_n = math.sqrt(1.0 + visits[node] + vloss[node])
        late = move_count > 150
        b = 0.001 * (move_count - 150) if late else 0.0
        pass_move = n * n
        best_slot = start
        best_score = -1e18
        for s in range(start, end):
            ch = cnode[s]
            vis = visits[ch]
            if vis == 0:
                q = pv + optimism
            else:
                q = -vsum[ch] / vis + optimism / (1.0 + vis)
            score = q + c_puct * priors[ch] * sqrt_n / (1.0 + vis + vloss[ch])
            if late and cmove[s] == pass_move:
                score += b
            if score > best_score:
                best_score = score
                best_slot = s
        return best_slot

    # ---------- 选择路径（等价原 _simulate_select）----------
    @njit(cache=True)
    def _select_np(board, st, go, fp, cap,
                   parent, cstart, ccount, cmove, cnode,
                   visits, vsum, priors, vloss, expanded,
                   root, c_puct, vl, n, path):
        node = root
        path_len = 0
        path[path_len] = node
        path_len += 1
        vloss[node] += vl
        while expanded[node] == 1 and ccount[node] > 0:
            s = _select_child(cmove, cnode, cstart, ccount, visits, vsum, priors, vloss,
                              node, c_puct, st[5], n)
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
            return node, 1, path_len      # 终局
        return node, 0, path_len          # 叶节点（待展开）

    # ---------- 回溯+展开（等价原 _simulate_backup）----------
    @njit(cache=True)
    def _backup_np(board, st, go, fp, cap,
                   parent, cstart, ccount, cmove, cnode,
                   visits, vsum, priors, vloss, expanded,
                   leaf, path, path_len, value, policy, has_policy,
                   legal, legal_cap, vl, n, next_id_ptr, cpool_ptr):
        for i in range(path_len - 1, -1, -1):
            node = path[i]
            vloss[node] -= vl
            visits[node] += 1
            vsum[node] += value
            value = -value
        if has_policy and expanded[leaf] == 0:
            legal_len = _legal_moves_and_mask(board, st, n, legal, legal_cap)
            cstart[leaf] = cpool_ptr[0]
            for i in range(legal_len):
                mv = legal[i]
                new_id = next_id_ptr[0]
                next_id_ptr[0] += 1
                parent[new_id] = leaf
                cstart[new_id] = -1
                ccount[new_id] = 0
                visits[new_id] = 0
                vsum[new_id] = 0.0
                priors[new_id] = policy[mv]
                vloss[new_id] = 0
                expanded[new_id] = 0
                slot = cpool_ptr[0]
                cpool_ptr[0] += 1
                cmove[slot] = mv
                cnode[slot] = new_id
            ccount[leaf] = legal_len
            expanded[leaf] = 1

    # ---------- 压缩：把当前根子树重排到数组头部，回收死节点 ----------
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

    # ---------- 供 gui 读取的轻量视图（只读数组树，不复制）----------
    class _NodeView:
        __slots__ = ('m', 'id')
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
        __slots__ = ('m',)
        def __init__(self, m):
            self.m = m
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

    class MCTS(_BaseMCTS):
        def __init__(self, c_puct=C_PUCT, num_simulations=NUM_SIMULATIONS,
                     temperature=TEMPERATURE, onnx_path='model.onnx', device='cpu',
                     dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_epsilon=DIRICHLET_EPSILON,
                     board_size=BOARD_SIZE, batch_size=BATCH_SIZE_MCTS):
            super().__init__(c_puct, num_simulations, temperature, onnx_path, device,
                             dirichlet_alpha, dirichlet_epsilon, board_size, batch_size)
            CAP = _CAP
            self._parent = np.full(CAP, -1, dtype=np.int32)
            self._cstart = np.zeros(CAP, dtype=np.int32)
            self._ccount = np.zeros(CAP, dtype=np.int32)
            self._visits = np.zeros(CAP, dtype=np.int32)
            self._vsum = np.zeros(CAP, dtype=np.float32)
            self._priors = np.zeros(CAP, dtype=np.float32)
            self._vloss = np.zeros(CAP, dtype=np.int32)
            self._expanded = np.zeros(CAP, dtype=np.int8)
            self._cmove = np.zeros(CAP, dtype=np.int32)
            self._cnode = np.zeros(CAP, dtype=np.int32)
            self._next = np.zeros(1, dtype=np.int32)
            self._cpool = np.zeros(1, dtype=np.int32)
            self._path = np.zeros(2 * self.board_size * self.board_size + 4, dtype=np.int32)
            self._legal = np.zeros(self.board_size * self.board_size + 1, dtype=np.int32)
            self._mask = np.zeros(self.board_size * self.board_size + 1, dtype=np.float32)
            self._s_id = np.full(CAP, -1, dtype=np.int32)
            self._s_order = np.zeros(CAP, dtype=np.int32)
            self._s_stack = np.zeros(CAP, dtype=np.int32)
            self._s_cmove = np.zeros(CAP, dtype=np.int32)
            self._s_cnode = np.zeros(CAP, dtype=np.int32)
            self._root = None
            self.root = None

        def _idx(self, move):
            if move == PASS_MOVE or (move[0] == -1 and move[1] == -1):
                return self.pass_idx
            return move[0] * self.board_size + move[1]

        def init_root(self, game):
            legal_moves, legal_mask = game.get_legal_moves_and_mask()
            if not legal_moves:
                self._root = None
                self.root = None
                return
            state = game.get_canonical_state()
            policy, _ = self._inference(state, legal_mask)
            moves = list(legal_moves)
            probs = np.array([policy[self._idx(m)] for m in moves], dtype=np.float32)
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
            mixed = (1 - self.dirichlet_epsilon) * probs + self.dirichlet_epsilon * noise
            mixed /= mixed.sum()
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
                self._priors[new_id] = mixed[i]
                self._vloss[new_id] = 0
                self._expanded[new_id] = 0
                slot = self._cpool[0]
                self._cpool[0] += 1
                self._cmove[slot] = self._idx(m)
                self._cnode[slot] = new_id
            self._ccount[root] = len(moves)
            self._root = root
            self.root = _RootView(self)

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
                self._root = found
                self._compact()          # 每步压缩一次，回收死分支，保持容量
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
        def simulate_batch(self, game, n):
            if self._root is None:
                return
            if self._next[0] > _CAP - 20000:     # 安全网（正常每步已压缩，不会触发）
                self._compact()
            remaining = n
            while remaining > 0:
                batch = min(self.batch_size, remaining)
                sims = []
                for _ in range(batch):
                    gc = game.copy()
                    leaf, is_term, plen = _select_np(
                        gc.board, gc._st, gc._go, gc._fp, gc._cap,
                        self._parent, self._cstart, self._ccount, self._cmove, self._cnode,
                        self._visits, self._vsum, self._priors, self._vloss, self._expanded,
                        self._root, self.c_puct, VIRTUAL_LOSS, self.board_size, self._path)
                    if leaf >= 0:
                        sims.append((leaf, gc, is_term, self._path[:plen].copy(), plen))
                nn = [i for i, s in enumerate(sims) if not s[2]]
                policies = [None] * len(sims)
                values = [None] * len(sims)
                if nn:
                    states = [sims[i][1].get_canonical_state() for i in nn]
                    masks = [sims[i][1].get_legal_moves_and_mask()[1] for i in nn]
                    ps, vs = self._batch_inference(states, masks)
                    for j, i in enumerate(nn):
                        policies[i] = ps[j]
                        values[i] = vs[j]
                for i, (leaf, gc, is_term, path, plen) in enumerate(sims):
                    if is_term:
                        if gc.winner is None:
                            gc._compute_winner()
                        value = 0.0 if gc.winner == 0 else gc.get_terminal_value(gc.current_player)
                        policy = np.zeros(0, dtype=np.float32)
                        has_policy = 0
                    else:
                        value = values[i]
                        policy = policies[i]
                        has_policy = 1
                    _backup_np(
                        gc.board, gc._st, gc._go, gc._fp, gc._cap,
                        self._parent, self._cstart, self._ccount, self._cmove, self._cnode,
                        self._visits, self._vsum, self._priors, self._vloss, self._expanded,
                        leaf, path, plen, value, policy, has_policy,
                        self._legal, self._mask, VIRTUAL_LOSS, self.board_size,
                        self._next, self._cpool)
                remaining -= batch

        def get_move_probs(self, game, temp=None, target_visits=1):
            if temp is None:
                temp = self.temperature
            if self._root is None:
                self.init_root(game)
            if self._root is None:
                return {}
            current = int(self._visits[self._root])
            remaining = max(1, target_visits - current)
            while remaining > 0:
                b = min(self.batch_size, remaining)
                self.simulate_batch(game, b)
                remaining -= b
            return self._get_visit_probs(temp)[0]

        def _get_visit_probs(self, temp):
            root = self._root
            n = self.board_size
            move_visits = {}
            for s in range(self._cstart[root], self._cstart[root] + self._ccount[root]):
                mv = self._cmove[s]
                move = PASS_MOVE if mv == n * n else (mv // n, mv % n)
                move_visits[move] = int(self._visits[self._cnode[s]])
            if temp <= 0.1:
                best = max(move_visits, key=move_visits.get)
                return {best: 1.0}, best
            moves = list(move_visits.keys())
            visits = np.array([move_visits[m] for m in moves])
            probs = visits ** (1.0 / temp)
            probs /= probs.sum()
            return {m: p for m, p in zip(moves, probs)}, moves[np.argmax(probs)]
