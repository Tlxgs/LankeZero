"""
围棋游戏逻辑
"""
import numpy as np
from copy import deepcopy
import math
from numba import njit
# ==================== 超参数 ====================
BOARD_SIZE = 13
KOMI = 7.5          # 贴目（白方补偿）
MAX_MOVES = 200     # 最大步数防止死循环
PASS_MOVE = (-1, -1)
MIN_MOVES_BEFORE_PASS = 110   # 前n手不能Pass
SCALE = 5
PASS_LIMIT = 6
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
@njit(cache=True)
def _group_has_liberty(board, r0, c0, n):
    """整块是否还有气（自杀判定）"""
    color = board[r0, c0]
    if color == 0:
        return False
    visited = np.zeros((n, n), dtype=np.bool_)
    stack = np.zeros(n * n, dtype=np.int32)
    top = 0
    stack[top] = r0 * n + c0
    top += 1
    visited[r0, c0] = True
    while top > 0:
        top -= 1
        cur = stack[top]
        r = cur // n
        c = cur % n
        if r > 0:
            if board[r - 1, c] == 0:
                return True
            if board[r - 1, c] == color and not visited[r - 1, c]:
                visited[r - 1, c] = True
                stack[top] = (r - 1) * n + c
                top += 1
        if r < n - 1:
            if board[r + 1, c] == 0:
                return True
            if board[r + 1, c] == color and not visited[r + 1, c]:
                visited[r + 1, c] = True
                stack[top] = (r + 1) * n + c
                top += 1
        if c > 0:
            if board[r, c - 1] == 0:
                return True
            if board[r, c - 1] == color and not visited[r, c - 1]:
                visited[r, c - 1] = True
                stack[top] = r * n + (c - 1)
                top += 1
        if c < n - 1:
            if board[r, c + 1] == 0:
                return True
            if board[r, c + 1] == color and not visited[r, c + 1]:
                visited[r, c + 1] = True
                stack[top] = r * n + (c + 1)
                top += 1
    return False

@njit(cache=True)
def _capture_opponent(board, row, col, player, n, cap):
    """提掉无气对方棋块；返回被提子数，坐标写入 cap[0:count]（扁平索引）"""
    opp = -player
    count = 0
    processed = np.zeros((n, n), dtype=np.bool_)
    stack = np.zeros(n * n, dtype=np.int32)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = row + dr
        nc = col + dc
        if not (0 <= nr < n and 0 <= nc < n):
            continue
        if board[nr, nc] == opp and not processed[nr, nc]:
            top = 0
            stack[top] = nr * n + nc
            top += 1
            processed[nr, nc] = True
            block_start = count
            while top > 0:
                top -= 1
                cur = stack[top]
                cap[count] = cur
                count += 1
                r = cur // n
                c = cur % n
                if r > 0 and board[r - 1, c] == opp and not processed[r - 1, c]:
                    processed[r - 1, c] = True
                    stack[top] = (r - 1) * n + c
                    top += 1
                if r < n - 1 and board[r + 1, c] == opp and not processed[r + 1, c]:
                    processed[r + 1, c] = True
                    stack[top] = (r + 1) * n + c
                    top += 1
                if c > 0 and board[r, c - 1] == opp and not processed[r, c - 1]:
                    processed[r, c - 1] = True
                    stack[top] = r * n + (c - 1)
                    top += 1
                if c < n - 1 and board[r, c + 1] == opp and not processed[r, c + 1]:
                    processed[r, c + 1] = True
                    stack[top] = r * n + (c + 1)
                    top += 1
            # 检查整块是否有气；有气则回滚（不提取）
            has_air = False
            for j in range(block_start, count):
                cur2 = cap[j]
                r = cur2 // n
                c = cur2 % n
                if r > 0 and board[r - 1, c] == 0:
                    has_air = True
                    break
                if r < n - 1 and board[r + 1, c] == 0:
                    has_air = True
                    break
                if c > 0 and board[r, c - 1] == 0:
                    has_air = True
                    break
                if c < n - 1 and board[r, c + 1] == 0:
                    has_air = True
                    break
            if has_air:
                count = block_start
    return count

@njit(cache=True)
def _is_legal_impl(board, row, col, player, ko_r, ko_c, n):
    """等价 Python _is_legal_move（临时落子后恢复，不污染棋盘）"""
    if board[row, col] != 0:
        return False
    board[row, col] = player
    cap = np.zeros(n * n, dtype=np.int32)
    captured = _capture_opponent(board, row, col, player, n, cap)
    board[row, col] = 0
    if ko_r != -2 and row == ko_r and col == ko_c and captured > 0:
        return False                              # 劫争
    if captured == 0:
        board[row, col] = player
        ok = _group_has_liberty(board, row, col, n)
        board[row, col] = 0
        return ok                                 # 自杀判定
    return True

@njit(cache=True)
def _make_move(board, st, go, fp, row, col, cap):
    """落子（含 Pass/提子/劫检测）。
    st: [0]当前玩家 [1]上一步r [2]上一步c [3]劫r [4]劫c
        [5]步数 [6]连续pass [7]黑总pass [8]白总pass [9]winner(-2=None) [10]min_moves"""
    n = board.shape[0]
    if row == -1 and col == -1:                   # Pass
        st[1] = -1
        st[2] = -1
        if st[0] == 1:
            st[7] += 1
        else:
            st[8] += 1
        st[6] += 1
        st[5] += 1
        st[0] = -st[0]
        if st[6] >= 2:
            go[0] = True
        return True
    board[row, col] = st[0]
    captured = _capture_opponent(board, row, col, st[0], n, cap)
    if captured > 0:
        for j in range(captured):
            ci = cap[j]
            board[ci // n, ci % n] = 0
        if captured == 1:                         # 劫检测（等价 Python 逻辑）
            ko_r = cap[0] // n
            ko_c = cap[0] % n
            board[ko_r, ko_c] = -st[0]
            c2 = _capture_opponent(board, ko_r, ko_c, -st[0], n, cap)
            board[ko_r, ko_c] = 0
            if c2 == 1 and cap[0] == row * n + col:
                st[3] = ko_r
                st[4] = ko_c
            else:
                st[3] = -2
                st[4] = -2
        else:
            st[3] = -2
            st[4] = -2
    else:
        st[3] = -2
        st[4] = -2
    st[1] = row
    st[2] = col
    st[6] = 0
    st[5] += 1
    if st[5] >= MAX_MOVES:
        go[0] = True
    else:
        st[0] = -st[0]
    return True

@njit(cache=True)
def _legal_moves_and_mask(board, st, n, moves, mask):
    """合法着法；moves: int32缓冲(n*n+1)，mask: float32(n*n+1)"""
    total = n * n
    mask[:] = 0.0
    count = 0
    if st[0] == 1:
        forced = st[8] >= PASS_LIMIT
    else:
        forced = st[7] >= PASS_LIMIT
    if forced:                                    # 强制 pass
        moves[count] = total
        count += 1
        mask[total] = 1.0
        return count
    for r in range(n):
        for c in range(n):
            if board[r, c] != 0:
                continue
            i = r * n + c
            has_air = False
            if r > 0 and board[r - 1, c] == 0:
                has_air = True
            elif r < n - 1 and board[r + 1, c] == 0:
                has_air = True
            elif c > 0 and board[r, c - 1] == 0:
                has_air = True
            elif c < n - 1 and board[r, c + 1] == 0:
                has_air = True
            if has_air:                           # 快速路径：邻格为空
                moves[count] = i
                count += 1
                mask[i] = 1.0
                continue
            if _is_legal_impl(board, r, c, st[0], st[3], st[4], n):
                moves[count] = i
                count += 1
                mask[i] = 1.0
    if st[5] >= st[10]:
        moves[count] = total
        count += 1
        mask[total] = 1.0
    return count
@njit(cache=True)
def _liberty_channels(board, n, player, own_out, opp_out):
    """己方/对方棋块气数通道：气数/4，上限1；空点或非本颜色为0。"""
    labels = np.zeros((n, n), dtype=np.int32)      # 0=未访问/空，>0=组号
    lib_seen = np.zeros(n * n, dtype=np.int32)     # 印记法去重（存组号）
    lib = np.zeros(n * n + 1, dtype=np.int32)      # 每组的独立气数
    stack = np.zeros(n * n, dtype=np.int32)
    next_gid = 1
    for r in range(n):
        for c in range(n):
            color = board[r, c]
            if color == 0 or labels[r, c] != 0:
                continue
            gid = next_gid
            next_gid += 1
            top = 0
            stack[top] = r * n + c
            top += 1
            labels[r, c] = gid
            while top > 0:
                top -= 1
                cur = stack[top]
                cr = cur // n
                cc = cur % n
                # 上
                if cr > 0:
                    nb = board[cr - 1, cc]
                    if nb == 0:
                        pos = (cr - 1) * n + cc
                        if lib_seen[pos] != gid:
                            lib_seen[pos] = gid
                            lib[gid] += 1
                    elif nb == color and labels[cr - 1, cc] == 0:
                        labels[cr - 1, cc] = gid
                        stack[top] = (cr - 1) * n + cc
                        top += 1
                # 下
                if cr < n - 1:
                    nb = board[cr + 1, cc]
                    if nb == 0:
                        pos = (cr + 1) * n + cc
                        if lib_seen[pos] != gid:
                            lib_seen[pos] = gid
                            lib[gid] += 1
                    elif nb == color and labels[cr + 1, cc] == 0:
                        labels[cr + 1, cc] = gid
                        stack[top] = (cr + 1) * n + cc
                        top += 1
                # 左
                if cc > 0:
                    nb = board[cr, cc - 1]
                    if nb == 0:
                        pos = cr * n + (cc - 1)
                        if lib_seen[pos] != gid:
                            lib_seen[pos] = gid
                            lib[gid] += 1
                    elif nb == color and labels[cr, cc - 1] == 0:
                        labels[cr, cc - 1] = gid
                        stack[top] = cr * n + (cc - 1)
                        top += 1
                # 右
                if cc < n - 1:
                    nb = board[cr, cc + 1]
                    if nb == 0:
                        pos = cr * n + (cc + 1)
                        if lib_seen[pos] != gid:
                            lib_seen[pos] = gid
                            lib[gid] += 1
                    elif nb == color and labels[cr, cc + 1] == 0:
                        labels[cr, cc + 1] = gid
                        stack[top] = cr * n + (cc + 1)
                        top += 1
    for r in range(n):
        for c in range(n):
            b = board[r, c]
            if b == 0:
                continue
            v = lib[labels[r, c]] * 0.25
            if v > 1.0:
                v = 1.0
            if b == player:
                own_out[r, c] = v
            else:
                opp_out[r, c] = v
def upgrade_state_channels(state, player=None):
    """旧版4通道状态 → 6通道：根据 ch0/ch1 重建棋面并计算气数通道。

    player: 该局面的当前玩家（±1）；缺省时由贴目通道符号推断。
    """
    n = state.shape[1]
    if player is None:
        player = 1 if state[3, 0, 0] > 0 else -1
    board = np.where(state[0] > 0.5, 1, np.where(state[1] > 0.5, -1, 0)).astype(np.int8)
    out = np.zeros((6, n, n), dtype=np.float32)
    out[:4] = state
    _liberty_channels(board, n, int(player), out[4], out[5])
    return out

@njit(cache=True)
def _canonical_state(board, st, n, komi, player, out):
    kch = np.float32(st[0] * komi / 14.0)
    last_valid = st[1] >= 0
    for r in range(n):
        for c in range(n):
            out[0, r, c] = 1.0 if board[r, c] == player else 0.0
            out[1, r, c] = 1.0 if board[r, c] == -player else 0.0
            out[2, r, c] = 1.0 if (last_valid and st[1] == r and st[2] == c) else 0.0
            out[3, r, c] = kch
    # 通道4=己方棋块气数/4（上限1），通道5=对方棋块气数/4（上限1）
    _liberty_channels(board, n, player, out[4], out[5])


class GoGame:
    """numba 加速版围棋（API 与 Python 版完全一致）"""

    def __init__(self, board_size=BOARD_SIZE, komi=KOMI):
        self.board_size = board_size
        self.komi = komi
        self._simulate_mode = False
        self.reset()

    def reset(self):
        n = self.board_size
        self.board = np.zeros((n, n), dtype=np.int8)
        self._st = np.zeros(12, dtype=np.int32)
        self._st[0] = 1                    # 当前玩家
        self._st[1] = -2                   # 上一步 r（-2=None）
        self._st[2] = -2
        self._st[3] = -2                   # 劫点 r
        self._st[4] = -2
        self._st[9] = -2                   # winner（-2=None）
        self._st[10] = MIN_MOVES_BEFORE_PASS
        self._go = np.zeros(1, dtype=np.bool_)
        self._fp = np.zeros(1, dtype=np.float64)
        self._cap = np.zeros(self.board_size * self.board_size, dtype=np.int32)
        self._terminal_cache = {}

    # ---- 属性（与 Python 版同名同语义）----
    @property
    def current_player(self):
        return int(self._st[0])

    @current_player.setter
    def current_player(self, v):
        self._st[0] = int(v)

    @property
    def last_move(self):
        if self._st[1] == -2:
            return None
        return (int(self._st[1]), int(self._st[2]))

    @last_move.setter
    def last_move(self, v):
        if v is None:
            self._st[1] = self._st[2] = -2
        else:
            self._st[1], self._st[2] = int(v[0]), int(v[1])

    @property
    def ko_point(self):
        if self._st[3] == -2:
            return None
        return (int(self._st[3]), int(self._st[4]))

    @ko_point.setter
    def ko_point(self, v):
        if v is None:
            self._st[3] = self._st[4] = -2
        else:
            self._st[3], self._st[4] = int(v[0]), int(v[1])

    @property
    def move_count(self):
        return int(self._st[5])

    @move_count.setter
    def move_count(self, v):
        self._st[5] = int(v)

    @property
    def pass_count(self):
        return int(self._st[6])

    @pass_count.setter
    def pass_count(self, v):
        self._st[6] = int(v)

    @property
    def total_pass_count(self):
        return {1: int(self._st[7]), -1: int(self._st[8])}

    @property
    def game_over(self):
        return bool(self._go[0])

    @game_over.setter
    def game_over(self, v):
        self._go[0] = bool(v)

    @property
    def winner(self):
        return None if self._st[9] == -2 else int(self._st[9])

    @winner.setter
    def winner(self, v):
        self._st[9] = -2 if v is None else int(v)

    @property
    def final_points(self):
        return float(self._fp[0])

    @final_points.setter
    def final_points(self, v):
        self._fp[0] = float(v)

    @property
    def empty_points(self):
        return {(r, c) for r in range(self.board_size)
                for c in range(self.board_size)
                if self.board[r, c] == 0}

    # ---- 热路径（numba）----
    def forced_pass(self):
        if self._st[0] == 1:
            return self._st[8] >= PASS_LIMIT
        return self._st[7] >= PASS_LIMIT

    def is_legal_move(self, row, col):
        if self._go[0]:
            return False
        if (row, col) == PASS_MOVE:
            return self._st[5] >= self._st[10]
        if self.forced_pass():
            return False
        return _is_legal_impl(self.board, row, col, self._st[0],
                                self._st[3], self._st[4], self.board_size)

    def make_move(self, row, col, check_legal=False):
        if check_legal and not self.is_legal_move(row, col):
            return False
        _make_move(self.board, self._st, self._go, self._fp,
                    row, col, self._cap)
        if self._go[0] and not self._simulate_mode and self.winner is None:
            self._compute_winner()
        return True

    def get_legal_moves_and_mask(self):
        n = self.board_size
        total = n * n
        moves = np.zeros(total + 1, dtype=np.int32)
        mask = np.zeros(total + 1, dtype=np.float32)
        cnt = _legal_moves_and_mask(self.board, self._st, n, moves, mask)
        out = []
        for j in range(cnt):
            i = moves[j]
            if i == total:
                out.append(PASS_MOVE)
            else:
                out.append((i // n, i % n))
        return out, mask

    def get_canonical_state(self, player=None):
        if player is None:
            player = self._st[0]
        out = np.zeros((6, self.board_size, self.board_size), dtype=np.float32)
        _canonical_state(self.board, self._st, self.board_size,
                            self.komi, int(player), out)
        return out


    def copy(self):
        g = GoGame.__new__(GoGame)
        g.board_size = self.board_size
        g.komi = self.komi
        g.board = self.board.copy()
        g._st = self._st.copy()
        g._go = self._go.copy()
        g._fp = self._fp.copy()
        g._cap = np.zeros(self.board_size * self.board_size, dtype=np.int32)
        g._terminal_cache = {}
        g._simulate_mode = False
        return g

    # ---- 以下冷路径沿用纯 Python（只在终局/复盘时调用，不影响速度）----
    def compute_score(self):
        board = self.board
        alive = self._compute_alive_mask(board)
        clean = np.where(alive, board, 0).astype(np.int8)
        black_stones = int(np.sum(clean == 1))
        white_stones = int(np.sum(clean == -1))
        territory_black, territory_white = self._territory(clean)
        black_score = black_stones + territory_black
        white_score = white_stones + territory_white + self.komi
        return black_score, white_score, black_score - white_score

    def compute_ownership_absolute(self):
        alive = self._compute_alive_mask(self.board)
        clean = np.where(alive, self.board, 0).astype(np.int8)
        own = clean.copy()
        n = self.board_size
        visited = np.zeros_like(clean, dtype=bool)
        for r in range(n):
            for c in range(n):
                if clean[r, c] != 0 or visited[r, c]:
                    continue
                stack, pts = [(r, c)], []
                visited[r, c] = True
                black_adj = white_adj = 0
                while stack:
                    cr, cc = stack.pop()
                    pts.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            if clean[nr, nc] == 0 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                            elif clean[nr, nc] == 1:
                                black_adj += 1
                            elif clean[nr, nc] == -1:
                                white_adj += 1
                if black_adj > 0 and white_adj == 0:
                    for p in pts:
                        own[p] = 1
                elif white_adj > 0 and black_adj == 0:
                    for p in pts:
                        own[p] = -1
        return own

    def compute_ownership_map(self, player=None):
        own_abs = self.compute_ownership_absolute()
        if player is None:
            return own_abs
        own_view = np.zeros_like(own_abs)
        own_view[own_abs == player] = 1
        own_view[own_abs == -player] = -1
        return own_view

    def _compute_winner(self):
        self.final_points = self.compute_score()[2]
        if self.final_points > 0:
            self.winner = 1
        elif self.final_points < 0:
            self.winner = -1
        else:
            self.winner = 0

    def _territory(self, board):
        n = self.board_size
        visited = np.zeros_like(board, dtype=bool)
        territory_black = territory_white = 0
        for r in range(n):
            for c in range(n):
                if board[r, c] != 0 or visited[r, c]:
                    continue
                stack, pts = [(r, c)], []
                visited[r, c] = True
                black_adj = white_adj = 0
                while stack:
                    cr, cc = stack.pop()
                    pts.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            if board[nr, nc] == 0 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                            elif board[nr, nc] == 1:
                                black_adj += 1
                            elif board[nr, nc] == -1:
                                white_adj += 1
                if black_adj > 0 and white_adj == 0:
                    territory_black += len(pts)
                elif white_adj > 0 and black_adj == 0:
                    territory_white += len(pts)
        return territory_black, territory_white

    def _analyze(self, board):
        n = self.board_size
        groups = []
        point_to_group = {}
        visited = np.zeros_like(board, dtype=bool)
        for r in range(n):
            for c in range(n):
                if board[r, c] == 0 or visited[r, c]:
                    continue
                color = int(board[r, c])
                stack, pts = [(r, c)], set()
                visited[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    pts.add((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < n and 0 <= nc < n and not visited[nr, nc]
                                and board[nr, nc] == color):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                gid = len(groups)
                groups.append((color, frozenset(pts)))
                for p in pts:
                    point_to_group[p] = gid
        regions = []
        region_groups = []
        group_regions = [[] for _ in groups]
        point_to_region = {}
        visited = np.zeros_like(board, dtype=bool)
        for r in range(n):
            for c in range(n):
                if board[r, c] != 0 or visited[r, c]:
                    continue
                stack, pts, adj = [(r, c)], set(), set()
                visited[r, c] = True
                while stack:
                    cr, cc = stack.pop()
                    pts.add((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            if board[nr, nc] == 0 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                            elif board[nr, nc] != 0:
                                adj.add(point_to_group[(nr, nc)])
                rid = len(regions)
                regions.append(frozenset(pts))
                region_groups.append(frozenset(adj))
                for g in adj:
                    group_regions[g].append(rid)
                for p in pts:
                    point_to_region[p] = rid
        return groups, regions, region_groups, group_regions, point_to_region

    def _compute_alive_mask(self, board):
        THRESH = 3.0
        STRENGTH = 10.0
        n = self.board_size
        groups, regions, region_groups, group_regions, point_to_region = self._analyze(board)
        if not groups:
            return np.zeros_like(board, dtype=bool)
        contact = [{} for _ in groups]
        for gid, (_, pts) in enumerate(groups):
            for r, c in pts:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and board[nr, nc] == 0:
                        rid = point_to_region[(nr, nc)]
                        contact[gid][rid] = contact[gid].get(rid, 0) + 1

        def potentials(strength):
            bp = np.zeros(len(regions))
            wp = np.zeros(len(regions))
            for rid, gids in enumerate(region_groups):
                for gid in gids:
                    pot = strength[gid] * contact[gid].get(rid, 0)
                    if groups[gid][0] == 1:
                        bp[rid] += pot
                    else:
                        wp[rid] += pot
            return bp, wp

        def has_eye(gid, bp, wp):
            color = groups[gid][0]
            for rid in group_regions[gid]:
                my = bp[rid] if color == 1 else wp[rid]
                op = wp[rid] if color == 1 else bp[rid]
                if my - op >= THRESH or (op == 0 and my > 0):
                    return True
            return False

        bp, wp = potentials(np.ones(len(groups)))
        strength = np.array([STRENGTH if has_eye(g, bp, wp) else 1.0
                                for g in range(len(groups))])
        bp, wp = potentials(strength)
        dead = set()
        for gid in range(len(groups)):
            color = groups[gid][0]
            if has_eye(gid, bp, wp):
                continue
            if any((wp[rid] if color == 1 else bp[rid]) -
                    (bp[rid] if color == 1 else wp[rid]) >= THRESH
                    for rid in group_regions[gid]):
                dead.add(gid)
        mask = np.zeros_like(board, dtype=bool)
        for gid, (_, pts) in enumerate(groups):
            if gid not in dead:
                for r, c in pts:
                    mask[r, c] = True
        return mask

    def get_terminal_value(self, player):
        if not self.game_over:
            return None
        board_hash = hash(self.board.tobytes())
        cache_key = (board_hash, player)
        if cache_key in self._terminal_cache:
            return self._terminal_cache[cache_key]
        scale = SCALE
        value_base = math.fabs(np.tanh(self.final_points / scale))
        if self.winner == 0:
            value = 0.0
        else:
            value = value_base if self.winner == player else -value_base
        self._terminal_cache[cache_key] = value
        return value

    def __str__(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        s = "   " + " ".join(f"{i:2d}" for i in range(self.board_size)) + "\n"
        for i in range(self.board_size):
            s += f"{i:2d} " + "  ".join(symbols[self.board[i, j]] for j in range(self.board_size)) + "\n"
        return s
