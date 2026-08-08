"""
围棋游戏逻辑
"""
import numpy as np
from copy import deepcopy
import math
from numba import njit
from hyperparams import (BOARD_SIZE, KOMI, MAX_MOVES, PASS_MOVE,
                         MIN_MOVES_BEFORE_PASS, SCALE, PASS_LIMIT,
                         SAFE_CAPTURE_PASSES)

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
    """己方/对方棋块危急度通道：1气=1.0（最危急），>=4气=0（安全）；空点或非本颜色为0。"""
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
            v = (4 - min(lib[labels[r, c]], 4)) / 3.0   # 危急度：1气=1.0最危急，>=4气=0安全
            if b == player:
                own_out[r, c] = v
            else:
                opp_out[r, c] = v

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
    # 通道4=己方棋块危急度（1气=1.0最危急），通道5=对方棋块危急度
    _liberty_channels(board, n, player, out[4], out[5])


@njit(cache=True)
def _life_death(board, g_label, n, g_color, g_adj_regions, g_adj_contact, g_adj_cnt,
                ng, thresh, strength_eye):
    """死活启发式：pass1潜力→眼判定→强度；pass2强度潜力→死棋判定；
    最后加“气数硬判定”：气≤1的棋块必死（对手提子必合法：提后己方有气）。
    返回 (alive int8[ng+1], g_eyes int32[ng+1] 每组眼区域数)。"""
    max_r = g_adj_regions.shape[1]
    bp = np.zeros(max_r, dtype=np.float64)
    wp = np.zeros(max_r, dtype=np.float64)
    strength = np.ones(ng + 1, dtype=np.float64)
    for g in range(1, ng + 1):
        for j in range(g_adj_cnt[g]):
            rid = g_adj_regions[g, j]
            pot = float(g_adj_contact[g, j])
            if g_color[g] == 1:
                bp[rid] += pot
            else:
                wp[rid] += pot
    for g in range(1, ng + 1):
        for j in range(g_adj_cnt[g]):
            rid = g_adj_regions[g, j]
            if g_color[g] == 1:
                my = bp[rid]
                op = wp[rid]
            else:
                my = wp[rid]
                op = bp[rid]
            if my - op >= thresh or (op == 0.0 and my > 0.0):
                strength[g] = strength_eye
                break
    bp[:] = 0.0
    wp[:] = 0.0
    for g in range(1, ng + 1):
        for j in range(g_adj_cnt[g]):
            rid = g_adj_regions[g, j]
            pot = strength[g] * float(g_adj_contact[g, j])
            if g_color[g] == 1:
                bp[rid] += pot
            else:
                wp[rid] += pot
    alive = np.ones(ng + 1, dtype=np.int8)
    g_eyes = np.zeros(ng + 1, dtype=np.int32)
    for g in range(1, ng + 1):
        eye_cnt = 0
        for j in range(g_adj_cnt[g]):
            rid = g_adj_regions[g, j]
            if g_color[g] == 1:
                my = bp[rid]
                op = wp[rid]
            else:
                my = wp[rid]
                op = bp[rid]
            if my - op >= thresh or (op == 0.0 and my > 0.0):
                eye_cnt += 1
        g_eyes[g] = eye_cnt
        if eye_cnt > 0:
            continue
        for j in range(g_adj_cnt[g]):
            rid = g_adj_regions[g, j]
            if g_color[g] == 1:
                my = bp[rid]
                op = wp[rid]
            else:
                my = wp[rid]
                op = bp[rid]
            if op - my >= thresh:
                alive[g] = 0
                break
    # 气数硬判定：气≤1的棋块必死（1气可被直接提掉；0气不可能存在于合法盘面）。
    # 修复“只剩一口气仍判活”——接触潜力模型无法表达“唯一气点=立即被提”。
    g_libs = np.zeros(ng + 1, dtype=np.int32)
    for r in range(n):
        for c in range(n):
            g = g_label[r, c]
            if g > 0:
                if r > 0 and board[r - 1, c] == 0:
                    g_libs[g] += 1
                if r < n - 1 and board[r + 1, c] == 0:
                    g_libs[g] += 1
                if c > 0 and board[r, c - 1] == 0:
                    g_libs[g] += 1
                if c < n - 1 and board[r, c + 1] == 0:
                    g_libs[g] += 1
    for g in range(1, ng + 1):
        if g_libs[g] <= 1:
            alive[g] = 0
    return alive, g_eyes


@njit(cache=True)
def _group_size_liberties(board, n, r, c):
    """BFS棋块(r,c)：返回(大小, 独立气数)。气用seen印记去重（同一空点只计一次）。"""
    color = board[r, c]
    seen = np.zeros(n * n, dtype=np.int8)   # 0=未访问 1=棋块点 2=空点已计气
    stack = np.zeros(n * n, dtype=np.int32)
    top = 0
    stack[top] = r * n + c
    top += 1
    seen[r * n + c] = 1
    size = 0
    libs = 0
    while top > 0:
        top -= 1
        cur = stack[top]
        cr = cur // n
        cc = cur % n
        size += 1
        # 上
        if cr > 0:
            nb = cur - n
            v = board[cr - 1, cc]
            if v == color:
                if seen[nb] == 0:
                    seen[nb] = 1
                    stack[top] = nb
                    top += 1
            elif v == 0:
                if seen[nb] == 0:
                    seen[nb] = 2
                    libs += 1
        # 下
        if cr < n - 1:
            nb = cur + n
            v = board[cr + 1, cc]
            if v == color:
                if seen[nb] == 0:
                    seen[nb] = 1
                    stack[top] = nb
                    top += 1
            elif v == 0:
                if seen[nb] == 0:
                    seen[nb] = 2
                    libs += 1
        # 左
        if cc > 0:
            nb = cur - 1
            v = board[cr, cc - 1]
            if v == color:
                if seen[nb] == 0:
                    seen[nb] = 1
                    stack[top] = nb
                    top += 1
            elif v == 0:
                if seen[nb] == 0:
                    seen[nb] = 2
                    libs += 1
        # 右
        if cc < n - 1:
            nb = cur + 1
            v = board[cr, cc + 1]
            if v == color:
                if seen[nb] == 0:
                    seen[nb] = 1
                    stack[top] = nb
                    top += 1
            elif v == 0:
                if seen[nb] == 0:
                    seen[nb] = 2
                    libs += 1
    return size, libs


@njit(cache=True)
def _flood_clear(board, n, sr, sc, color, dead):
    """整块移除（提子），被移除的点写入dead（n,n布尔）。"""
    stack = np.zeros(n * n, dtype=np.int32)
    top = 0
    stack[top] = sr * n + sc
    top += 1
    board[sr, sc] = 0
    while top > 0:
        top -= 1
        cur = stack[top]
        cr = cur // n
        cc = cur % n
        dead[cr, cc] = True
        if cr > 0 and board[cr - 1, cc] == color:
            board[cr - 1, cc] = 0
            stack[top] = cur - n
            top += 1
        if cr < n - 1 and board[cr + 1, cc] == color:
            board[cr + 1, cc] = 0
            stack[top] = cur + n
            top += 1
        if cc > 0 and board[cr, cc - 1] == color:
            board[cr, cc - 1] = 0
            stack[top] = cur - 1
            top += 1
        if cc < n - 1 and board[cr, cc + 1] == color:
            board[cr, cc + 1] = 0
            stack[top] = cur + 1
            top += 1


@njit(cache=True)
def _remove_adjacent_captured(board, n, r, c, color, dead):
    """落子(r,c)后，提掉相邻的0气color棋块（提子只可能发生在4邻域）。"""
    if r > 0 and board[r - 1, c] == color:
        sz, libs = _group_size_liberties(board, n, r - 1, c)
        if libs == 0:
            _flood_clear(board, n, r - 1, c, color, dead)
    if r < n - 1 and board[r + 1, c] == color:
        sz, libs = _group_size_liberties(board, n, r + 1, c)
        if libs == 0:
            _flood_clear(board, n, r + 1, c, color, dead)
    if c > 0 and board[r, c - 1] == color:
        sz, libs = _group_size_liberties(board, n, r, c - 1)
        if libs == 0:
            _flood_clear(board, n, r, c - 1, color, dead)
    if c < n - 1 and board[r, c + 1] == color:
        sz, libs = _group_size_liberties(board, n, r, c + 1)
        if libs == 0:
            _flood_clear(board, n, r, c + 1, color, dead)


@njit(cache=True)
def _add_connection_points(board, n):
    """连接点辅助局面：一遍扫描（就地修改board），在连接点放置对应方棋子。
    连接点定义：某空点四邻中恰好 2个同方子+2个空位（四邻）；边点（3邻）为 2同方+1空；角点（2邻）永不是。
    边扫描边落子：先落子的连接子会改变后续点的邻位构成，使其不再是连接点（符合用户定义）。
    放置后所在棋块至少有2口气（定义保证），不会自杀。"""
    for r in range(n):
        for c in range(n):
            if board[r, c] != 0:
                continue
            same_b = same_w = empty = total = 0
            # 上
            if r > 0:
                total += 1
                v = board[r - 1, c]
                if v == 1:
                    same_b += 1
                elif v == -1:
                    same_w += 1
                else:
                    empty += 1
            # 下
            if r < n - 1:
                total += 1
                v = board[r + 1, c]
                if v == 1:
                    same_b += 1
                elif v == -1:
                    same_w += 1
                else:
                    empty += 1
            # 左
            if c > 0:
                total += 1
                v = board[r, c - 1]
                if v == 1:
                    same_b += 1
                elif v == -1:
                    same_w += 1
                else:
                    empty += 1
            # 右
            if c < n - 1:
                total += 1
                v = board[r, c + 1]
                if v == 1:
                    same_b += 1
                elif v == -1:
                    same_w += 1
                else:
                    empty += 1
            if total <= 2:
                continue   # 角点（2邻）永不是连接点
            if same_b == 2 and empty == total - 2:
                board[r, c] = 1
            elif same_w == 2 and empty == total - 2:
                board[r, c] = -1


@njit(cache=True)
def _safe_capture_dead(board, n, attacker, max_passes=SAFE_CAPTURE_PASSES):
    """安全点捕获法（用户方法第一步/第二步共用）：
    攻击方只走“安全点”、迭代落子（防守方全程不落子），吃掉防守方死子。
    安全点：落子后己方棋块<4子→恒安全（小块点眼杀棋）；棋块>=4子→气须>=2（防双活自陷）。
    己方单眼不填：四邻全为己方棋/盘外的点跳过（填了自紧气，削弱后续攻击）。
    只迭代 max_passes 轮（默认3）：1-2轮内吃掉的判死；多轮才吃掉≈已活（防守方不抵抗还拖多轮）→判活。
    返回防守方被吃掉的死子掩码 (n,n) bool。"""
    b = board.copy()
    dead = np.zeros((n, n), dtype=np.bool_)
    defender = -attacker
    scratch = np.zeros((n, n), dtype=np.bool_)
    # 无防守方棋子 → 无死子
    has_def = False
    for r in range(n):
        for c in range(n):
            if board[r, c] == defender:
                has_def = True
                break
        if has_def:
            break
    if not has_def:
        return dead
    for _pass in range(max_passes):
        placed = False
        for r in range(n):
            for c in range(n):
                if b[r, c] != 0:
                    continue
                # 己方单眼不填：四邻全为己方棋/盘外 → 该点就是己方单眼，填了自紧气削弱后续攻击
                if ((r == 0 or b[r - 1, c] == attacker) and
                        (r == n - 1 or b[r + 1, c] == attacker) and
                        (c == 0 or b[r, c - 1] == attacker) and
                        (c == n - 1 or b[r, c + 1] == attacker)):
                    continue
                # 候选模拟：落子 → 提相邻防守子 → 算棋块大小/气
                tmp = b.copy()
                tmp[r, c] = attacker
                scratch[:] = False
                _remove_adjacent_captured(tmp, n, r, c, defender, scratch)
                size, libs = _group_size_liberties(tmp, n, r, c)
                if libs == 0:
                    continue   # 自杀，非法点
                if size >= 4 and libs < 2:
                    continue   # 大块只剩1气，不安全（双活陷阱：谁填谁完蛋）
                # 安全点 → 正式落子并吃子
                b[r, c] = attacker
                _remove_adjacent_captured(b, n, r, c, defender, dead)
                placed = True
        if not placed:
            break
    return dead


@njit(cache=True)
def _regions_and_colors(board, n, max_c, r_label, r_size, r_has_b, r_has_w):
    """标记空区域并统计各区域邻接的黑/白（供领地/中立判定）。返回区域数nr（1-based）。"""
    stack = np.zeros(n * n, dtype=np.int32)
    nr = 0
    for r in range(n):
        for c in range(n):
            if board[r, c] != 0 or r_label[r, c] != 0:
                continue
            nr += 1
            r_size[nr] = 0
            top = 0
            stack[top] = r * n + c
            top += 1
            r_label[r, c] = nr
            while top > 0:
                top -= 1
                cur = stack[top]
                cr = cur // n
                cc = cur % n
                r_size[nr] += 1
                # 上
                if cr > 0:
                    v = board[cr - 1, cc]
                    if v == 0:
                        if r_label[cr - 1, cc] == 0:
                            r_label[cr - 1, cc] = nr
                            stack[top] = cur - n
                            top += 1
                    elif v == 1:
                        r_has_b[nr] = True
                    else:
                        r_has_w[nr] = True
                # 下
                if cr < n - 1:
                    v = board[cr + 1, cc]
                    if v == 0:
                        if r_label[cr + 1, cc] == 0:
                            r_label[cr + 1, cc] = nr
                            stack[top] = cur + n
                            top += 1
                    elif v == 1:
                        r_has_b[nr] = True
                    else:
                        r_has_w[nr] = True
                # 左
                if cc > 0:
                    v = board[cr, cc - 1]
                    if v == 0:
                        if r_label[cr, cc - 1] == 0:
                            r_label[cr, cc - 1] = nr
                            stack[top] = cur - 1
                            top += 1
                    elif v == 1:
                        r_has_b[nr] = True
                    else:
                        r_has_w[nr] = True
                # 右
                if cc < n - 1:
                    v = board[cr, cc + 1]
                    if v == 0:
                        if r_label[cr, cc + 1] == 0:
                            r_label[cr, cc + 1] = nr
                            stack[top] = cur + 1
                            top += 1
                    elif v == 1:
                        r_has_b[nr] = True
                    else:
                        r_has_w[nr] = True
    return nr


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
        data = self._merged_analysis()
        # 中立区域（黑白边界）：归属0，目数黑白各0.5（目差中相互抵消）
        black_score = data['black_stones'] + data['tb'] + 0.5 * data['neutral']
        white_score = data['white_stones'] + data['tw'] + 0.5 * data['neutral'] + self.komi
        return black_score, white_score, data['final_diff']

    def compute_ownership_absolute(self):
        return self._merged_analysis()['own']

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

    def _territory(self, board=None):
        data = self._merged_analysis()
        return data['tb'], data['tw']

    # ---- 终局判定（安全点捕获法，取代接触潜力启发式/MCTS验证）----
    def terminal_analysis(self):
        """终局完整分析（训练标签用）：
        安全点捕获法判定黑白死子 → 干净棋盘领地/归属。
        更新 self.final_points / self.winner，返回绝对归属图 own(H,W int8)。"""
        data = self._merged_analysis()
        self._fp[0] = float(data['final_diff'])
        self._st[9] = 1 if data['final_diff'] > 0 else (-1 if data['final_diff'] < 0 else 0)
        return data['own']

    def _compute_alive_mask(self, board=None):
        return self._merged_analysis()['alive_mask']

    def _merged_analysis(self):
        """安全点捕获法（用户方法）：
        0. 连接点辅助局面：原盘复制后一遍扫描放置连接子（防“本可随时连接却被分开吃掉”的活棋误判死）；
        第一步 黑攻：在辅助局面上只走“安全点”迭代落子（白全程不落子），吃掉的白子=DeadWhiteMask；
        第二步 白攻：白棋只走“安全点”迭代落子，吃掉的黑子=DeadBlackMask；
        第三步：数目用原始棋局（连接子不在原盘上，MASK覆盖到它们无影响）：黑独占区域=黑地、白独占=白地、
                黑白边界区域=中立（归属0，目数黑白各0.5，目差中相互抵消）。
        安全点：落子后己方棋块<3子→恒安全（小块点眼杀棋）；棋块>=3子→须气>=2（防双活自陷）。"""
        board = self.board
        n = self.board_size
        # 连接点辅助局面：只用于算两个MASK，数目仍用原盘
        aux = board.copy()
        _add_connection_points(aux, n)
        dead_white = _safe_capture_dead(aux, n, 1, SAFE_CAPTURE_PASSES)    # 黑攻 → 白死子
        dead_black = _safe_capture_dead(aux, n, -1, SAFE_CAPTURE_PASSES)   # 白攻 → 黑死子
        clean = board.copy()
        clean[dead_white] = 0
        clean[dead_black] = 0
        black_stones = int(np.sum(clean == 1))
        white_stones = int(np.sum(clean == -1))
        # 干净棋盘区域分析：黑独占/白独占/中立
        max_c = n * n
        r_label = np.zeros((n, n), dtype=np.int32)
        r_size = np.zeros(max_c + 1, dtype=np.int32)
        r_has_b = np.zeros(max_c + 1, dtype=np.bool_)
        r_has_w = np.zeros(max_c + 1, dtype=np.bool_)
        nr = _regions_and_colors(clean, n, max_c, r_label, r_size, r_has_b, r_has_w)
        tb = tw = neutral = 0
        for rid in range(1, nr + 1):
            if r_has_b[rid] and not r_has_w[rid]:
                tb += int(r_size[rid])
            elif r_has_w[rid] and not r_has_b[rid]:
                tw += int(r_size[rid])
            else:
                neutral += int(r_size[rid])
        # 归属图：活子±1；黑/白独占区域±1；中立=0
        own = clean.copy().astype(np.int8)
        for r in range(n):
            for c in range(n):
                rid = r_label[r, c]
                if rid > 0:
                    if r_has_b[rid] and not r_has_w[rid]:
                        own[r, c] = 1
                    elif r_has_w[rid] and not r_has_b[rid]:
                        own[r, c] = -1
        return {
            'alive_mask': (clean != 0),
            'own': own,
            'tb': tb, 'tw': tw, 'neutral': neutral,
            'black_stones': black_stones, 'white_stones': white_stones,
            'final_diff': (black_stones + tb) - (white_stones + tw + self.komi),
        }

    def get_terminal_value(self, player):
        if not self.game_over:
            return None
        board_hash = hash(self.board.tobytes())
        cache_key = (board_hash, player)
        if cache_key in self._terminal_cache:
            return self._terminal_cache[cache_key]
        # P0修复：终局回传原始目差（无界），与网络 value（ownership.sum()）同尺度。
        # 旧版 /SCALE=10 会令搜索中终局分支的 Q 系统性偏小（官子不敏感、不主动终局）。
        value_base = math.fabs(self.final_points)
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
