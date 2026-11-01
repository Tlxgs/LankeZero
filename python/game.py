"""
围棋游戏逻辑
"""
import numpy as np
from numba import njit
from hyperparams import (BOARD_SIZE, KOMI, PASS_MOVE, HOLE, MAX_BOARD_CELLS,
                         board_rules, validate_hollow, PRUNE_DEAD_LUMPS,
                         INPUT_CHANNELS,
                         CH_OWN_STONES, CH_OPP_STONES, CH_LAST_MOVE, CH_KOMI,
                         CH_OWN_URGENCY, CH_OPP_URGENCY, CH_LEGAL_MASK,
                         CH_OWN_EYES, CH_OPP_EYES, CH_BOUNDARY)


def boundary_plane(board_size, hollow=0):
    """生成边界输入通道：洞内 -1、内外边界 +1、其余可落子点 0。

    13 路镂空 7×7 时：中心 7×7=-1，9×9 最外圈=+1，13×13 最外圈=+1，
    其余可落子点=0。hollow=0 时退化为「最外圈 1、内部 0」。
    """
    n = int(board_size)
    h = validate_hollow(n, hollow)
    out = np.zeros((n, n), dtype=np.float32)
    out[0, :] = 1.0
    out[-1, :] = 1.0
    out[:, 0] = 1.0
    out[:, -1] = 1.0
    if h:
        lo = (n - h) // 2
        hi = lo + h
        out[lo:hi, lo:hi] = -1.0
        # 洞外一圈：hollow+2 尺寸的空心矩形的最外圈。
        out[lo - 1, lo - 1:hi + 1] = 1.0
        out[hi, lo - 1:hi + 1] = 1.0
        out[lo - 1:hi + 1, lo - 1] = 1.0
        out[lo - 1:hi + 1, hi] = 1.0
    return out

# Zobrist 表：用于「全历史局面」精确比对（位置超级劫），
# 同时保留上一局面哈希以兼容旧接口。
# 颜色下标 0=黑、1=白；扁平下标 r*n+c，覆盖当前支持的最大棋盘。
_ZOBRIST = np.random.RandomState(0x5EED).randint(
    1, 2**63 - 1, size=(2, MAX_BOARD_CELLS), dtype=np.uint64)
_MASK32 = np.uint64(0xFFFFFFFF)


@njit(cache=True)
def _hash_from_slots(st):
    """从 _st[13:15] 取当前局面哈希，_st[15:17] 取上一局面哈希。"""
    cur = ((np.uint64(st[13]) & _MASK32)
           | ((np.uint64(st[14]) & _MASK32) << np.uint64(32)))
    prev = ((np.uint64(st[15]) & _MASK32)
            | ((np.uint64(st[16]) & _MASK32) << np.uint64(32)))
    return cur, prev


@njit(cache=True)
def _set_hash_slots(st, cur, prev):
    st[13] = np.int32(cur & _MASK32)
    st[14] = np.int32((cur >> np.uint64(32)) & _MASK32)
    st[15] = np.int32(prev & _MASK32)
    st[16] = np.int32((prev >> np.uint64(32)) & _MASK32)


@njit(cache=True)
def _board_hash(board, n):
    h = np.uint64(0)
    for r in range(n):
        for c in range(n):
            v = board[r, c]
            if v != 0 and v != HOLE:
                h ^= _ZOBRIST[0 if v == 1 else 1, r * n + c]
    return h


@njit(cache=True)
def _sync_history(st, hist, cur_h):
    """保证「当前局面」已写入历史，返回历史条数。

    正常对局中当前局面就是历史最后一条，这里只做一次 O(1) 比对；
    手工改棋盘/直接写 _st 的调用方（测试、SGF 导入等）会被顺带补上，
    因此全历史判定对旧接口仍然成立。
    """
    hist_n = st[17]
    if hist_n <= 0 or hist[hist_n - 1] != cur_h:
        if hist_n < hist.shape[0]:
            hist[hist_n] = cur_h
            hist_n += 1
            st[17] = hist_n
    return hist_n


@njit(cache=True)
def _history_contains(hist, hist_n, h):
    for i in range(hist_n):
        if hist[i] == h:
            return True
    return False

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
def _is_legal_impl(board, row, col, player, ko_r, ko_c, cur_h, prev_h,
                   use_hash, n, hist, hist_n):
    """等价 Python _is_legal_move（临时落子后恢复，不污染棋盘）。

    use_hash=True 时，用「落子后的局面哈希」与全历史比对判位置超级劫；
    没有历史状态的调用方（_legal_mask_for 的 D4 增强）仍用旧 ko 点判据。
    非提子着法不可能重现旧局面，因此只在 captured>0 时查历史。
    """
    if board[row, col] != 0:
        return False
    board[row, col] = player
    cap = np.zeros(n * n, dtype=np.int32)
    captured = _capture_opponent(board, row, col, player, n, cap)
    board[row, col] = 0
    if captured > 0:
        if use_hash:
            ci = 0 if player == 1 else 1
            opp = 1 - ci
            h = cur_h ^ _ZOBRIST[ci, row * n + col]
            for j in range(captured):
                h ^= _ZOBRIST[opp, cap[j]]
            if hist_n > 0:
                if _history_contains(hist, hist_n, h):
                    return False                  # 重现历史局面 = 超级劫
            elif h == prev_h:
                return False
        elif ko_r != -2 and row == ko_r and col == ko_c:
            return False                          # 无历史状态时的旧兼容判据
    if captured == 0:
        board[row, col] = player
        ok = _group_has_liberty(board, row, col, n)
        board[row, col] = 0
        return ok                                 # 自杀判定
    return True

@njit(cache=True)
def _make_move(board, st, go, fp, row, col, cap, hist):
    """落子（含 Pass/提子/超级劫检测），非法超级劫着法会回滚并返回 False。
    st: [0]当前玩家 [1]上一步r [2]上一步c [3]劫r [4]劫c
        [5]步数 [6]连续pass [7]黑总pass [8]白总pass [9]winner(-2=None)
        [10]min_moves [11]max_moves [12]pass_limit
        [13:15]当前局面 Zobrist 哈希 [15:17]上一局面哈希
        [17]历史局面条数。
    """
    n = board.shape[0]
    cur_h = ((np.uint64(st[13]) & _MASK32)
             | ((np.uint64(st[14]) & _MASK32) << np.uint64(32)))
    hist_n = _sync_history(st, hist, cur_h)
    if row == -1 and col == -1:                   # Pass
        st[1] = -1
        st[2] = -1
        st[3] = -2                                # 停一手会解除劫禁
        st[4] = -2
        if st[0] == 1:
            st[7] += 1
        else:
            st[8] += 1
        st[6] += 1
        st[5] += 1
        st[0] = -st[0]
        _set_hash_slots(st, cur_h, cur_h)
        if st[6] >= 2 or st[5] >= st[11]:
            go[0] = True
        return True
    h = cur_h
    ci = 0 if st[0] == 1 else 1
    h ^= _ZOBRIST[ci, row * n + col]
    board[row, col] = st[0]
    captured = _capture_opponent(board, row, col, st[0], n, cap)
    if captured > 0:
        for j in range(captured):
            idx = cap[j]
            h ^= _ZOBRIST[1 - ci, idx]
            board[idx // n, idx % n] = 0
        # 全历史比对：非提子着法不可能重现旧局面，只有提子才需要查。
        if hist_n > 0 and _history_contains(hist, hist_n, h):
            for j in range(captured):              # 回滚：还原被提子与本手
                idx = cap[j]
                board[idx // n, idx % n] = -st[0]
            board[row, col] = 0
            return False
        if captured == 1:                         # 简单劫点标记（仅供旧接口/GUI 展示）
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
    _set_hash_slots(st, h, cur_h)
    if hist_n < hist.shape[0]:                     # 记录本手后的局面
        hist[hist_n] = h
        st[17] = hist_n + 1
    st[0] = -st[0]          # 终局后 current_player = 下一手方，终局价值口径与双 Pass 一致
    if st[5] >= st[11]:
        go[0] = True
    return True

@njit(cache=True)
def _move_captures(board, r, c, player, n, seen, stack, stamp):
    """快速判断在 (r,c) 落子是否会提掉相邻敌块。

    只对「有相邻空点」的候选点使用；有相邻空点仍可能提子
    （空点只给新子提供气，不一定给相邻敌块提供气），因此不能一概走快速路径。
    先看相邻敌子本地是否有空邻点；有则其整块必有气，可跳过 BFS。
    """
    opp = -player
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < n and 0 <= nc < n) or board[nr, nc] != opp:
            continue
        idx0 = nr * n + nc
        if seen[idx0] == stamp:
            continue
        # 本地快速排除：该敌子除 (r,c) 外还有空邻点，整块至少有这口气。
        local_air = False
        for dr2, dc2 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r2 = nr + dr2
            c2 = nc + dc2
            if not (0 <= r2 < n and 0 <= c2 < n):
                continue
            if r2 == r and c2 == c:
                continue
            if board[r2, c2] == 0:
                local_air = True
                break
        if local_air:
            continue
        top = 0
        stack[top] = idx0
        top += 1
        seen[idx0] = stamp
        has_air = False
        while top > 0:
            top -= 1
            cur = stack[top]
            cr = cur // n
            cc = cur % n
            for dr2, dc2 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                r2 = cr + dr2
                c2 = cc + dc2
                if not (0 <= r2 < n and 0 <= c2 < n):
                    continue
                v = board[r2, c2]
                if v == 0:
                    if not (r2 == r and c2 == c):
                        has_air = True
                elif v == opp:
                    idx2 = r2 * n + c2
                    if seen[idx2] != stamp:
                        seen[idx2] = stamp
                        stack[top] = idx2
                        top += 1
            if has_air:
                break
        if not has_air:
            return True
    return False


@njit(cache=True)
def _legal_moves_and_mask(board, st, n, moves, mask, hist):
    """合法着法；moves: int32缓冲(n*n+1)，mask: float32(n*n+1)"""
    total = n * n
    mask[:] = 0.0
    count = 0
    if st[0] == 1:
        forced = st[8] >= st[12]
    else:
        forced = st[7] >= st[12]
    if forced:                                    # 强制 pass
        moves[count] = total
        count += 1
        mask[total] = 1.0
        return count
    cur_h, prev_h = _hash_from_slots(st)
    hist_n = _sync_history(st, hist, cur_h)
    seen = np.zeros(total, dtype=np.int32)
    stack = np.zeros(total, dtype=np.int32)
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
            # 有相邻空点也可能提子（还取决于敌块是否另有气），
            # 提子就可能重现历史局面，必须走完整判定。
            is_ko = st[3] != -2 and r == st[3] and c == st[4]
            if has_air and not is_ko:
                opp = -st[0]
                adj_opp = False
                if r > 0 and board[r - 1, c] == opp:
                    adj_opp = True
                elif r < n - 1 and board[r + 1, c] == opp:
                    adj_opp = True
                elif c > 0 and board[r, c - 1] == opp:
                    adj_opp = True
                elif c < n - 1 and board[r, c + 1] == opp:
                    adj_opp = True
                if (not adj_opp
                        or not _move_captures(board, r, c, st[0], n,
                                              seen, stack, i + 1)):
                    moves[count] = i
                    count += 1
                    mask[i] = 1.0
                    continue
            if _is_legal_impl(board, r, c, st[0], st[3], st[4],
                              cur_h, prev_h, True, n, hist, hist_n):
                moves[count] = i
                count += 1
                mask[i] = 1.0
    if st[5] >= st[10]:
        moves[count] = total
        count += 1
        mask[total] = 1.0
    return count


@njit(cache=True)
def _legal_mask_for(board, player, ko_r, ko_c, n, mask):
    """只算棋盘合法点（不含 pass），用于黑白互换增强重算 legal 通道。

    board 用绝对/当前玩家相对符号均可，只要与 player 一致；ko_r/ko_c=-2 表示无劫。
    这里没有历史局面，超级劫退化为旧的 ko 点判据。
    """
    mask[:] = 0.0
    dummy_hist = np.zeros(1, dtype=np.uint64)
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
            is_ko = ko_r != -2 and r == ko_r and c == ko_c
            if (has_air and not is_ko) or _is_legal_impl(
                    board, r, c, player, ko_r, ko_c,
                    np.uint64(0), np.uint64(0), False, n, dummy_hist, 0):
                mask[i] = 1.0


@njit(cache=True)
def _eye_color(board, n, r, c):
    """点 (r,c) 是不是「单点真眼」；是则返回眼的归属色（+1/-1），不是返回 0。

    判据（贴边/角/镂空边时方向数自动变少，无需特判）：
      ① 上下左右（盘内且不是洞）必须全是同一种颜色的子；
      ② 对角：**中心**（4 个对角在盘内）至多 1 个敌方子；**边/角/洞边**
         （2/1 个有效对角）必须全为己方。
    洞与棋盘外同义：既不是气，也不是敌方子，不参与对角计数。
    """
    col = 0
    # ① 四个正邻：同色且非空；洞视作盘外，跳过。
    if r > 0:
        v = board[r - 1, c]
        if v != HOLE:
            if v == 0:
                return 0
            col = v
    if r < n - 1:
        v = board[r + 1, c]
        if v != HOLE:
            if v == 0 or (col != 0 and v != col):
                return 0
            if col == 0:
                col = v
    if c > 0:
        v = board[r, c - 1]
        if v != HOLE:
            if v == 0 or (col != 0 and v != col):
                return 0
            if col == 0:
                col = v
    if c < n - 1:
        v = board[r, c + 1]
        if v != HOLE:
            if v == 0 or (col != 0 and v != col):
                return 0
            if col == 0:
                col = v
    if col == 0:
        return 0
    # ② 对角：nd = 有效对角数（4=中心、2=边、1=角/洞边），enemy=敌方数，nonown=非己方数（含空）
    nd = 0
    enemy = 0
    nonown = 0
    if r > 0 and c > 0:
        v = board[r - 1, c - 1]
        if v != HOLE:
            nd += 1
            if v == -col:
                enemy += 1
                nonown += 1
            elif v == 0:
                nonown += 1
    if r > 0 and c < n - 1:
        v = board[r - 1, c + 1]
        if v != HOLE:
            nd += 1
            if v == -col:
                enemy += 1
                nonown += 1
            elif v == 0:
                nonown += 1
    if r < n - 1 and c > 0:
        v = board[r + 1, c - 1]
        if v != HOLE:
            nd += 1
            if v == -col:
                enemy += 1
                nonown += 1
            elif v == 0:
                nonown += 1
    if r < n - 1 and c < n - 1:
        v = board[r + 1, c + 1]
        if v != HOLE:
            nd += 1
            if v == -col:
                enemy += 1
                nonown += 1
            elif v == 0:
                nonown += 1
    if nd == 4:
        return col if enemy <= 1 else 0
    return col if nonown == 0 else 0


@njit(cache=True)
def _group_channels(board, n, player, own_urg, opp_urg, own_eye, opp_eye):
    """按棋块（同色连通）算三类特征，写进四个平面（下标见 hyperparams 的 CH_*）：

    危急度（own_urg/opp_urg）：1 气 = 1.0 最危急，>=4 气 = 0 安全；空点与非本颜色为 0。
    眼（own_eye/opp_eye）：0 眼 = 0、1 眼 = 0.5、>=2 眼 = 1，**整个广义块共享同一个值**。

    眼不再数「单点真眼」，而是折算**眼区**（Benson 的 region）：一个极大的 4-邻接空点连通块，
    其全部盘内 4-邻点都是同一种颜色的棋子（可来自多块棋；对角不参与连通、也不参与包围判定）。
    折算规则：
      面积 1    → 仍要求过 `_eye_color`（含对角条件）才算 1 只 —— 它是「现成的眼」，必须是真眼；
      面积 2~5  → 1 只；面积 >=6 → 2 只。这一档是「眼形空间」（能做出眼），不是「已经是眼」：
                  直二做不出两只眼、三目要抢先手 ⇒ 都值 1 只；直四及以上无条件活 ⇒ 值 2 只。
    为什么换掉旧判据：只看单点真眼时，任何大眼空间（直二 / 方四 / 刀把五…）里每个点都有空的正邻、
    过不了判据，**整片记 0 眼** —— 实测 86% 的块恒为 0，通道几乎不携带信息（改前实测见
    `tools/srv/probe_eye_space.py`）。

    广义连接：**共享 >=2 个空点**的同色棋块并成一组（对手一手切不断，斜对角、双叫吃那类
    「没连成一块但本质上是一大块」的形状）；组内眼区**去重**后求和（两块共享的同一只眼只算一次，
    否则共活形状会被数成两只）。危急度仍按**原始块**算 —— 合并后的气数会重复计、反而失真。
    """
    nn = n * n
    labels = np.zeros((n, n), dtype=np.int32)      # 0=未访问/空，>0=块号
    lib_seen = np.zeros(nn, dtype=np.int32)        # 印记法去重（存块号）
    lib = np.zeros(nn + 1, dtype=np.int32)         # 每块的独立气数
    stack = np.zeros(nn, dtype=np.int32)
    nblk = 0
    for r in range(n):
        for c in range(n):
            color = board[r, c]
            if color == 0 or color == HOLE or labels[r, c] != 0:
                continue
            nblk += 1
            gid = nblk
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

    # ---- 空区域（眼区候选）：区域号 + 点列表，一次 BFS 拿全 ----
    # 存点列表是为了按区域枚举它的点；否则每判一个区域都要重扫全盘。
    reg_lab = np.full((n, n), -1, dtype=np.int32)
    reg_pts = np.zeros(nn, dtype=np.int32)
    reg_start = np.zeros(nn + 1, dtype=np.int32)
    reg_size = np.zeros(nn + 1, dtype=np.int32)
    nreg = 0
    npos = 0
    for r in range(n):
        for c in range(n):
            if board[r, c] != 0 or reg_lab[r, c] >= 0:
                continue
            nreg += 1
            rid = nreg
            reg_start[rid] = npos
            top = 0
            stack[top] = r * n + c
            top += 1
            reg_lab[r, c] = rid
            cnt = 0
            while top > 0:
                top -= 1
                cur = stack[top]
                reg_pts[npos] = cur
                npos += 1
                cnt += 1
                cr = cur // n
                cc = cur % n
                if cr > 0 and board[cr - 1, cc] == 0 and reg_lab[cr - 1, cc] < 0:
                    reg_lab[cr - 1, cc] = rid
                    stack[top] = (cr - 1) * n + cc
                    top += 1
                if cr < n - 1 and board[cr + 1, cc] == 0 and reg_lab[cr + 1, cc] < 0:
                    reg_lab[cr + 1, cc] = rid
                    stack[top] = (cr + 1) * n + cc
                    top += 1
                if cc > 0 and board[cr, cc - 1] == 0 and reg_lab[cr, cc - 1] < 0:
                    reg_lab[cr, cc - 1] = rid
                    stack[top] = cr * n + (cc - 1)
                    top += 1
                if cc < n - 1 and board[cr, cc + 1] == 0 and reg_lab[cr, cc + 1] < 0:
                    reg_lab[cr, cc + 1] = rid
                    stack[top] = cr * n + (cc + 1)
                    top += 1
            reg_size[rid] = cnt

    # ---- 广义连接：并查集 + 「同色块对 → 共享空点数」的哈希计数表 ----
    parent = np.arange(nblk + 1, dtype=np.int32)
    # njit 里没有 dict，用开放寻址哈希。只有被两块以上同色棋夹住的空点才会插入；
    # 19 路极端局面可能超过 512 个块对，给到 4096 避免静默漏合并（溢出只会漏特征，不会崩）。
    HS = 4096
    hkey = np.full(HS, -1, dtype=np.int32)
    hcnt = np.zeros(HS, dtype=np.int32)
    blk_l = np.empty(4, dtype=np.int32)
    blk_c = np.empty(4, dtype=np.int8)
    nbb = nblk + 1
    for r in range(n):
        for c in range(n):
            if board[r, c] != 0:
                continue
            k = 0
            if r > 0 and board[r - 1, c] != 0 and board[r - 1, c] != HOLE:
                blk_l[k] = labels[r - 1, c]
                blk_c[k] = board[r - 1, c]
                k += 1
            if r < n - 1 and board[r + 1, c] != 0 and board[r + 1, c] != HOLE:
                blk_l[k] = labels[r + 1, c]
                blk_c[k] = board[r + 1, c]
                k += 1
            if c > 0 and board[r, c - 1] != 0 and board[r, c - 1] != HOLE:
                blk_l[k] = labels[r, c - 1]
                blk_c[k] = board[r, c - 1]
                k += 1
            if c < n - 1 and board[r, c + 1] != 0 and board[r, c + 1] != HOLE:
                blk_l[k] = labels[r, c + 1]
                blk_c[k] = board[r, c + 1]
                k += 1
            # 同一个邻块可能从两个方向贴住这个空点（L 形包围）⇒ 块对会被重复枚举，把「共享空点数」
            # 虚高（实测把只共享 1 个空点的对算成 2，越过下面的 >=2 阈值）⇒ 先按 (色, 块号) 去重。
            kk = 0
            for i in range(k):
                dup = False
                for j in range(i):
                    if blk_c[j] == blk_c[i] and blk_l[j] == blk_l[i]:
                        dup = True
                        break
                if not dup:
                    blk_l[kk] = blk_l[i]
                    blk_c[kk] = blk_c[i]
                    kk += 1
            k = kk
            for i in range(k):
                for j in range(i + 1, k):
                    if blk_c[i] != blk_c[j]:       # 只并同色块：中间夹着对方子说明这两块并没连上
                        continue
                    a = blk_l[i]
                    b = blk_l[j]
                    if a == b:
                        continue
                    key = a * nbb + b if a < b else b * nbb + a
                    h = key % HS
                    step = 0
                    while step < HS:
                        if hkey[h] == -1:
                            hkey[h] = key
                            hcnt[h] = 1
                            break
                        if hkey[h] == key:
                            hcnt[h] += 1
                            break
                        h += 1
                        if h >= HS:
                            h = 0
                        step += 1
    for h in range(HS):
        if hkey[h] >= 0 and hcnt[h] >= 2:
            a = hkey[h] // nbb
            b = hkey[h] % nbb
            while parent[a] != a:
                a = parent[a]
            while parent[b] != b:
                b = parent[b]
            if a != b:
                parent[a] = b

    # ---- 眼区判定与折算，把等效眼数记到「组」上（组内按区域去重）----
    grp_eye = np.zeros(nblk + 1, dtype=np.int32)
    mark = np.zeros(nblk + 1, dtype=np.int32)      # 本区域已给哪些组加过眼（用戳号免清零）
    stamp = 0
    for rid in range(1, nreg + 1):
        s0 = reg_start[rid]
        e0 = s0 + reg_size[rid]
        # 归属色：区域的全部盘内邻点必须同色。出现第二种色 ⇒ 公共气；一个邻点都没有 ⇒ 整盘空。
        col = 0
        bad = False
        for i in range(s0, e0):
            cur = reg_pts[i]
            cr = cur // n
            cc = cur % n
            if cr > 0 and board[cr - 1, cc] != 0:
                v = board[cr - 1, cc]
                if v != HOLE:
                    if col == 0:
                        col = v
                    elif v != col:
                        bad = True
                        break
            if cr < n - 1 and board[cr + 1, cc] != 0:
                v = board[cr + 1, cc]
                if v != HOLE:
                    if col == 0:
                        col = v
                    elif v != col:
                        bad = True
                        break
            if cc > 0 and board[cr, cc - 1] != 0:
                v = board[cr, cc - 1]
                if v != HOLE:
                    if col == 0:
                        col = v
                    elif v != col:
                        bad = True
                        break
            if cc < n - 1 and board[cr, cc + 1] != 0:
                v = board[cr, cc + 1]
                if v != HOLE:
                    if col == 0:
                        col = v
                    elif v != col:
                        bad = True
                        break
        if bad or col == 0:
            continue
        if reg_size[rid] == 1:
            cur = reg_pts[s0]
            if _eye_color(board, n, cur // n, cur % n) == 0:
                continue                           # 假眼（对角条件不满足）⇒ 不计
            e = 1
        elif reg_size[rid] <= 5:
            e = 1
        else:
            e = 2
        stamp += 1
        for i in range(s0, e0):
            cur = reg_pts[i]
            cr = cur // n
            cc = cur % n
            if cr > 0 and board[cr - 1, cc] == col:
                g = labels[cr - 1, cc]
                while parent[g] != g:
                    g = parent[g]
                if mark[g] != stamp:
                    mark[g] = stamp
                    grp_eye[g] += e
            if cr < n - 1 and board[cr + 1, cc] == col:
                g = labels[cr + 1, cc]
                while parent[g] != g:
                    g = parent[g]
                if mark[g] != stamp:
                    mark[g] = stamp
                    grp_eye[g] += e
            if cc > 0 and board[cr, cc - 1] == col:
                g = labels[cr, cc - 1]
                while parent[g] != g:
                    g = parent[g]
                if mark[g] != stamp:
                    mark[g] = stamp
                    grp_eye[g] += e
            if cc < n - 1 and board[cr, cc + 1] == col:
                g = labels[cr, cc + 1]
                while parent[g] != g:
                    g = parent[g]
                if mark[g] != stamp:
                    mark[g] = stamp
                    grp_eye[g] += e

    for r in range(n):
        for c in range(n):
            b = board[r, c]
            if b == 0 or b == HOLE:
                continue
            gid = labels[r, c]
            g = gid
            while parent[g] != g:
                g = parent[g]
            v = (4 - min(lib[gid], 4)) / 3.0       # 危急度仍按**原始块**的气数（合并后会重复计）
            ee = grp_eye[g]
            ev = 0.0 if ee == 0 else (0.5 if ee == 1 else 1.0)
            if b == player:
                own_urg[r, c] = v
                own_eye[r, c] = ev
            else:
                opp_urg[r, c] = v
                opp_eye[r, c] = ev

@njit(cache=True)
def _canonical_state(board, st, n, komi, player, out, boundary):
    """当前玩家视角规范输入（训练/自对弈/MCTS/GUI 全链路共用同一编码）。
    各通道下标见 hyperparams 的 CH_* 常量 —— 布局只在那边定义一处，这里只负责填值。
    通道2=上一手位置；通道3=贴目标量 kch = player*komi/14（player=当前玩家色，黑+1/白-1；
        符号按俗称“黑贴X目”的贴目方向约定：黑行棋=+komi（黑欠白komi，黑须净胜过贴目），
        白行棋=-komi（反向，即“白倒贴”视角，白含komi优势）；网络学到的语义
        =当前玩家需克服的贴目，编码全链路自洽。若按“受益方”约定取反（白+黑-）亦可，
        但改符号必须重训，勿单独改动！）
    合法点掩码不在本函数内填 —— 它的来源（_legal_moves_and_mask）在热路径上
    每个节点本来就要算一次，由调用方复用、见 get_canonical_state。"""
    kch = np.float32(player * komi / 14.0)
    last_valid = st[1] >= 0
    # 边界通道由 GoGame 每局预生成：洞 -1、内外边界 +1、其余可落子点 0。
    out[CH_BOUNDARY] = boundary
    for r in range(n):
        for c in range(n):
            out[CH_OWN_STONES, r, c] = 1.0 if board[r, c] == player else 0.0
            out[CH_OPP_STONES, r, c] = 1.0 if board[r, c] == -player else 0.0
            out[CH_LAST_MOVE, r, c] = 1.0 if (last_valid and st[1] == r and st[2] == c) else 0.0
            out[CH_KOMI, r, c] = kch
    # 危急度与眼都由棋块（同色连通）决定，一次遍历里一起算完
    _group_channels(board, n, player,
                    out[CH_OWN_URGENCY], out[CH_OPP_URGENCY],
                    out[CH_OWN_EYES], out[CH_OPP_EYES])


@njit(cache=True)
def _dead_lump_mask(board, n, out):
    """终局清理：标出「子数 <= 2」或「独立气数 == 1」的棋块（out 为 True 的格子要移除）。

    为什么只清这两类：终局盘上的死子/残子绝大多数就是孤立一两颗、或者已经被打吃只剩一气，
    而这两条判据够硬，不会伤到活棋。再往上做攻杀就得假设「防守方全程不落子」，
    那等价于「活得下来 ≈ 有两只真眼」，会大面积过度判死。
    只做一遍即可：移除小块只会让邻块的气变多，不会制造出新的待清块。
    """
    labels = np.zeros((n, n), dtype=np.int32)     # 0=未访问/空，>0=组号
    lib_seen = np.zeros(n * n, dtype=np.int32)    # 印记法去重（存组号）
    lib = np.zeros(n * n + 1, dtype=np.int32)
    gsize = np.zeros(n * n + 1, dtype=np.int32)
    stack = np.zeros(n * n, dtype=np.int32)
    next_gid = 1
    for r in range(n):
        for c in range(n):
            color = board[r, c]
            if color == 0 or color == HOLE or labels[r, c] != 0:
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
                gsize[gid] += 1
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
            if board[r, c] == 0 or board[r, c] == HOLE:
                continue
            gid = labels[r, c]
            out[r, c] = (gsize[gid] <= 2) or (lib[gid] <= 1)


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
                    elif v == HOLE:
                        pass
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
                    elif v == HOLE:
                        pass
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
                    elif v == HOLE:
                        pass
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
                    elif v == HOLE:
                        pass
                    elif v == 1:
                        r_has_b[nr] = True
                    else:
                        r_has_w[nr] = True
    return nr


class GoGame:
    """numba 加速版围棋（API 与 Python 版完全一致）"""

    def __init__(self, board_size=BOARD_SIZE, komi=KOMI, hollow=0):
        self.board_size = int(board_size)
        self.komi = float(komi)
        self.hollow = validate_hollow(self.board_size, hollow)
        n = self.board_size
        h = self.hollow
        self._hole_lo = (n - h) // 2 if h else n
        self._hole_hi = self._hole_lo + h if h else n
        self._boundary = boundary_plane(n, h)
        self._simulate_mode = False
        self.reset()

    def reset(self):
        n = self.board_size
        rules = board_rules(n, self.hollow)
        self.board = np.zeros((n, n), dtype=np.int8)
        if self.hollow:
            self.board[self._hole_lo:self._hole_hi, self._hole_lo:self._hole_hi] = HOLE
        # [13:15] 当前局面 Zobrist 哈希，[15:17] 上一局面哈希；
        # [17] 全历史局面条数。劫判定使用全历史（位置超级劫）。
        self._st = np.zeros(19, dtype=np.int32)
        self._st[0] = 1                    # 当前玩家
        self._st[1] = -2                   # 上一步 r（-2=None）
        self._st[2] = -2
        self._st[3] = -2                   # 劫点 r
        self._st[4] = -2
        self._st[9] = -2                   # winner（-2=None）
        self._st[10] = rules["min_moves"]  # 前 N 手不允许 pass
        self._st[11] = rules["max_moves"]  # 单局最大手数
        self._st[12] = rules["pass_limit"] # 单方累计 pass 上限
        self._st[17] = 1                   # 历史里已有空盘
        self._hist = np.zeros(rules["max_moves"] + 2, dtype=np.uint64)
        self._go = np.zeros(1, dtype=np.bool_)
        self._fp = np.zeros(1, dtype=np.float64)
        self._cap = np.zeros(self.board_size * self.board_size, dtype=np.int32)

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

    # ---- 热路径（numba）----
    def forced_pass(self):
        if self._st[0] == 1:
            return self._st[8] >= self._st[12]
        return self._st[7] >= self._st[12]

    def is_legal_move(self, row, col):
        if self._go[0]:
            return False
        if (row, col) == PASS_MOVE:
            return self._st[5] >= self._st[10]
        n = self.board_size
        if not (0 <= row < n and 0 <= col < n):
            return False
        if self.forced_pass():
            return False
        cur_h, prev_h = _hash_from_slots(self._st)
        hist_n = _sync_history(self._st, self._hist, cur_h)
        return _is_legal_impl(self.board, row, col, self._st[0],
                              self._st[3], self._st[4], cur_h, prev_h,
                              True, self.board_size, self._hist, hist_n)

    def make_move(self, row, col, check_legal=False):
        if check_legal and not self.is_legal_move(row, col):
            return False
        ok = _make_move(self.board, self._st, self._go, self._fp,
                        row, col, self._cap, self._hist)
        if not ok:
            return False
        if self._go[0] and not self._simulate_mode and self.winner is None:
            self._compute_winner()
        return True

    def get_legal_indices_and_mask(self):
        """热路径：直接返回扁平着法下标 + 合法掩码。

        与 get_legal_moves_and_mask 的点集和顺序完全一致，只是跳过
        “整数下标 -> (r,c) tuple -> 整数下标”的往返。MCTS 的 _backup_np
        只需要扁平下标，用本接口可以省掉每个叶子数百次 tuple 构造。
        """
        n = self.board_size
        total = n * n
        moves = np.zeros(total + 1, dtype=np.int32)
        mask = np.zeros(total + 1, dtype=np.float32)
        cnt = _legal_moves_and_mask(self.board, self._st, n, moves, mask,
                                    self._hist)
        return moves[:cnt].copy(), mask

    def get_legal_moves_and_mask(self):
        n = self.board_size
        total = n * n
        indices, mask = self.get_legal_indices_and_mask()
        out = []
        for i in indices:
            if i == total:
                out.append(PASS_MOVE)
            else:
                out.append((int(i) // n, int(i) % n))
        return out, mask

    def get_legal_mask(self):
        """只取合法点掩码（n*n+1，末位 n*n 是 pass）。给手上没有掩码的调用方兜底；
        热路径（自对弈/MCTS/GUI）本来就算过一次 _legal_moves_and_mask，应把结果传进来复用。"""
        n = self.board_size
        total = n * n
        moves = np.zeros(total + 1, dtype=np.int32)
        mask = np.zeros(total + 1, dtype=np.float32)
        _legal_moves_and_mask(self.board, self._st, n, moves, mask,
                              self._hist)
        return mask

    def get_canonical_state(self, player=None, legal_mask=None):
        if player is None:
            player = self._st[0]
        n = self.board_size
        out = np.zeros((INPUT_CHANNELS, n, n), dtype=np.float32)
        _canonical_state(self.board, self._st, n, self.komi, int(player), out,
                         self._boundary)
        if legal_mask is None:
            if int(player) == int(self._st[0]):
                legal_mask = self.get_legal_mask()
            else:
                # 允许指定对手视角：合法点必须按该玩家重算（自杀判定随颜色变）。
                st = self._st.copy()
                st[0] = int(player)
                total = n * n
                moves = np.zeros(total + 1, dtype=np.int32)
                legal_mask = np.zeros(total + 1, dtype=np.float32)
                # 传历史副本：st 是临时副本，_sync_history 不应改动本局历史计数。
                _legal_moves_and_mask(self.board, st, n, moves, legal_mask,
                                      self._hist.copy())
        # 掩码是 n*n+1、末位为 pass：19×19 平面装不下这一位，只取前 n*n 位。
        # 丢掉它不算信息回退 —— pass 合法性由步数下限决定，此前也从未进过任何通道。
        out[CH_LEGAL_MASK] = legal_mask[:n * n].reshape(n, n)
        return out


    def is_hole(self, row, col):
        n = self.board_size
        return (0 <= row < n and 0 <= col < n
                and self.board[row, col] == HOLE)

    @property
    def hole_lo(self):
        return int(self._hole_lo)

    @property
    def hole_hi(self):
        return int(self._hole_hi)

    @property
    def valid_mask(self):
        """可落子点掩码（洞为 False；内外边界都算可落子）。"""
        return self._boundary >= 0.0

    def copy(self):
        g = GoGame.__new__(GoGame)
        g.board_size = self.board_size
        g.komi = self.komi
        g.hollow = self.hollow
        g._hole_lo = self._hole_lo
        g._hole_hi = self._hole_hi
        g._boundary = self._boundary.copy()
        g.board = self.board.copy()
        g._st = self._st.copy()
        g._hist = self._hist.copy()
        g._go = self._go.copy()
        g._fp = self._fp.copy()
        g._cap = np.zeros(self.board_size * self.board_size, dtype=np.int32)
        g._simulate_mode = False
        return g

    # ---- 以下冷路径沿用纯 Python（只在终局/复盘时调用，不影响速度）----
    def compute_score(self):
        data = self._merged_analysis()
        # 中立区域（黑白边界）：归属0，目数黑白各0.5（目差中相互抵消）
        black_score = data['black_stones'] + data['tb'] + 0.5 * data['neutral']
        white_score = data['white_stones'] + data['tw'] + 0.5 * data['neutral'] + self.komi
        return black_score, white_score, data['final_diff']

    def _compute_winner(self):
        self.final_points = self.compute_score()[2]
        if self.final_points > 0:
            self.winner = 1
        elif self.final_points < 0:
            self.winner = -1
        else:
            self.winner = 0

    # ---- 终局判定（Tromp-Taylor 式数目 + 小块清理）----
    def terminal_analysis(self):
        """终局完整分析（训练标签用）：
        盘上活子 + 空区域 flood fill 得到领地/归属，更新 final_points / winner，
        返回绝对归属图 own(H,W int8)。"""
        data = self._merged_analysis()
        self._fp[0] = float(data['final_diff'])
        self._st[9] = 1 if data['final_diff'] > 0 else (-1 if data['final_diff'] < 0 else 0)
        return data['own']

    def _merged_analysis(self):
        """终局领地/归属分析：所有棋子算活，只对空区域做 flood fill。

        只挨黑=黑地、只挨白=白地、两种都挨=中立（归属 0，目数黑白各 0.5，
        目差中相互抵消）。数目之前可选移除「子数 <= 2」或「独立气数 == 1」
        的残块，避免终局死子污染标签。
        """
        board = self.board
        n = self.board_size
        clean = board.copy()
        if PRUNE_DEAD_LUMPS:
            dead = np.zeros((n, n), dtype=np.bool_)
            _dead_lump_mask(board, n, dead)
            clean[dead] = 0
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
        # 归属图：活子±1；黑/白独占区域±1；中立=0；洞=0
        own = clean.copy().astype(np.int8)
        own[clean == HOLE] = 0
        for r in range(n):
            for c in range(n):
                rid = r_label[r, c]
                if rid > 0:
                    if r_has_b[rid] and not r_has_w[rid]:
                        own[r, c] = 1
                    elif r_has_w[rid] and not r_has_b[rid]:
                        own[r, c] = -1
        return {
            'own': own,
            'tb': tb, 'tw': tw, 'neutral': neutral,
            'black_stones': black_stones, 'white_stones': white_stones,
            'final_diff': (black_stones + tb) - (white_stones + tw + self.komi),
        }

    def __str__(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        s = "   " + " ".join(f"{i:2d}" for i in range(self.board_size)) + "\n"
        for i in range(self.board_size):
            line = []
            for j in range(self.board_size):
                v = self.board[i, j]
                line.append('#' if v == HOLE else symbols[v])
            s += f"{i:2d} " + "  ".join(line) + "\n"
        return s
