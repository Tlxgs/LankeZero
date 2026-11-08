"""
围棋游戏逻辑
"""
import numpy as np
from numba import njit

import shapes
from hyperparams import (BOARD_SIZE, KOMI, PASS_MOVE, HOLE, MAX_BOARD_CELLS,
                         board_rules_for_area, validate_komi,
                         PRUNE_DEAD_LUMPS,
                         INPUT_CHANNELS, GLOBAL_DIM, G_KOMI, KOMI_SCALE,
                         G_OWN_STONES, G_OPP_STONES, G_OWN_CAPS, G_OPP_CAPS, G_AREA,
                         CH_OWN_STONES, CH_OPP_STONES, CH_LAST_MOVE,
                         CH_OWN_URGENCY, CH_OPP_URGENCY, CH_LEGAL_MASK,
                         CH_BOUNDARY,
                         CH_OWN_LADDER, CH_OPP_LADDER,
                         CH_OWN_ALIVE, CH_OPP_ALIVE)


def boundary_plane_from_mask(hole):
    """由形状掩码生成边界输入通道：洞内 -1、边界 +1、其余可落子点 0。

    规则与旧的「居中方块洞」版逐点一致：
    - 盘外边界只取最外圈（4 邻域语义，实心盘的编码一点不变）；
    - 洞的外轮廓取 8 邻域 —— 旧版画的是「hollow+2 空心矩形的最外圈」，
      包含四个对角点，等价于切比雪夫距离 1 的一圈。

    换成掩码之后，心形、圆环、多个洞、贴边缺口都走同一条路径。
    """
    hole = np.asarray(hole, dtype=bool)
    n = hole.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    if hole.any():
        pad = np.pad(hole, 1, constant_values=False)
        near = np.zeros((n, n), dtype=bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    near |= pad[1 + dr:1 + dr + n, 1 + dc:1 + dc + n]
        out[~hole & near] = 1.0
    out[0, :] = 1.0
    out[-1, :] = 1.0
    out[:, 0] = 1.0
    out[:, -1] = 1.0
    out[hole] = -1.0            # 洞贴着盘边时，别被最外圈覆盖
    return out


def boundary_plane(board_size, hole_mask=None):
    """兼容入口：给掩码就按掩码画，不给就是实心盘。"""
    n = int(board_size)
    if hole_mask is None:
        return boundary_plane_from_mask(np.zeros((n, n), dtype=bool))
    return boundary_plane_from_mask(shapes.validate_mask(hole_mask, n))

# Zobrist 表：用于「全历史局面」精确比对（位置超级劫）。
# 颜色下标 0=黑、1=白；扁平下标 r*n+c，覆盖当前支持的最大棋盘。
_ZOBRIST = np.random.RandomState(0x5EED).randint(
    1, 2**63 - 1, size=(2, MAX_BOARD_CELLS), dtype=np.uint64)
_MASK32 = np.uint64(0xFFFFFFFF)


@njit(cache=True)
def _hash_from_slots(st):
    """从 _st[8:10] 取当前局面哈希。"""
    return ((np.uint64(st[8]) & _MASK32)
            | ((np.uint64(st[9]) & _MASK32) << np.uint64(32)))


@njit(cache=True)
def _set_hash_slots(st, cur):
    st[8] = np.int32(cur & _MASK32)
    st[9] = np.int32((cur >> np.uint64(32)) & _MASK32)


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
    hist_n = st[10]
    if hist_n <= 0 or hist[hist_n - 1] != cur_h:
        if hist_n < hist.shape[0]:
            hist[hist_n] = cur_h
            hist_n += 1
            st[10] = hist_n
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
def _is_legal_impl(board, row, col, player, cur_h, n, hist, hist_n):
    """等价 Python _is_legal_move（临时落子后恢复，不污染棋盘）。

    用「落子后的局面哈希」与全历史比对判位置超级劫（位置超级劫已覆盖普通劫，
    所以不需要另外的劫点判据）。非提子着法不可能重现旧局面，因此只在 captured>0
    时查历史。
    """
    if board[row, col] != 0:
        return False
    board[row, col] = player
    cap = np.zeros(n * n, dtype=np.int32)
    captured = _capture_opponent(board, row, col, player, n, cap)
    board[row, col] = 0
    if captured > 0:
        ci = 0 if player == 1 else 1
        opp = 1 - ci
        h = cur_h ^ _ZOBRIST[ci, row * n + col]
        for j in range(captured):
            h ^= _ZOBRIST[opp, cap[j]]
        if hist_n > 0 and _history_contains(hist, hist_n, h):
            return False                          # 重现历史局面 = 超级劫
    if captured == 0:
        board[row, col] = player
        ok = _group_has_liberty(board, row, col, n)
        board[row, col] = 0
        return ok                                 # 自杀判定
    return True


def position_legal_mask(board, player, n):
    """不含超级劫的合法性掩码（只看提子/自杀），供界面手动标注旧数据用。

    旧 npz 的 `legals` 掩码可能带着已废弃的「pass 上限」限制（达到上限时落子全禁），
    标注重算时不能直接信它。`board` 用行棋方视角（+1 = 行棋方棋子）。
    """
    board = np.asarray(board, dtype=np.int8)
    n = int(n)
    player = int(player)
    out = np.zeros((n, n), dtype=bool)
    cap = np.zeros(n * n, dtype=np.int32)
    for r in range(n):
        for c in range(n):
            if board[r, c] != 0:
                continue
            b = board.copy()
            b[r, c] = player
            captured = int(_capture_opponent(b, r, c, player, n, cap))
            for j in range(captured):
                b[int(cap[j]) // n, int(cap[j]) % n] = 0
            out[r, c] = bool(_group_has_liberty(b, r, c, n))
    return out

@njit(cache=True)
def _make_move(board, st, go, fp, row, col, cap, hist):
    """落子（含 Pass/提子/超级劫检测），非法超级劫着法会回滚并返回 False。
    st: [0]当前玩家 [1]上一步r [2]上一步c
        [3]步数 [4]连续pass [5]winner(-2=None)
        [6]min_moves [7]max_moves
        [8:10]当前局面 Zobrist 哈希 [10]历史局面条数
        [11]/[12] 黑白子数 [13]/[14] 黑白累计提子
    """
    n = board.shape[0]
    cur_h = _hash_from_slots(st)
    hist_n = _sync_history(st, hist, cur_h)
    if row == -1 and col == -1:                   # Pass
        st[1] = -1
        st[2] = -1
        st[4] += 1
        st[3] += 1
        st[0] = -st[0]
        _set_hash_slots(st, cur_h)
        if st[4] >= 2 or st[3] >= st[7]:
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
    st[1] = row
    st[2] = col
    st[4] = 0
    st[3] += 1
    # 增量维护的棋子/提子计数（放在超级劫回滚之后，回滚路径不用撤销）
    st[11 + ci] += 1
    if captured > 0:
        st[11 + (1 - ci)] -= captured
        st[13 + ci] += captured
    _set_hash_slots(st, h)
    if hist_n < hist.shape[0]:                     # 记录本手后的局面
        hist[hist_n] = h
        st[10] = hist_n + 1
    st[0] = -st[0]          # 终局后 current_player = 下一手方，终局价值口径与双 Pass 一致
    if st[3] >= st[7]:
        go[0] = True
    return True

@njit(cache=True)
def _make_move_no_capture(board, st, go, fp, row, col, cap, hist, win_len):
    """无提子/无气/无劫的落子（五子棋这类「占位 + 连子判胜」的规则用）。

    win_len <= 0 表示不计胜负（纯占位）；>=2 时「同色连成 win_len 或更长」即胜。
    其余约定与 _make_move 一致：落子后换手、st[3] 计手、终局置 go[0]。
    """
    if row < 0 or col < 0:
        return False
    n = board.shape[0]
    if board[row, col] != 0:
        return False
    player = st[0]
    board[row, col] = player
    st[1] = row
    st[2] = col
    st[3] += 1
    if win_len >= 2:
        won = False
        for k in range(4):
            if k == 0:
                dr, dc = 0, 1
            elif k == 1:
                dr, dc = 1, 0
            elif k == 2:
                dr, dc = 1, 1
            else:
                dr, dc = 1, -1
            cnt = 1
            rr, cc = row + dr, col + dc
            while 0 <= rr < n and 0 <= cc < n and board[rr, cc] == player:
                cnt += 1
                rr += dr
                cc += dc
            rr, cc = row - dr, col - dc
            while 0 <= rr < n and 0 <= cc < n and board[rr, cc] == player:
                cnt += 1
                rr -= dr
                cc -= dc
            if cnt >= win_len:
                won = True
                break
        if won:
            go[0] = True
            st[5] = player
    if not go[0] and st[3] >= n * n:
        go[0] = True
        st[5] = 0
    st[0] = -player
    return True


@njit(cache=True)
def _make_move_dispatch(board, st, go, fp, cap, hist, row, col, kind, win_len):
    """MCTS 用的落子入口：kind=0 围棋（提子/劫/超级劫），kind=1 无提子连子规则。

    为什么不用「把函数当参数传进来」：Numba 的磁盘缓存会把函数类型写进索引再
    pickle，Dispatcher 不可序列化，cache=True 时保存索引直接抛
    ReferenceError（GUI 里表现为搜索线程静默停掉），只能靠关缓存绕开，
    代价是每个进程首次搜索要重编译 10 秒以上。改成按 kind 分支后，
    所有 numba 函数都能正常缓存。
    """
    if kind == 0:
        return _make_move(board, st, go, fp, row, col, cap, hist)
    return _make_move_no_capture(board, st, go, fp, row, col, cap, hist, win_len)


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
    """合法着法；moves: int32缓冲(n*n+1)，mask: float32(n*n+1)

    超级劫是唯一的重复局面判据（已覆盖普通劫）。提子点自己有气也可能提子，
    所以不能只走「有气」快速路径，必须用 _move_captures 判定是否真的提子。
    """
    total = n * n
    mask[:] = 0.0
    count = 0
    cur_h = _hash_from_slots(st)
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
            if has_air:
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
            if _is_legal_impl(board, r, c, st[0], cur_h, n, hist, hist_n):
                moves[count] = i
                count += 1
                mask[i] = 1.0
    if st[3] >= st[6]:
        moves[count] = total
        count += 1
        mask[total] = 1.0
    return count


@njit(cache=False)
def _ladder_chain(board, n, r, c, pr, pc, lr, lc):
    color = board[r, c]
    if color != 1 and color != -1:
        return 0, 0
    seen = np.zeros((n, n), dtype=np.uint8)
    stack = np.zeros(n * n, dtype=np.int32)
    stack[0] = r * n + c
    top = 1
    seen[r, c] = 1
    np_ = 0
    nl = 0
    while top > 0:
        top -= 1
        cur = stack[top]
        cr = cur // n
        cc = cur % n
        pr[np_] = cr
        pc[np_] = cc
        np_ += 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = cr + dr
            nc = cc + dc
            if nr < 0 or nr >= n or nc < 0 or nc >= n:
                continue
            v = board[nr, nc]
            if v == 0:
                dup = False
                for k in range(nl):
                    if lr[k] == nr and lc[k] == nc:
                        dup = True
                        break
                if not dup:
                    lr[nl] = nr
                    lc[nl] = nc
                    nl += 1
            elif v == color and seen[nr, nc] == 0:
                seen[nr, nc] = 1
                stack[top] = nr * n + nc
                top += 1
    return np_, nl


@njit(cache=False)
def _ladder_play(board, n, r, c, color, opr, opc, olr, olc):
    b = board.copy()
    b[r, c] = color
    captured = 0
    seen = np.zeros((n, n), dtype=np.uint8)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if nr < 0 or nr >= n or nc < 0 or nc >= n:
            continue
        if b[nr, nc] != -color or seen[nr, nc] != 0:
            continue
        np2, nl2 = _ladder_chain(b, n, nr, nc, opr, opc, olr, olc)
        for k in range(np2):
            seen[opr[k], opc[k]] = 1
        if nl2 == 0:
            for k in range(np2):
                b[opr[k], opc[k]] = 0
            captured += np2
    np3, nl3 = _ladder_chain(b, n, r, c, opr, opc, olr, olc)
    if nl3 == 0:
        return b, captured, False
    return b, captured, True


@njit(cache=False)
def _ladder_dfs(board, n, dr0, dc0, ar0, ac0, color, turn, depth, max_depth,
                pr, pc, lr, lc, qr, qc, opr, opc, olr, olc):
    if depth > max_depth:
        return 0
    np_, nl_ = _ladder_chain(board, n, dr0, dc0, pr, pc, lr, lc)
    if np_ == 0 or nl_ == 0:
        return -1
    if nl_ >= 3:
        return 1

    # 递归会复用 pr/pc/lr/lc；先把自己的气拷出来，否则下一轮读到的是子节点的数据。
    mr = np.zeros(n * n, dtype=np.int32)
    mc = np.zeros(n * n, dtype=np.int32)
    for i in range(nl_):
        mr[i] = lr[i]
        mc[i] = lc[i]

    if turn == color:
        if nl_ != 1:
            return 0
        cand_r = np.zeros(n * n, dtype=np.int32)
        cand_c = np.zeros(n * n, dtype=np.int32)
        cand_r[0] = mr[0]
        cand_c[0] = mc[0]
        ncand = 1
        seen_att = np.zeros((n, n), dtype=np.uint8)
        for k in range(np_):
            sr = pr[k]
            sc = pc[k]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = sr + dr
                nc = sc + dc
                if nr < 0 or nr >= n or nc < 0 or nc >= n:
                    continue
                if board[nr, nc] != -color or seen_att[nr, nc] != 0:
                    continue
                anp, anl = _ladder_chain(board, n, nr, nc, opr, opc, olr, olc)
                for t in range(anp):
                    seen_att[opr[t], opc[t]] = 1
                if anl == 1:
                    ar = olr[0]
                    ac = olc[0]
                    dup = False
                    for t in range(ncand):
                        if cand_r[t] == ar and cand_c[t] == ac:
                            dup = True
                            break
                    if not dup:
                        cand_r[ncand] = ar
                        cand_c[ncand] = ac
                        ncand += 1
        unknown = False
        for t in range(ncand):
            b2, cap, legal = _ladder_play(board, n, cand_r[t], cand_c[t], color,
                                          opr, opc, olr, olc)
            if not legal:
                continue
            np2, nl2 = _ladder_chain(b2, n, dr0, dc0, pr, pc, lr, lc)
            if np2 == 0:
                continue
            if nl2 >= 3:
                return 1
            r2 = _ladder_dfs(b2, n, dr0, dc0, ar0, ac0, color, -turn,
                             depth + 1, max_depth, pr, pc, lr, lc,
                             qr, qc, opr, opc, olr, olc)
            if r2 == 1:
                return 1
            if r2 == 0:
                unknown = True
        return 0 if unknown else -1

    if nl_ > 2:
        return 0
    unknown = False
    for k in range(nl_):
        b2, cap, legal = _ladder_play(board, n, mr[k], mc[k], turn,
                                      opr, opc, olr, olc)
        if not legal:
            continue
        np2, nl2 = _ladder_chain(b2, n, dr0, dc0, pr, pc, lr, lc)
        if np2 == 0:
            return -1
        if cap > 0:
            unknown = True
            continue
        if nl2 != 1:
            continue
        # 攻方自己只剩 1 气：这是自打吃，不是征子应手，整条分支直接剪掉，
        # 不递归、也不记为 UNKNOWN。
        anp, anl = _ladder_chain(b2, n, mr[k], mc[k], opr, opc, olr, olc)
        if anl <= 1:
            continue
        r2 = _ladder_dfs(b2, n, dr0, dc0, mr[k], mc[k], color, color,
                         depth + 1, max_depth, pr, pc, lr, lc,
                         qr, qc, opr, opc, olr, olc)
        if r2 == -1:
            return -1
        if r2 == 0:
            unknown = True
    return 0 if unknown else 1


@njit(cache=False)
def _ladder_status(board, n, dr0, dc0, turn, max_depth):
    cap = n * n
    pr = np.zeros(cap, dtype=np.int32)
    pc = np.zeros(cap, dtype=np.int32)
    lr = np.zeros(cap, dtype=np.int32)
    lc = np.zeros(cap, dtype=np.int32)
    qr = np.zeros(cap, dtype=np.int32)
    qc = np.zeros(cap, dtype=np.int32)
    opr = np.zeros(cap, dtype=np.int32)
    opc = np.zeros(cap, dtype=np.int32)
    olr = np.zeros(cap, dtype=np.int32)
    olc = np.zeros(cap, dtype=np.int32)
    np_, nl_ = _ladder_chain(board, n, dr0, dc0, pr, pc, lr, lc)
    if np_ == 0 or nl_ != 1:
        return 0
    color = board[dr0, dc0]
    if turn != color:
        return -1
    return _ladder_dfs(board, n, dr0, dc0, dr0, dc0, color, turn, 0,
                       max_depth, pr, pc, lr, lc, qr, qc,
                       opr, opc, olr, olc)


@njit(cache=False)
def _ladder_channels(board, n, player, own_ladder, opp_ladder):
    """给所有「只剩 1 气」的棋块写征子状态。

    每个块都按「该块自己先走」求值：己方通道假设当前方先走，对方通道假设对方先走。
    这样对方通道也会出现 +1（假设对方先走能逃），而不是一律 -1。
    """
    own_ladder[:, :] = 0.0
    opp_ladder[:, :] = 0.0
    labels = np.zeros((n, n), dtype=np.int32)
    lib = np.zeros(n * n + 1, dtype=np.int32)
    rep_r = np.zeros(n * n + 1, dtype=np.int32)
    rep_c = np.zeros(n * n + 1, dtype=np.int32)
    gcol = np.zeros(n * n + 1, dtype=np.int8)
    lib_seen = np.zeros(n * n, dtype=np.int32)
    stack = np.zeros(n * n, dtype=np.int32)
    status = np.zeros(n * n + 1, dtype=np.int8)
    nblk = 0
    for r in range(n):
        for c in range(n):
            color = board[r, c]
            if color != 1 and color != -1:
                continue
            if labels[r, c] != 0:
                continue
            nblk += 1
            gid = nblk
            gcol[gid] = color
            rep_r[gid] = r
            rep_c[gid] = c
            stack[0] = r * n + c
            top = 1
            labels[r, c] = gid
            while top > 0:
                top -= 1
                cur = stack[top]
                cr = cur // n
                cc = cur % n
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if nr < 0 or nr >= n or nc < 0 or nc >= n:
                        continue
                    v = board[nr, nc]
                    if v == 0:
                        pos = nr * n + nc
                        if lib_seen[pos] != gid:
                            lib_seen[pos] = gid
                            lib[gid] += 1
                    elif v == color and labels[nr, nc] == 0:
                        labels[nr, nc] = gid
                        stack[top] = nr * n + nc
                        top += 1

    for gid in range(1, nblk + 1):
        if lib[gid] != 1:
            continue
        # 两个通道都按「该棋块自己先走」求值：己方通道=当前方先走，
        # 对方通道=假设对方先走、他能不能逃。不要按当前行棋方直接判 -1。
        status[gid] = _ladder_status(board, n, rep_r[gid], rep_c[gid],
                                     gcol[gid], 4 * n)

    for r in range(n):
        for c in range(n):
            gid = labels[r, c]
            if gid == 0 or status[gid] == 0:
                continue
            if gcol[gid] == player:
                own_ladder[r, c] = status[gid]
            else:
                opp_ladder[r, c] = status[gid]


@njit(cache=True)
def _urgency_channels(board, n, player, own_urg, opp_urg):
    """按棋块（同色 4-连通）算危急度，写进两个平面（下标见 hyperparams 的 CH_*）：

    1 气 = 1.0 最危急，>=4 气 = 0 安全；空点与非本颜色为 0。

    眼通道已删除：原来那套「眼区折算」在复杂死活里错误率很高，模型照着学等于把
    启发式的错误固化成输入（消融：去掉只掉 2~4% 损失，目差 MAE +1.5 目）。
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

    for r in range(n):
        for c in range(n):
            b = board[r, c]
            if b == 0 or b == HOLE:
                continue
            v = (4 - min(lib[labels[r, c]], 4)) / 3.0
            if b == player:
                own_urg[r, c] = v
            else:
                opp_urg[r, c] = v

@njit(cache=True)
def _benson_color(board, n, color, out):
    """把 color 方的 Benson「无条件活」（pass-alive）棋子写进 out（1.0 / 0.0）。

    定义（Benson 1976，与「对手全 pass 也杀不掉」等价）：
      黑棋包围区域 R = 不含 color 子、且边界邻点全是 color 子的区域
                       （R 取「非 color、非洞」的连通分量，边界条件自动成立）。
      R 是串 s 的关键区域 ⇔ ①R 是包围区域；②与 R 相邻的每个 color 串都在 S 里；
                              ③R 内每个点要么是对方子，要么是 s 的气。
      S 无条件活 ⇔ S 中每个串都至少有 2 个关键区域。
    Benson 算法：从全部串开始，反复删掉「关键区域 < 2 个」的串，直到不动点；
    剩下的就是最大无条件活集合。

    实现要点：
      - 条件③用「空点相邻串集合的交集」判定，每个空点最多 4 个相邻串，
        区域只需做 ≤4 元小集合求交，避免 O(|R|×|S|)。
      - 删串会杀死「与它相邻的所有区域」（条件②是区域级的），用队列级联传播，
        每个串/区域最多处理一次，整体 O(N + 边数)。
    本引擎禁止自杀 ⇒ 用标准（宽）关键区域。
    """
    N = n * n
    out[:, :] = 0.0

    # ---- 1. 串标记（所有棋子；同色 4-连通）----
    label = np.full((n, n), -1, dtype=np.int32)
    s_color = np.full(N, 2, dtype=np.int8)
    stack = np.zeros(N, dtype=np.int32)
    ns = 0
    for r0 in range(n):
        for c0 in range(n):
            v = board[r0, c0]
            if (v != 1 and v != -1) or label[r0, c0] != -1:
                continue
            label[r0, c0] = ns
            s_color[ns] = v
            top = 1
            stack[0] = r0 * n + c0
            while top > 0:
                top -= 1
                cur = stack[top]
                cr = cur // n
                cc = cur % n
                if cr > 0 and board[cr - 1, cc] == v and label[cr - 1, cc] == -1:
                    label[cr - 1, cc] = ns
                    stack[top] = (cr - 1) * n + cc
                    top += 1
                if cr < n - 1 and board[cr + 1, cc] == v and label[cr + 1, cc] == -1:
                    label[cr + 1, cc] = ns
                    stack[top] = (cr + 1) * n + cc
                    top += 1
                if cc > 0 and board[cr, cc - 1] == v and label[cr, cc - 1] == -1:
                    label[cr, cc - 1] = ns
                    stack[top] = cr * n + (cc - 1)
                    top += 1
                if cc < n - 1 and board[cr, cc + 1] == v and label[cr, cc + 1] == -1:
                    label[cr, cc + 1] = ns
                    stack[top] = cr * n + (cc + 1)
                    top += 1
            ns += 1
    if ns == 0:
        return

    # ---- 2. 区域：非 color、非洞的连通分量（边界条件自动成立）----
    reg = np.full((n, n), -1, dtype=np.int32)
    nr = 0
    for r0 in range(n):
        for c0 in range(n):
            v = board[r0, c0]
            if v == color or v == HOLE or reg[r0, c0] != -1:
                continue
            reg[r0, c0] = nr
            top = 1
            stack[0] = r0 * n + c0
            while top > 0:
                top -= 1
                cur = stack[top]
                cr = cur // n
                cc = cur % n
                if cr > 0 and board[cr - 1, cc] != color and board[cr - 1, cc] != HOLE \
                        and reg[cr - 1, cc] == -1:
                    reg[cr - 1, cc] = nr
                    stack[top] = (cr - 1) * n + cc
                    top += 1
                if cr < n - 1 and board[cr + 1, cc] != color and board[cr + 1, cc] != HOLE \
                        and reg[cr + 1, cc] == -1:
                    reg[cr + 1, cc] = nr
                    stack[top] = (cr + 1) * n + cc
                    top += 1
                if cc > 0 and board[cr, cc - 1] != color and board[cr, cc - 1] != HOLE \
                        and reg[cr, cc - 1] == -1:
                    reg[cr, cc - 1] = nr
                    stack[top] = cr * n + (cc - 1)
                    top += 1
                if cc < n - 1 and board[cr, cc + 1] != color and board[cr, cc + 1] != HOLE \
                        and reg[cr, cc + 1] == -1:
                    reg[cr, cc + 1] = nr
                    stack[top] = cr * n + (cc + 1)
                    top += 1
            nr += 1
    if nr == 0:
        return

    # ---- 3. 每个区域：相邻串数 + 关键串交集（条件③）----
    adj_cnt = np.zeros(nr, dtype=np.int32)
    vit_buf = np.full((nr, 4), -1, dtype=np.int32)
    vit_n = np.full(nr, -1, dtype=np.int32)     # -1 = 该区域还没有空点
    seen = np.zeros(ns, dtype=np.int32)
    tmp = np.zeros(4, dtype=np.int32)
    for r in range(n):
        for c in range(n):
            rid = reg[r, c]
            if rid < 0:
                continue
            ntmp = 0
            for k in range(4):
                if k == 0:
                    rr, cc = r - 1, c
                elif k == 1:
                    rr, cc = r + 1, c
                elif k == 2:
                    rr, cc = r, c - 1
                else:
                    rr, cc = r, c + 1
                if rr < 0 or rr >= n or cc < 0 or cc >= n:
                    continue
                if board[rr, cc] != color:
                    continue
                sid = label[rr, cc]
                if seen[sid] != rid + 1:
                    seen[sid] = rid + 1
                    adj_cnt[rid] += 1
                dup = False
                for t in range(ntmp):
                    if tmp[t] == sid:
                        dup = True
                        break
                if not dup:
                    tmp[ntmp] = sid
                    ntmp += 1
            if board[r, c] != 0:
                continue
            # 空点：与当前交集求交
            if vit_n[rid] < 0:
                vit_n[rid] = ntmp
                for t in range(ntmp):
                    vit_buf[rid, t] = tmp[t]
            else:
                keep = 0
                for t in range(vit_n[rid]):
                    s0 = vit_buf[rid, t]
                    hit = False
                    for m in range(ntmp):
                        if tmp[m] == s0:
                            hit = True
                            break
                    if hit:
                        vit_buf[rid, keep] = s0
                        keep += 1
                vit_n[rid] = keep

    # ---- 4. CSR：区域→相邻串、区域→关键串、串→相邻区域 ----
    adj_off = np.zeros(nr + 1, dtype=np.int32)
    vit_off = np.zeros(nr + 1, dtype=np.int32)
    for rid in range(nr):
        adj_off[rid + 1] = adj_off[rid] + adj_cnt[rid]
        vn = vit_n[rid]
        if vn < 0:
            vn = 0
        vit_off[rid + 1] = vit_off[rid] + vn
    adj_idx = np.zeros(adj_off[nr], dtype=np.int32)
    vit_idx = np.zeros(vit_off[nr], dtype=np.int32)
    pos = np.zeros(nr, dtype=np.int32)
    seen[:] = 0
    for r in range(n):
        for c in range(n):
            rid = reg[r, c]
            if rid < 0:
                continue
            for k in range(4):
                if k == 0:
                    rr, cc = r - 1, c
                elif k == 1:
                    rr, cc = r + 1, c
                elif k == 2:
                    rr, cc = r, c - 1
                else:
                    rr, cc = r, c + 1
                if rr < 0 or rr >= n or cc < 0 or cc >= n:
                    continue
                if board[rr, cc] != color:
                    continue
                sid = label[rr, cc]
                if seen[sid] != rid + 1:
                    seen[sid] = rid + 1
                    adj_idx[adj_off[rid] + pos[rid]] = sid
                    pos[rid] += 1
    for rid in range(nr):
        vn = vit_n[rid]
        if vn < 0:
            continue
        for t in range(vn):
            vit_idx[vit_off[rid] + t] = vit_buf[rid, t]

    r_cnt = np.zeros(ns, dtype=np.int32)
    for e in range(adj_off[nr]):
        r_cnt[adj_idx[e]] += 1
    r_off = np.zeros(ns + 1, dtype=np.int32)
    for s in range(ns):
        r_off[s + 1] = r_off[s] + r_cnt[s]
    r_idx = np.zeros(r_off[ns], dtype=np.int32)
    fpos = np.zeros(ns, dtype=np.int32)
    for rid in range(nr):
        for e in range(adj_off[rid], adj_off[rid + 1]):
            s = adj_idx[e]
            r_idx[r_off[s] + fpos[s]] = rid
            fpos[s] += 1

    # ---- 5. 单调删除到不动点 ----
    alive = np.zeros(ns, dtype=np.bool_)
    cnt = np.zeros(ns, dtype=np.int32)
    for e in range(vit_off[nr]):
        cnt[vit_idx[e]] += 1
    queue = np.zeros(ns, dtype=np.int32)
    qt = 0
    for s in range(ns):
        if s_color[s] == color:
            alive[s] = True
            if cnt[s] < 2:
                queue[qt] = s
                qt += 1
    reg_alive = np.ones(nr, dtype=np.bool_)
    qh = 0
    while qh < qt:
        s = queue[qh]
        qh += 1
        if not alive[s]:
            continue
        alive[s] = False
        for e in range(r_off[s], r_off[s + 1]):
            rid = r_idx[e]
            if not reg_alive[rid]:
                continue
            reg_alive[rid] = False
            for e2 in range(vit_off[rid], vit_off[rid + 1]):
                s2 = vit_idx[e2]
                if alive[s2]:
                    cnt[s2] -= 1
                    if cnt[s2] < 2:
                        queue[qt] = s2
                        qt += 1

    # ---- 6. 输出掩码 ----
    for r in range(n):
        for c in range(n):
            if board[r, c] == color and alive[label[r, c]]:
                out[r, c] = 1.0


@njit(cache=True)
def _benson_channels(board, n, player, own_out, opp_out):
    """己方/对方两个 Benson 无条件活通道（0/1）。"""
    _benson_color(board, n, player, own_out)
    _benson_color(board, n, -player, opp_out)


@njit(cache=True)
def _canonical_state(board, st, n, player, out, boundary):
    """当前玩家视角的**空间**输入（训练/自对弈/MCTS/GUI 全链路共用同一编码）。

    各通道下标见 hyperparams 的 CH_* 常量 —— 布局只在那边定义一处，这里只负责填值。
    通道2=上一手位置；贴目不再占用空间通道，改由 GoGame.get_global_features()
    生成独立的全局向量（当前 1 维，后续可扩成向量）。

    合法点掩码不在本函数内填 —— 它的来源（_legal_moves_and_mask）在热路径上
    每个节点本来就要算一次，由调用方复用、见 get_canonical_state。"""
    last_valid = st[1] >= 0
    # 边界通道由 GoGame 每局预生成：洞 -1、内外边界 +1、其余可落子点 0。
    out[CH_BOUNDARY] = boundary
    for r in range(n):
        for c in range(n):
            out[CH_OWN_STONES, r, c] = 1.0 if board[r, c] == player else 0.0
            out[CH_OPP_STONES, r, c] = 1.0 if board[r, c] == -player else 0.0
            out[CH_LAST_MOVE, r, c] = 1.0 if (last_valid and st[1] == r and st[2] == c) else 0.0
    # 危急度按棋块（同色连通）决定；征子通道单独算，避免把递归求解器塞进带缓存的函数。
    _urgency_channels(board, n, player,
                      out[CH_OWN_URGENCY], out[CH_OPP_URGENCY])
    # Benson 无条件活：两个 0/1 平面（形式化判定，不是启发式眼位）。
    _benson_channels(board, n, player, out[CH_OWN_ALIVE], out[CH_OPP_ALIVE])


@njit(cache=True)
def _dead_lump_mask(board, n, out):
    """终局清理：标出要移除的死子（out 为 True 的格子）。

    规则（迭代到不动点）：
      0. **只剩一口气**：直接清除，优先级高于下面所有规则。
      1. **孤子**：同色连通块 <= 5 子，且块内任何一颗子斜对角都没有同色棋。
         - 1-2 子：直接判死清除。
         - 3-5 子：紧邻敌方棋块里只要有气数 >= 3 的，判死清除；全 < 3 则保留（双活）。
      2. 其余一律保留。

    为什么第 1 条要迭代：一个子本身不孤立，是因为它斜对角还有同色接应；但那颗
    接应子可能只剩一口气、会被第 2 条清掉。清掉之后它就变成了孤子，必须重新判定。
    每一轮先按当前盘面把所有待清块**同时**标出（不是边标边改），再统一移除，
    最多 n*n 轮（每轮至少少一颗子）。

    ⚠️ 「小块」不等于「孤子」：尖、飞这类形状在斜线上还有接应，不能按孤子清。
    ⚠️ 孤子的死活只由「紧邻敌方棋块的气」决定，不再看自己的气：双活里两块各自
       只剩一两口气是常态，用「自己气少」判定会把双活整个清掉。
    """
    out[:, :] = False
    work = np.copy(board)
    labels = np.zeros((n, n), dtype=np.int32)     # 0=未访问/空，>0=组号
    lib_seen = np.zeros(n * n, dtype=np.int32)    # 印记法去重（存组号）
    lib = np.zeros(n * n + 1, dtype=np.int32)
    gsize = np.zeros(n * n + 1, dtype=np.int32)
    gdiag = np.zeros(n * n + 1, dtype=np.bool_)   # 组内是否有棋子在斜对角挨着同色
    elib = np.zeros(n * n + 1, dtype=np.int32)    # 紧邻敌块的最大气数
    eseen = np.zeros(n * n + 1, dtype=np.int32)   # 敌块去重（印记=当前组号）
    deadg = np.zeros(n * n + 1, dtype=np.bool_)
    stack = np.zeros(n * n, dtype=np.int32)

    for _pass in range(n * n):
        labels[:, :] = 0
        # ⚠️ lib_seen 必须每轮清零：组号每轮都从 1 重新分配，印记残留会让本轮
        # 的 gid 撞上上一轮同号的印记，把已有的气当成「已数过」→ 气数被少算，
        # 大块会被误判成「气 <= 1」整片清掉。
        lib_seen[:] = 0
        lib[:] = 0
        gsize[:] = 0
        gdiag[:] = False
        elib[:] = 0
        eseen[:] = 0
        deadg[:] = False
        next_gid = 1
        # ---------- 1) 分组：大小、气、斜对角接应 ----------
        for r in range(n):
            for c in range(n):
                color = work[r, c]
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
                    if not gdiag[gid]:
                        if (cr > 0 and cc > 0 and work[cr - 1, cc - 1] == color) or \
                           (cr > 0 and cc < n - 1 and work[cr - 1, cc + 1] == color) or \
                           (cr < n - 1 and cc > 0 and work[cr + 1, cc - 1] == color) or \
                           (cr < n - 1 and cc < n - 1 and work[cr + 1, cc + 1] == color):
                            gdiag[gid] = True
                    if cr > 0:
                        nb = work[cr - 1, cc]
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
                        nb = work[cr + 1, cc]
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
                        nb = work[cr, cc - 1]
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
                        nb = work[cr, cc + 1]
                        if nb == 0:
                            pos = cr * n + (cc + 1)
                            if lib_seen[pos] != gid:
                                lib_seen[pos] = gid
                                lib[gid] += 1
                        elif nb == color and labels[cr, cc + 1] == 0:
                            labels[cr, cc + 1] = gid
                            stack[top] = cr * n + (cc + 1)
                            top += 1
        # ---------- 2) 每个块紧邻敌块的最大气数 ----------
        for r in range(n):
            for c in range(n):
                gid = labels[r, c]
                if gid == 0:
                    continue
                for d in range(4):
                    nr = r + (-1 if d == 0 else (1 if d == 1 else 0))
                    nc = c + (-1 if d == 2 else (1 if d == 3 else 0))
                    if nr < 0 or nr >= n or nc < 0 or nc >= n:
                        continue
                    eg = labels[nr, nc]
                    if eg == 0 or eg == gid:
                        continue
                    if eseen[eg] != gid:
                        eseen[eg] = gid
                        if lib[eg] > elib[gid]:
                            elib[gid] = lib[eg]
        # ---------- 3) 判定（先全部判定，再统一移除） ----------
        for r in range(n):
            for c in range(n):
                gid = labels[r, c]
                if gid == 0 or deadg[gid]:
                    continue
                if lib[gid] <= 1:
                    deadg[gid] = True              # 只剩一气：优先级最高，直接清
                elif not gdiag[gid] and gsize[gid] <= 2:
                    deadg[gid] = True              # 1-2 子孤子：直接清
                elif not gdiag[gid] and gsize[gid] <= 5:
                    deadg[gid] = elib[gid] >= 3    # 3-5 子孤子：敌块有 3 气才清，否则双活
                else:
                    deadg[gid] = False
        removed = 0
        for r in range(n):
            for c in range(n):
                gid = labels[r, c]
                if gid > 0 and deadg[gid]:
                    out[r, c] = True
                    work[r, c] = 0
                    removed += 1
        if removed == 0:
            break


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
    """numba 加速版围棋（API 与 Python 版完全一致）

    形状只有一个来源：`self.hole_mask`（(n,n) bool，True = 不能落子）。
    居中方块洞只是 `hollow` 这个便捷写法生成的一种掩码；心形、圆环、
    多个洞直接传 `hole_mask` 即可，规则层不做任何区分。
    """

    def __init__(self, board_size=BOARD_SIZE, komi=KOMI, hole_mask=None,
                 start_board=None):
        self.board_size = int(board_size)
        # 贴目只要求 0.5 粒度（整数合法）：终局目差 = 整数 - 贴目。
        self.komi = validate_komi(komi)
        n = self.board_size
        self.hole_mask = shapes.validate_mask(
            shapes.solid(n) if hole_mask is None else hole_mask, n)
        # 自定义开局：直接摆好黑白子，永远黑先；空盘(None)就是普通开局
        self._start_stones = shapes.validate_stones(start_board, self.hole_mask, n)
        self._boundary = boundary_plane_from_mask(self.hole_mask)
        self._simulate_mode = False
        self.reset()

    def reset(self):
        n = self.board_size
        rules = board_rules_for_area(shapes.playable_area(self.hole_mask))
        self.board = np.zeros((n, n), dtype=np.int8)
        self.board[self.hole_mask] = HOLE
        # [0]当前方 [1]/[2]上一手 [3]手数 [4]连续pass [5]winner(-2=未定)
        # [6]/[7] min_moves/max_moves
        # [8:10] 当前局面 Zobrist 哈希，[10] 全历史局面条数（位置超级劫）
        # [11]/[12] 黑白子数 [13]/[14] 黑白累计提子（增量维护，全局特征直接读）
        self._st = np.zeros(15, dtype=np.int32)
        self._st[0] = 1                    # 当前玩家
        self._st[1] = -2                   # 上一步 r（-2=None）
        self._st[2] = -2
        self._st[5] = -2                   # winner（-2=None）
        self._st[6] = rules["min_moves"]   # 前 N 手不允许 pass
        self._st[7] = rules["max_moves"]   # 单局最大手数
        self._st[10] = 1                   # 历史里已有空盘
        self._hist = np.zeros(rules["max_moves"] + 2, dtype=np.uint64)
        self._go = np.zeros(1, dtype=np.bool_)
        self._fp = np.zeros(1, dtype=np.float64)
        self._cap = np.zeros(self.board_size * self.board_size, dtype=np.int32)
        if getattr(self, "_start_stones", None) is not None:
            self._apply_start(self._start_stones)

    def _apply_start(self, stones):
        """把自定义开局的棋子摆上盘，并重建哈希/历史。

        与 setup_handicap 不同：这里黑白都有，且无论双方子数多少，
        **永远黑先**（用户约定：开局摆子只是局面，不是让子）。
        """
        n = self.board_size
        st = np.asarray(stones, dtype=np.int8)
        self.board[st == 1] = 1
        self.board[st == -1] = -1
        self._check_start_legal(st)
        self._st[11] = int(np.count_nonzero(self.board == 1))
        self._st[12] = int(np.count_nonzero(self.board == -1))
        self._st[0] = 1                    # 黑先
        self._st[1] = self._st[2] = -2
        self._st[3] = 0
        self._st[4] = 0
        h = _board_hash(self.board, n)
        _set_hash_slots(self._st, h)
        self._hist[0] = h
        self._st[10] = 1

    def _check_start_legal(self, stones):
        """开局局面合法性：不允许有「一口气都没有」的棋块。"""
        n = self.board_size
        b = self.board
        seen = np.zeros((n, n), dtype=bool)
        for r in range(n):
            for c in range(n):
                if b[r, c] == 0 or b[r, c] == HOLE or seen[r, c]:
                    continue
                color = b[r, c]
                cells = []
                stack = [(r, c)]
                seen[r, c] = True
                libs = 0
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if not (0 <= nr < n and 0 <= nc < n):
                            continue
                        v = b[nr, nc]
                        if v == 0:
                            libs += 1
                        elif v == color and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                if libs == 0:
                    raise ValueError(
                        "开局不合法：(%d,%d) 所在的%s棋块没有气"
                        % (r, c, "黑" if color == 1 else "白"))

    def setup_handicap(self, points):
        """让子：把黑子直接摆上盘，白先走（不走 make_move 的黑白轮转）。

        让子只进回放、不生成训练样本；贴 0 目由调用方（selfplay）在建局时决定。
        摆完必须重算盘面哈希并重建历史 —— 否则超级劫会拿「空盘」的历史去比对。
        返回实际摆上的子数。
        """
        n = self.board_size
        placed = 0
        last = None
        for r, c in points:
            r, c = int(r), int(c)
            if not (0 <= r < n and 0 <= c < n):
                continue
            if self.board[r, c] != 0:
                continue
            self.board[r, c] = 1           # 让子永远给黑
            self._st[11] += 1
            self._st[3] += 1
            placed += 1
            last = (r, c)
        if not placed:
            return 0
        h = _board_hash(self.board, n)
        _set_hash_slots(self._st, h)
        self._hist[0] = h
        self._st[10] = 1
        self._st[0] = -1                   # 白先走
        self._st[1], self._st[2] = last
        return placed

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
    def move_count(self):
        return int(self._st[3])

    @move_count.setter
    def move_count(self, v):
        self._st[3] = int(v)

    @property
    def pass_count(self):
        return int(self._st[4])

    @pass_count.setter
    def pass_count(self, v):
        self._st[4] = int(v)

    @property
    def game_over(self):
        return bool(self._go[0])

    @game_over.setter
    def game_over(self, v):
        self._go[0] = bool(v)

    @property
    def winner(self):
        return None if self._st[5] == -2 else int(self._st[5])

    @winner.setter
    def winner(self, v):
        self._st[5] = -2 if v is None else int(v)

    @property
    def final_points(self):
        return float(self._fp[0])

    @final_points.setter
    def final_points(self, v):
        self._fp[0] = float(v)

    # ---- 热路径（numba）----
    def is_legal_move(self, row, col):
        if self._go[0]:
            return False
        if (row, col) == PASS_MOVE:
            # pass 只受 min_moves 限制；没有 pass 上限，落子永远不被 pass 计数阻塞
            return self._st[3] >= self._st[6]
        n = self.board_size
        if not (0 <= row < n and 0 <= col < n):
            return False
        cur_h = _hash_from_slots(self._st)
        hist_n = _sync_history(self._st, self._hist, cur_h)
        return _is_legal_impl(self.board, row, col, self._st[0],
                              cur_h, self.board_size, self._hist, hist_n)

    def make_move(self, row, col, check_legal=False):
        if check_legal and not self.is_legal_move(row, col):
            return False
        if (row, col) != PASS_MOVE:
            n = self.board_size
            # 越界坐标会被 numpy 负索引回绕成"另一端的点"，重叠落子会直接覆盖
            # 已有棋子。两者都会把盘面和 Zobrist 哈希一起写坏，而 _make_move
            # 仍然返回 True。挡在 numba 入口前，成本是一次比较。
            if not (0 <= row < n and 0 <= col < n):
                return False
            if self.board[row, col] != 0:
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
        own_ladder = np.zeros((n, n), dtype=np.float32)
        opp_ladder = np.zeros((n, n), dtype=np.float32)
        _canonical_state(self.board, self._st, n, int(player), out,
                         self._boundary)
        _ladder_channels(self.board, n, int(player), own_ladder, opp_ladder)
        out[CH_OWN_LADDER] = own_ladder
        out[CH_OPP_LADDER] = opp_ladder
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

    def get_global_features(self, player=None):
        """当前玩家视角的全局特征向量（与空间输入分离）。

        数量类特征统一归一化到 [-1,1]：``2*count/可落子面积 - 1``。
        棋子数与提子数是增量维护的（见 _make_move），这里只做读取和缩放，
        不扫棋盘 —— 这函数在 MCTS 每个叶子上都要跑。
        """
        if player is None:
            player = self._st[0]
        p = int(player)
        area = float(self.playable_area)
        st = self._st
        own = float(st[11] if p == 1 else st[12])
        opp = float(st[12] if p == 1 else st[11])
        own_cap = float(st[13] if p == 1 else st[14])
        opp_cap = float(st[14] if p == 1 else st[13])
        out = np.zeros((GLOBAL_DIM,), dtype=np.float32)
        out[G_KOMI] = np.float32(p * self.komi / KOMI_SCALE)
        out[G_OWN_STONES] = np.float32(2.0 * own / area - 1.0)
        out[G_OPP_STONES] = np.float32(2.0 * opp / area - 1.0)
        out[G_OWN_CAPS] = np.float32(min(1.0, max(-1.0, 10.0 * own_cap / area - 1.0)))
        out[G_OPP_CAPS] = np.float32(min(1.0, max(-1.0, 10.0 * opp_cap / area - 1.0)))
        out[G_AREA] = np.float32(area / 361.0 - 1.0)
        return out

    def get_state_bundle(self, player=None, legal_mask=None):
        """一次生成 (空间状态, 全局向量)，保证两者使用同一个 player。"""
        if player is None:
            player = self._st[0]
        state = self.get_canonical_state(player=player, legal_mask=legal_mask)
        return state, self.get_global_features(player=player)


    def is_hole(self, row, col):
        n = self.board_size
        return (0 <= row < n and 0 <= col < n
                and self.board[row, col] == HOLE)

    @property
    def valid_mask(self):
        """可落子点掩码（洞为 False；内外边界都算可落子）。"""
        return self._boundary >= 0.0

    @property
    def playable_area(self):
        return int((~self.hole_mask).sum())

    def copy(self):
        g = GoGame.__new__(GoGame)
        g.board_size = self.board_size
        g.komi = self.komi
        # 形状与边界平面在整局里都不变：共享引用而不是逐次深拷贝。
        # MCTS 每次模拟都会 copy 一次，省掉两份 n² 数组。
        g.hole_mask = self.hole_mask
        g._boundary = self._boundary
        g.board = self.board.copy()
        g._st = self._st.copy()
        g._hist = self._hist.copy()
        g._go = self._go.copy()
        g._fp = self._fp.copy()
        g._cap = np.zeros(self.board_size * self.board_size, dtype=np.int32)
        g._simulate_mode = False
        # 开局棋子不随对局变化，共享引用即可；漏掉它会让「悔棋后重来」炸在 reset()
        g._start_stones = self._start_stones
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
        self._st[5] = 1 if data['final_diff'] > 0 else (-1 if data['final_diff'] < 0 else 0)
        return data['own']

    def _merged_analysis(self):
        """终局领地/归属分析：所有棋子算活，只对空区域做 flood fill。

        只挨黑=黑地、只挨白=白地、两种都挨=中立（归属 0，目数黑白各 0.5，
        目差中相互抵消）。数目之前按 `_dead_lump_mask` 的规则移除死子
        （孤子 + 紧邻敌块气数 >= 3；或独立气数 <= 1），避免终局死子污染标签。
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
