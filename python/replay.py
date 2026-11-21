"""紧凑、磁盘友好的围棋训练数据。

每局一个 .npz，保存的是重建空间输入所需的**最小信息**：
  board/player/last/legal + 2 个 uint8 危急度平面；
  边界与征子通道运行时按形状掩码生成，不落盘（只存 67 字节的位图）。
  贴目按局存储，训练时单独变成全局向量（不占空间通道）。
策略只存非零项（top-p 后通常十几个点），不存 362 维 float16 稠密向量。
归属标签每局只存一份，样本只引用它；own_mask 标记这批样本是否参与归属损失。
贴目按局存储：模型输入通道 3 用该局真实贴目，不再强制 7.5。

旧 pkl / 旧 6 通道 / 旧 7 通道数据不兼容，也不尝试兼容。
"""
import os
import pickle
import random
import threading
import time
from datetime import datetime

import numpy as np

import shapes
from hyperparams import (BOARD_SIZE, KOMI, PASS_MOVE, INPUT_CHANNELS,
                         GLOBAL_DIM, G_KOMI, KOMI_SCALE,
                         G_OWN_STONES, G_OPP_STONES, G_OWN_CAPS, G_OPP_CAPS, G_AREA,
                         CH_OWN_STONES, CH_OPP_STONES, CH_LAST_MOVE,
                         CH_OWN_URGENCY, CH_OPP_URGENCY, CH_LEGAL_MASK,
                         CH_BOUNDARY,
                         CH_OWN_LADDER, CH_OPP_LADDER,
                         CH_OWN_REGION, CH_OPP_REGION, CH_NEUTRAL_REGION,
                         CH_OWN_CHAIN_SIZE, CH_OPP_CHAIN_SIZE,
                         CH_OWN_EYE_CNT, CH_OPP_EYE_CNT,
                         CH_OWN_EYE_AREA, CH_OPP_EYE_AREA,
                         is_half_step_komi, HOLE)
from game import (boundary_plane_from_mask, _ladder_channels, _region_channels,
                  _chain_region_channels)


DATA_VERSION = 9              # v9: 删掉眼通道平面（feats 由 4 排变 2 排）；旧数据用 tools/migrate_drop_eyes.py 升级
SUPPORTED_DATA_VERSIONS = (9,)
FEATURE_CHANNELS = 2

# 内存缓冲的**方向约定**：旧 → 新（头部最旧、尾部最新）。
# add_game 只从头部淘汰，所以头部必须是最旧的；早期缓存写的是「新 → 旧」，
# 混用会让新对局把刚加载的最新历史挤掉，故把版本写进缓存文件名，旧缓存直接作废。
CACHE_ORDER = 2


class Sample:
    """单条训练样本的紧凑内存表示。

    source/index 记录这条样本来自哪个 .npz、第几条（排查数据问题时用）；
    新对局样本没有磁盘来源，保持 None/-1。
    """
    __slots__ = ("board", "player", "last", "legal", "feat", "pidx", "pp", "own", "winner",
                 "komi", "own_mask", "source", "index", "hole_bits",
                 "cap_own", "cap_opp", "score_diff", "resigned", "own_known")

    def __init__(self, board, player, last, legal, feat, pidx, pp, own, winner,
                 komi=KOMI, own_mask=1, source=None, index=-1, hole_bits=None,
                 cap_own=0, cap_opp=0, score_diff=0.0, resigned=False,
                 own_known=None):
        self.board = board
        self.player = int(player)
        self.last = int(last)
        self.legal = legal
        self.feat = feat
        self.pidx = pidx
        self.pp = pp
        self.own = own
        self.winner = int(winner)
        self.komi = float(komi)
        self.own_mask = int(own_mask)
        self.source = source
        self.index = int(index)
        self.hole_bits = hole_bits       # 形状位图（uint8）；空数组 = 实心盘
        # 这一局面之前的累计提子（行棋方视角（己方/对方））
        self.cap_own = int(cap_own)
        self.cap_opp = int(cap_opp)
        # 黑视角终局目差（含贴目）；截断/无终局分为 0
        self.score_diff = float(score_diff)
        # 认输局：胜负有效，但没有终局目差；归属只标了确定的部分
        self.resigned = bool(resigned)
        # 归属标签可信点（(n,n) uint8）；None = 整盘可信
        self.own_known = None if own_known is None else np.asarray(
            own_known, dtype=np.uint8)

    def hole_mask(self, n):
        """本局形状掩码。位图为空表示整盘可下。"""
        if self.hole_bits is None or not len(self.hole_bits):
            return shapes.solid(int(n))
        return shapes.unpack_bits(self.hole_bits, int(n))


def state_to_sample(state, policy, player, last_move,
                    cap_own=0, cap_opp=0):
    """把空间规范状态 + 稠密策略转成紧凑样本（不保存通道原图；边界/征子按尺寸/镂空重建）。

    cap_own/cap_opp 是这一局面之前的累计提子数（行棋方视角（己方/对方））。
    合法点（含超级劫禁手）直接从状态里的合法点通道取，不单独存劫点。
    """
    state = np.asarray(state, dtype=np.float32)
    n = state.shape[-1]
    total = n * n
    board = ((state[CH_OWN_STONES] > 0.5).astype(np.int8)
             - (state[CH_OPP_STONES] > 0.5).astype(np.int8)).reshape(total)
    feat = np.empty((FEATURE_CHANNELS, total), dtype=np.uint8)
    feat[0] = np.rint(state[CH_OWN_URGENCY].reshape(total) * 3.0).astype(np.uint8)
    feat[1] = np.rint(state[CH_OPP_URGENCY].reshape(total) * 3.0).astype(np.uint8)
    legal = np.packbits(state[CH_LEGAL_MASK].reshape(total) > 0.5)
    if last_move is None or last_move == PASS_MOVE or last_move[0] < 0:
        last = -1
    else:
        last = int(last_move[0]) * n + int(last_move[1])
    policy = np.asarray(policy, dtype=np.float32).reshape(-1)
    pidx = np.flatnonzero(policy > 0).astype(np.uint16)
    pp = policy[pidx].astype(np.float16)
    if pidx.size == 0:
        legal_idx = np.flatnonzero(state[CH_LEGAL_MASK].reshape(total) > 0.5)
        pidx = legal_idx.astype(np.uint16)
        pp = np.full(pidx.size, 1.0 / max(pidx.size, 1), dtype=np.float16)
    return {
        "board": board,
        "player": int(player),
        "last": last,
        "legal": legal,
        "feat": feat,
        "pidx": pidx,
        "pp": pp,
        # 累计提子（行棋方视角（己方/对方））。棋子数可以直接从 board 数出来，不用存。
        "cap_own": int(cap_own),
        "cap_opp": int(cap_opp),
    }


def pack_game(samples, moves, winner, score_diff, ownership, board_size=BOARD_SIZE,
              komi=KOMI, own_mask=1, hole=None, start_stones=None, resigned=False,
              own_known=None):
    """把一局紧凑样本打包成 savez 用的 dict。

    hole 是棋盘形状掩码（(n,n) bool，True=洞）；None/全 False 表示实心盘。
    own_mask=0 表示这局只训练 policy/win，不训练 ownership。
    resigned=True 表示这局以认输结束：胜负有效，但没有终局目差。
    own_known 是 (n,n) bool，True=该点归属标签可信；None=整盘可信。
    认输局用 Benson 无条件活只标确定的部分，其余点掩掉（不进归属损失）。
    """
    n = int(board_size)
    if hole is None:
        hole_bits = np.zeros(0, dtype=np.uint8)
    else:
        hole = shapes.validate_mask(hole, n)
        hole_bits = shapes.pack_bits(hole) if hole.any() else np.zeros(0, dtype=np.uint8)
    if start_stones is None:
        start = np.zeros((n, n), dtype=np.int8)
    else:
        start = np.asarray(start_stones, dtype=np.int8).reshape(n, n)
    t = len(samples)
    if t == 0:
        raise ValueError("空对局不能保存")
    boards = np.stack([s["board"] for s in samples]).astype(np.int8)
    players = np.array([s["player"] for s in samples], dtype=np.int8)
    lasts = np.array([s["last"] for s in samples], dtype=np.int16)
    legals = np.stack([s["legal"] for s in samples]).astype(np.uint8)
    feats = np.stack([s["feat"] for s in samples]).astype(np.uint8)
    lens = np.array([len(s["pidx"]) for s in samples], dtype=np.uint16)
    pidx = np.concatenate([s["pidx"] for s in samples]).astype(np.uint16)
    pp = np.concatenate([s["pp"] for s in samples]).astype(np.float16)
    move_codes = np.zeros(t, dtype=np.int16)
    recent = moves[-t:] if len(moves) >= t else moves
    for i, mv in enumerate(recent):
        r, c = int(mv[0]), int(mv[1])
        move_codes[i] = 0 if (r < 0 or c < 0) else (r + 1) * (n + 1) + (c + 1)
    return {
        "boards": boards,
        "players": players,
        "lasts": lasts,
        "legals": legals,
        "feats": feats,
        "pol_len": lens,
        "pol_idx": pidx,
        "pol_p": pp,
        "moves": move_codes,
        "winner": np.int8(winner),
        "score_diff": np.float32(score_diff),
        "cap_own": np.array([s.get("cap_own", 0) for s in samples], dtype=np.int16),
        "cap_opp": np.array([s.get("cap_opp", 0) for s in samples], dtype=np.int16),
        "ownership": np.asarray(ownership, dtype=np.int8).reshape(n, n),
        "komi": np.float32(komi),
        "hole_bits": hole_bits,
        "own_mask": np.uint8(own_mask),
        "resigned": np.uint8(1 if resigned else 0),
        "own_known": (np.ones((n, n), dtype=np.uint8) if own_known is None
                      else np.asarray(own_known, dtype=np.uint8).reshape(n, n)),
        "start_stones": start,
    }


def payload_arrays(payload, board_size=BOARD_SIZE):
    """把 pack_game 的 dict 转成 npz 的键值；贴目/归属掩码按 payload 落盘。"""
    n = int(board_size)
    return {
        "version": np.int16(DATA_VERSION),
        "board_size": np.int16(n),
        "komi": np.float32(payload.get("komi", KOMI)),
        "hole_bits": np.asarray(payload.get("hole_bits", np.zeros(0, dtype=np.uint8)),
                                dtype=np.uint8),
        "own_mask": np.uint8(payload.get("own_mask", 1)),
        "boards": payload["boards"],
        "players": payload["players"],
        "lasts": payload["lasts"],
        "legals": payload["legals"],
        "feats": payload["feats"],
        "pol_len": payload["pol_len"],
        "pol_idx": payload["pol_idx"],
        "pol_p": payload["pol_p"],
        "moves": payload["moves"],
        "winner": payload["winner"],
        "score_diff": payload["score_diff"],
        "ownership": payload["ownership"],
        "cap_own": np.asarray(payload["cap_own"], dtype=np.int16),
        "cap_opp": np.asarray(payload["cap_opp"], dtype=np.int16),
        "resigned": np.uint8(payload.get("resigned", 0)),
        "own_known": np.asarray(payload.get("own_known",
                                           np.ones((n, n), dtype=np.uint8)),
                                dtype=np.uint8).reshape(n, n),
        "start_stones": np.asarray(payload.get("start_stones",
                                                np.zeros((n, n), dtype=np.int8)),
                                   dtype=np.int8).reshape(n, n),
    }


def load_npz_arrays(path):
    """把一局 .npz 的所有键读成普通 ndarray（不保留 NpzFile 句柄）。"""
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k].copy() for k in d.files}


class TrainingData:
    """内存缓冲 + 每局一个 npz。旧 pkl 不读取、不兼容。"""

    def __init__(self, max_size=200000, data_dir="data/", board_size=BOARD_SIZE,
                 encoder=None):
        self.max_size = int(max_size)
        self.data_dir = data_dir
        self.board_size = int(board_size)
        # 其他棋种（如五子棋）把自己的批量编码器传进来；None = 用下面的 go 编码。
        self.encoder = encoder
        self.data = []
        self.last_skipped = 0
        self._shape_cache = {}
        os.makedirs(data_dir, exist_ok=True)

    def _shape(self, n, bits):
        """(掩码, 边界平面)，按形状缓存 —— 一局几千条样本只算一次。"""
        key = (int(n), bits.tobytes() if bits is not None and len(bits) else b"")
        hit = self._shape_cache.get(key)
        if hit is None:
            mask = (shapes.solid(n) if len(key[1]) == 0
                    else shapes.unpack_bits(bits, n))
            hit = (mask, boundary_plane_from_mask(mask))
            self._shape_cache[key] = hit
        return hit

    def __len__(self):
        return len(self.data)

    def add_game(self, payload, source=None):
        """payload 来自 selfplay.play_one_game 的 compact 字段。

        source 是该局落盘的 .npz 路径（不保存数据时为 None）；
        和 index 一起用于界面手动改标签时同步内存缓冲。
        """
        samples = payload["samples"]
        own = payload["ownership"]
        winner = int(payload["winner"])
        komi = float(payload.get("komi", KOMI))
        hole = payload.get("hole")
        if hole is not None:
            hole = shapes.validate_mask(hole, self.board_size)
            hole_bits = shapes.pack_bits(hole) if hole.any() else None
        else:
            hole_bits = None
        own_mask = payload.get("own_mask")
        resigned = bool(payload.get("resigned", False))
        own_known = payload.get("own_known")
        packed = payload.get("payload")
        if own_mask is None:
            own_mask = (packed or {}).get("own_mask", 1)
        if own_known is None:
            # play_one_game 的返回值既有顶层字段也有内层打包 dict，两边都认
            own_known = (packed or {}).get("own_known")
        own_mask = int(own_mask)
        if own_known is not None:
            own_known = np.asarray(own_known, dtype=np.uint8).reshape(
                self.board_size, self.board_size)
        take = samples[-self.max_size:]
        start = len(samples) - len(take)
        if source is not None:
            source = os.path.normcase(os.path.abspath(source))
        drop = len(self.data) + len(take) - self.max_size
        if drop > 0:
            # 头部是最旧的（CACHE_ORDER 约定见文件顶部）；random 模式被打乱过，
            # 这里的「头部」就是随机样本 —— 即随机淘汰。
            del self.data[:drop]
        for j, s in enumerate(take):
            self.data.append(Sample(
                s["board"], s["player"], s["last"],
                s["legal"], s["feat"], s["pidx"], s["pp"], own, winner,
                komi=komi, own_mask=own_mask,
                source=source, index=start + j, hole_bits=hole_bits,
                cap_own=s.get("cap_own", 0), cap_opp=s.get("cap_opp", 0),
                score_diff=float(payload.get("score_diff", 0.0)),
                resigned=resigned, own_known=own_known))

    def sample(self, batch_size):
        """训练输入（兼容旧调用方）；需要终局目差时用 sample_with_scores。"""
        return self.sample_with_scores(batch_size)[:6]

    def sample_with_scores(self, batch_size):
        batch = random.sample(self.data, min(int(batch_size), len(self.data)))
        b = len(batch)
        n = self.board_size
        total = n * n
        if self.encoder is not None:
            # 其他棋种：编码器自己决定通道数和全局向量（go 的 9 通道见 _fill_go_states）。
            states, globals_ = self.encoder(batch, n)
            states = np.asarray(states, dtype=np.float32)
            globals_ = np.asarray(globals_, dtype=np.float32)
        else:
            states = np.zeros((b, INPUT_CHANNELS, n, n), dtype=np.float32)
            globals_ = np.zeros((b, GLOBAL_DIM), dtype=np.float32)
            self._fill_go_states(batch, states, globals_)
        policies = np.zeros((b, total + 1), dtype=np.float32)
        values = np.zeros((b,), dtype=np.float32)
        own = np.zeros((b, n, n), dtype=np.int8)
        own_masks = np.zeros((b,), dtype=np.uint8)
        # 行棋方视角的**实际**终局目差（含贴目换算：p*(score_diff+komi) 等于 p*(黑-白)），
        # 给「宏观领地损失」用；截断/无终局分的对局为 0。
        scores = np.zeros((b,), dtype=np.float32)
        # 归属标签可信点（逐点掩码）；认输局只标了 Benson 确定的部分
        own_knowns = np.ones((b, n, n), dtype=np.uint8)
        # 有真实终局目差的对局（宏观领地项只在这些局上算）：截断局、认输局都不算
        macro_masks = np.zeros((b,), dtype=np.uint8)
        for j, s in enumerate(batch):
            p = s.player
            pidx, pp = s.pidx, s.pp
            own_masks[j] = s.own_mask
            policies[j, pidx] = pp.astype(np.float32)
            if s.winner != 0:
                values[j] = 1.0 if s.winner == p else -1.0
                scores[j] = p * (s.score_diff + s.komi)
                macro_masks[j] = 0 if s.resigned else 1
            own[j] = s.own * p
            if s.own_known is not None:
                own_knowns[j] = s.own_known
        return (states, globals_, policies, values, own, own_masks, scores,
                own_knowns, macro_masks)

    def _fill_go_states(self, batch, states, globals_):
        """围棋的 9 通道编码 + 6 维全局向量（改动前 sample() 里的内联逻辑）。"""
        n = self.board_size
        total = n * n
        urg3 = 1.0 / 3.0
        for j, s in enumerate(batch):
            board = s.board.reshape(n, n)
            p = s.player
            mask, bd = self._shape(n, s.hole_bits)
            states[j, CH_OWN_STONES] = board > 0
            states[j, CH_OPP_STONES] = board < 0
            if s.last >= 0:
                states[j, CH_LAST_MOVE, s.last // n, s.last % n] = 1.0
            globals_[j, G_KOMI] = p * s.komi / KOMI_SCALE
            # 棋子数直接从已填的通道数出来（批量场景下这是 numpy 归约，不是扫棋盘）；
            # 提子数在数据里存着，按行棋方取己方/对方。
            area = float(total - int(mask.sum()))
            own_n = float(np.count_nonzero(board > 0))
            opp_n = float(np.count_nonzero(board < 0))
            own_cap = float(s.cap_own)
            opp_cap = float(s.cap_opp)
            globals_[j, G_OWN_STONES] = 2.0 * own_n / area - 1.0
            globals_[j, G_OPP_STONES] = 2.0 * opp_n / area - 1.0
            globals_[j, G_OWN_CAPS] = min(1.0, max(-1.0, 10.0 * own_cap / area - 1.0))
            globals_[j, G_OPP_CAPS] = min(1.0, max(-1.0, 10.0 * opp_cap / area - 1.0))
            globals_[j, G_AREA] = area / 361.0 - 1.0
            states[j, CH_OWN_URGENCY] = s.feat[0].reshape(n, n) * urg3
            states[j, CH_OPP_URGENCY] = s.feat[1].reshape(n, n) * urg3
            states[j, CH_LEGAL_MASK] = np.unpackbits(s.legal, count=total).reshape(n, n)
            states[j, CH_BOUNDARY] = bd
            # 征子通道：数据里的 board 是「当前行棋方视角」（+1=己方），
            # 求解器要绝对黑白，所以先乘 p 还原；洞在棋盘数组里是 0，再补 HOLE 哨兵。
            b_solver = (board.astype(np.int16) * p).astype(np.int8)
            b_solver[mask] = HOLE
            own_lad = np.zeros((n, n), dtype=np.float32)
            opp_lad = np.zeros((n, n), dtype=np.float32)
            _ladder_channels(b_solver, n, p, own_lad, opp_lad)
            states[j, CH_OWN_LADDER] = own_lad
            states[j, CH_OPP_LADDER] = opp_lad
            # 空点区域归属：同样用绝对黑白 + HOLE 墙。
            own_reg = np.zeros((n, n), dtype=np.float32)
            opp_reg = np.zeros((n, n), dtype=np.float32)
            neu_reg = np.zeros((n, n), dtype=np.float32)
            _region_channels(b_solver, n, p, own_reg, opp_reg, neu_reg)
            states[j, CH_OWN_REGION] = own_reg
            states[j, CH_OPP_REGION] = opp_reg
            states[j, CH_NEUTRAL_REGION] = neu_reg
            # 链级特征：子数（正向）+ 相邻单色区域个数/面积（危险度，越大越危险）
            sz_o = np.zeros((n, n), dtype=np.float32)
            sz_p = np.zeros((n, n), dtype=np.float32)
            cn_o = np.zeros((n, n), dtype=np.float32)
            cn_p = np.zeros((n, n), dtype=np.float32)
            ar_o = np.zeros((n, n), dtype=np.float32)
            ar_p = np.zeros((n, n), dtype=np.float32)
            _chain_region_channels(b_solver, n, p,
                                   sz_o, sz_p, cn_o, cn_p, ar_o, ar_p)
            states[j, CH_OWN_CHAIN_SIZE] = sz_o
            states[j, CH_OPP_CHAIN_SIZE] = sz_p
            states[j, CH_OWN_EYE_CNT] = cn_o
            states[j, CH_OPP_EYE_CNT] = cn_p
            states[j, CH_OWN_EYE_AREA] = ar_o
            states[j, CH_OPP_EYE_AREA] = ar_p

    # ---------------- 落盘 / 读取 ----------------
    def save_game(self, payload, data_dir=None, prefix="data"):
        target = self.data_dir if data_dir is None else data_dir
        os.makedirs(target, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = f"{prefix}_{ts}"
        path = os.path.join(target, base + ".npz")
        suffix = 1
        while os.path.exists(path):
            path = os.path.join(target, f"{base}_{suffix}.npz")
            suffix += 1
        np.savez_compressed(path, **payload_arrays(payload, self.board_size))
        n_saved = len(payload["boards"])
        print(f"[数据] 已保存 {n_saved} 条到 {path}")
        return path

    def load_all(self, mode="newest"):
        """把 data_dir 下所有 .npz 读进内存缓冲。

        慢在「每局一个 npz」：7000 个文件、每个要解压 ~20 个 zip 成员，
        一次冷启动要 20 秒以上（纯 CPU，多线程反而更慢 —— zipfile 解析吃 GIL）。
        所以这里加了一层**整目录缓存**（同目录 `_cache_v<数据版本>_<尺寸>.pkl`）：
        启动时直接 pickle 读回上一次的样本，只解析上次之后新增的 npz。
        文件被删掉时按 source 过滤；缓存损坏/版本不符就整体重建。
        """
        t0 = time.time()
        entries = []
        with os.scandir(self.data_dir) as it:
            for e in it:
                if not e.name.endswith(".npz"):
                    continue
                try:
                    st = e.stat()
                except OSError:
                    continue
                entries.append((e.path, st.st_size, int(st.st_mtime_ns)))
        if not entries:
            return False
        # 新的在前：缓存里的是上一次的，新文件一定更新
        entries.sort(key=lambda x: x[2], reverse=True)
        sig = {os.path.basename(p): (sz, mt) for p, sz, mt in entries}
        cached, old_sig = self._load_cache()
        new_files = [p for p, sz, mt in entries
                     if old_sig.get(os.path.basename(p)) != (sz, mt)]
        # 先读新文件（都是最近写的，排在最前），再拼上缓存
        self.data.clear()
        loaded = 0
        skipped = 0
        for f in new_files:
            try:
                loaded += self._load_file(f)
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"[数据] 跳过 {f}: {e}")
            if len(self.data) >= self.max_size:
                break
        fresh = self.data[:]          # _load_file 按「新→旧」的文件顺序追加
        fresh.reverse()               # 统一成「旧→新」，与缓冲的方向约定一致
        self.data.clear()
        if cached:
            alive = {os.path.normcase(os.path.abspath(p)) for p, _, _ in entries}
            # 同名文件被改写过（大小/时间变了）时，缓存里的旧样本必须丢掉
            reread = {os.path.normcase(os.path.abspath(f)) for f in new_files}
            self.data.extend(s for s in cached
                             if s.source in alive and s.source not in reread)
        self.data.extend(fresh)
        # 超容量时丢**最旧**的（头部），保留最新的那批。
        # 早先是 del self.data[max_size:]（截尾），缓存满时会把刚读进来的新文件样本
        # 整段砍掉，而缓存签名又把这些文件记成「已缓存」—— 那些对局就永远进不了缓冲。
        drop = len(self.data) - self.max_size
        if drop > 0:
            del self.data[:drop]
        if mode == "random":
            # 只打乱顺序（不换集合）：配合 add_game 的「从头部淘汰」，
            # 效果是随机淘汰而不是随机初始化。
            random.shuffle(self.data)
        self.last_skipped = skipped
        if skipped:
            print(f"[数据] 共跳过 {skipped} 个不兼容/损坏文件（旧版数据请重新生成）")
        if new_files or cached is None:
            self._save_cache_async(self.data, sig)
        print(f"[数据] 已加载 {len(self.data)} 条（{len(entries)} 个文件，"
              f"新增 {len(new_files)} 个，模式 {mode}，耗时 {time.time() - t0:.1f}s）")
        return bool(self.data)

    # ---------------- 整目录缓存 ----------------
    def _cache_path(self):
        # 带上 max_size：换容量等于换数据集，缓存不能混用
        return os.path.join(self.data_dir, "_cache_v%d_%d_%d_o%d.pkl"
                            % (DATA_VERSION, self.board_size, self.max_size,
                               CACHE_ORDER))

    def _load_cache(self):
        path = self._cache_path()
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and isinstance(obj.get("samples"), list):
                return obj["samples"], obj.get("sig") or {}
        except FileNotFoundError:
            return None, {}
        except Exception as e:
            print(f"[数据] 缓存不可用，重建：{e}")
        return None, {}

    def _save_cache_async(self, samples, sig):
        """后台写缓存：启动不被 3 秒的 pickle dump 阻塞。

        进程提前退出只会留下旧缓存（下次把那几个新文件重读一遍），不会损坏。
        """
        snapshot = list(samples)
        threading.Thread(target=self._save_cache, args=(snapshot, sig),
                         daemon=True).start()

    def _save_cache(self, samples, sig):
        path = self._cache_path()
        # 临时名唯一：后台线程可能和另一个实例的 dump 撞在一起
        tmp = "%s.%d.%d.tmp" % (path, os.getpid(), threading.get_ident())
        try:
            with open(tmp, "wb") as f:
                pickle.dump({"samples": list(samples), "sig": sig}, f, protocol=4)
            os.replace(tmp, path)
        except Exception as e:
            print(f"[数据] 写缓存失败（不影响训练）：{e}")
            try:
                os.remove(tmp)
            except OSError:
                pass

    def load_single(self, filename):
        try:
            n = self._load_file(filename)
        except Exception as e:
            print(f"[数据] 拒绝加载 {os.path.basename(filename)}: {e}")
            return False
        print(f"[数据] 已从 {os.path.basename(filename)} 加载 {n} 条")
        return n > 0

    def _load_file(self, filename):
        with np.load(filename, allow_pickle=False) as d:
            ver = int(d["version"])
            bs = int(d["board_size"])
            if ver not in SUPPORTED_DATA_VERSIONS:
                raise ValueError(f"数据版本 {ver} 不受支持（支持 {SUPPORTED_DATA_VERSIONS}）")
            if bs != self.board_size:
                raise ValueError(f"棋盘 {bs} != {self.board_size}")
            komi = float(d["komi"])
            # 负数贴目 = 倒贴目，同样是合格数据，只查有限性与绝对范围。
            if not np.isfinite(komi) or komi < -100.0 or komi > 100.0:
                raise ValueError(f"贴目 {komi} 超出合理范围")
            if self.encoder is None and not is_half_step_komi(komi):
                raise ValueError(f"贴目 {komi} 不是 0.5 的整数倍")
            hole_bits = d["hole_bits"].astype(np.uint8)
            if int(hole_bits.size) == 0:
                hole_bits = None
            own_mask = int(d["own_mask"]) if "own_mask" in d.files else 1
            if own_mask not in (0, 1):
                raise ValueError(f"own_mask {own_mask} 必须是 0/1")
            boards = d["boards"]
            players = d["players"]
            lasts = d["lasts"]
            legals = d["legals"]
            feats = d["feats"]
            lens = d["pol_len"].astype(np.int64)
            pidx = d["pol_idx"]
            pp = d["pol_p"]
            winner = int(d["winner"])
            score_diff = float(d["score_diff"]) if "score_diff" in d.files else 0.0
            resigned = bool(int(d["resigned"])) if "resigned" in d.files else False
            own_known = (d["own_known"].astype(np.uint8)
                         if "own_known" in d.files else None)
            own = d["ownership"].astype(np.int8)
            cap_own_arr = d["cap_own"].astype(np.int16)
            cap_opp_arr = d["cap_opp"].astype(np.int16)
        off = 0
        count = 0
        for i in range(len(boards)):
            k = int(lens[i])
            if len(self.data) >= self.max_size:
                break
            s = Sample(boards[i], players[i], lasts[i], legals[i], feats[i],
                       pidx[off:off + k], pp[off:off + k], own, winner,
                       komi=komi, own_mask=own_mask,
                       source=os.path.normcase(os.path.abspath(filename)), index=i,
                       hole_bits=hole_bits,
                       cap_own=int(cap_own_arr[i]), cap_opp=int(cap_opp_arr[i]),
                       score_diff=score_diff, resigned=resigned,
                       own_known=own_known)
            self.data.append(s)
            off += k
            count += 1
        return count