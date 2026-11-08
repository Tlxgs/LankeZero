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
import glob
import os
import random
import threading
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
                         CH_OWN_ALIVE, CH_OPP_ALIVE,
                         is_half_step_komi, HOLE)
from game import boundary_plane_from_mask, _ladder_channels, _benson_channels


DATA_VERSION = 9              # v9: 删掉眼通道平面（feats 由 4 排变 2 排）；旧数据用 tools/migrate_drop_eyes.py 升级
SUPPORTED_DATA_VERSIONS = (9,)
FEATURE_CHANNELS = 2

# 手动标注（UI 线程）与采样（训练线程）之间保护 Sample.pidx/pp 这对字段。
_LABEL_LOCK = threading.Lock()


class Sample:
    """单条训练样本的紧凑内存表示。

    source/index 只用于「界面手动改标签」时定位磁盘上的同一条样本；
    新对局样本没有磁盘来源，保持 None/-1。
    """
    __slots__ = ("board", "player", "last", "legal", "feat", "pidx", "pp", "own", "winner",
                 "komi", "own_mask", "source", "index", "hole_bits",
                 "cap_own", "cap_opp")

    def __init__(self, board, player, last, legal, feat, pidx, pp, own, winner,
                 komi=KOMI, own_mask=1, source=None, index=-1, hole_bits=None,
                 cap_own=0, cap_opp=0):
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

    def hole_mask(self, n):
        """本局形状掩码。位图为空表示整盘可下。"""
        if self.hole_bits is None or not len(self.hole_bits):
            return shapes.solid(int(n))
        return shapes.unpack_bits(self.hole_bits, int(n))

    def set_policy(self, pidx, pp):
        """成对替换策略标签（界面手动标注用）。

        pidx 与 pp 是两条独立赋值，训练线程可能夹在中间读到半新半旧的一对，
        所以读写都走同一把锁。
        """
        with _LABEL_LOCK:
            self.pidx = pidx
            self.pp = pp

    def get_policy(self):
        with _LABEL_LOCK:
            return self.pidx, self.pp


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
              komi=KOMI, own_mask=1, hole=None, start_stones=None):
    """把一局紧凑样本打包成 savez 用的 dict。

    hole 是棋盘形状掩码（(n,n) bool，True=洞）；None/全 False 表示实心盘。
    own_mask=0 表示这局只训练 policy/win，不训练 ownership。
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
        "start_stones": np.asarray(payload.get("start_stones",
                                                np.zeros((n, n), dtype=np.int8)),
                                   dtype=np.int8).reshape(n, n),
    }


def load_npz_arrays(path):
    """把一局 .npz 的所有键读成普通 ndarray（不保留 NpzFile 句柄）。"""
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k].copy() for k in d.files}


def policy_segment(arrays, sample_index):
    """返回某条样本的 (pol_idx, pol_p) 片段。"""
    lens = arrays["pol_len"].astype(np.int64).reshape(-1)
    i = int(sample_index)
    if not 0 <= i < len(lens):
        raise IndexError("样本下标 %d 超出 0..%d" % (i, len(lens) - 1))
    start = int(lens[:i].sum())
    end = start + int(lens[i])
    return arrays["pol_idx"][start:end], arrays["pol_p"][start:end]


def set_policy_onehot(path, sample_index, move_index):
    """把一局 .npz 中第 sample_index 条样本的 policy 改成 move_index 的 one-hot。

    move_index 是扁平着法下标：0..n*n-1 为盘面点，n*n 为 pass。
    只改 pol_len/pol_idx/pol_p，不动 moves（回放仍显示原来的实战着法）。
    直接覆盖原文件：写临时文件并 os.replace 原子替换，不生成 .bak。
    返回重新读出的数组字典。
    """
    path = os.path.abspath(path)
    arrays = load_npz_arrays(path)
    if int(arrays["version"]) not in SUPPORTED_DATA_VERSIONS:
        raise ValueError("数据版本 %d 不受支持（支持 %s）" % (
            int(arrays["version"]), SUPPORTED_DATA_VERSIONS))
    n = int(arrays["board_size"])
    total = n * n + 1
    lens = arrays["pol_len"].astype(np.int64).reshape(-1)
    i = int(sample_index)
    move = int(move_index)
    if not 0 <= i < len(lens):
        raise IndexError("样本下标 %d 超出 0..%d" % (i, len(lens) - 1))
    if not 0 <= move < total:
        raise ValueError("着法下标 %d 超出 0..%d" % (move, total - 1))
    if len(arrays["pol_idx"]) != int(lens.sum()):
        raise ValueError("pol_idx 长度与 pol_len 不一致")
    starts = np.concatenate(([0], np.cumsum(lens)))
    idx = arrays["pol_idx"]
    pp = arrays["pol_p"]
    new_idx = np.concatenate((idx[:starts[i]],
                              np.array([move], dtype=idx.dtype),
                              idx[starts[i + 1]:]))
    new_pp = np.concatenate((pp[:starts[i]],
                             np.array([1.0], dtype=pp.dtype),
                             pp[starts[i + 1]:]))
    arrays["pol_idx"] = new_idx
    arrays["pol_p"] = new_pp
    new_lens = lens.copy()
    new_lens[i] = 1
    arrays["pol_len"] = new_lens.astype(arrays["pol_len"].dtype)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return arrays


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
        own_mask = int(payload.get("own_mask", 1))
        take = samples[-self.max_size:]
        start = len(samples) - len(take)
        if source is not None:
            source = os.path.normcase(os.path.abspath(source))
        drop = len(self.data) + len(take) - self.max_size
        if drop > 0:
            del self.data[:drop]
        for j, s in enumerate(take):
            self.data.append(Sample(
                s["board"], s["player"], s["last"],
                s["legal"], s["feat"], s["pidx"], s["pp"], own, winner,
                komi=komi, own_mask=own_mask,
                source=source, index=start + j, hole_bits=hole_bits,
                cap_own=s.get("cap_own", 0), cap_opp=s.get("cap_opp", 0)))

    def sample(self, batch_size):
        batch = random.sample(self.data, min(int(batch_size), len(self.data)))
        b = len(batch)
        n = self.board_size
        total = n * n
        if self.encoder is not None:
            # 其他棋种：编码器自己决定通道数和全局向量（go 的 11 通道见 _fill_go_states）。
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
        for j, s in enumerate(batch):
            p = s.player
            # 手动标注会在 UI 线程改写 pidx/pp。两者必须成对读，否则会拿到
            # 「新下标 + 旧概率」，广播时形状不匹配直接把整轮训练打挂。
            pidx, pp = s.get_policy()
            own_masks[j] = s.own_mask
            policies[j, pidx] = pp.astype(np.float32)
            if s.winner != 0:
                values[j] = 1.0 if s.winner == p else -1.0
            own[j] = s.own * p
        return states, globals_, policies, values, own, own_masks

    def _fill_go_states(self, batch, states, globals_):
        """围棋的 11 通道编码 + 6 维全局向量（改动前 sample() 里的内联逻辑）。"""
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
            # Benson 无条件活（两平面）。数据里的 board 是行棋方视角且洞是 0，
            # 洞必须补成 HOLE 哨兵，否则会被当成空点当眼。
            b_benson = board.copy()
            b_benson[mask] = HOLE
            _benson_channels(b_benson, n, p, states[j, CH_OWN_ALIVE],
                             states[j, CH_OPP_ALIVE])

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
        files = glob.glob(os.path.join(self.data_dir, "*.npz"))
        if not files:
            return False
        if mode == "random":
            random.shuffle(files)
        else:
            files.sort(key=os.path.getmtime, reverse=True)
        self.data.clear()
        loaded = 0
        skipped = 0
        for f in files:
            try:
                n = self._load_file(f)
                loaded += n
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"[数据] 跳过 {f}: {e}")
            if len(self.data) >= self.max_size:
                break
        self.last_skipped = skipped
        if skipped:
            print(f"[数据] 共跳过 {skipped} 个不兼容/损坏文件（旧版数据请重新生成）")
        print(f"[数据] 已加载 {len(self.data)} 条 / {loaded} 条（{len(files)} 个文件，模式 {mode}）")
        return bool(self.data)

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
            if not np.isfinite(komi) or komi < 0.0 or komi > 100.0:
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
                       cap_own=int(cap_own_arr[i]), cap_opp=int(cap_opp_arr[i]))
            self.data.append(s)
            off += k
            count += 1
        return count