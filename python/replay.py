"""紧凑、磁盘友好的围棋训练数据。

每局一个 .npz，保存的是重建 10 通道输入所需的**最小信息**：
  board/player/last/legal + 4 个 uint8 特征平面（危急度x3、眼x2）；
  第 10 通道是棋盘边界，运行时按尺寸和 hollow 生成，不落盘。
策略只存非零项（top-p 后通常十几个点），不存 362 维 float16 稠密向量。
归属标签每局只存一份，样本只引用它；own_mask 标记这批样本是否参与归属损失。
贴目按局存储：模型输入通道 3 用该局真实贴目，不再强制 7.5。

旧 pkl / 旧 6 通道 / 旧 7 通道数据不兼容，也不尝试兼容。
"""
import glob
import os
import random
from datetime import datetime

import numpy as np

from hyperparams import (BOARD_SIZE, KOMI, PASS_MOVE, INPUT_CHANNELS,
                         CH_BOUNDARY, validate_hollow)
from game import boundary_plane


DATA_VERSION = 5              # v5: 每局 hollow；v3/v4 仍可读（hollow 默认 0）
SUPPORTED_DATA_VERSIONS = (3, 4, 5)
FEATURE_CHANNELS = 4


class Sample:
    """单条训练样本的紧凑内存表示。

    source/index 只用于「界面手动改标签」时定位磁盘上的同一条样本；
    新对局样本没有磁盘来源，保持 None/-1。
    """
    __slots__ = ("board", "player", "last", "ko", "legal", "feat", "pidx", "pp", "own", "winner",
                 "komi", "hollow", "own_mask", "source", "index")

    def __init__(self, board, player, last, ko, legal, feat, pidx, pp, own, winner,
                 komi=KOMI, hollow=0, own_mask=1, source=None, index=-1):
        self.board = board
        self.ko = int(ko)
        self.player = int(player)
        self.last = int(last)
        self.legal = legal
        self.feat = feat
        self.pidx = pidx
        self.pp = pp
        self.own = own
        self.winner = int(winner)
        self.komi = float(komi)
        self.hollow = int(hollow)
        self.own_mask = int(own_mask)
        self.source = source
        self.index = int(index)


def state_to_sample(state, policy, player, last_move, ko_point=None):
    """把 10 通道规范状态 + 稠密策略转成紧凑样本（不保存通道原图；边界按尺寸/镂空重建）。"""
    state = np.asarray(state, dtype=np.float32)
    n = state.shape[-1]
    total = n * n
    board = ((state[0] > 0.5).astype(np.int8)
             - (state[1] > 0.5).astype(np.int8)).reshape(total)
    feat = np.empty((FEATURE_CHANNELS, total), dtype=np.uint8)
    feat[0] = np.rint(state[4].reshape(total) * 3.0).astype(np.uint8)
    feat[1] = np.rint(state[5].reshape(total) * 3.0).astype(np.uint8)
    feat[2] = np.rint(state[7].reshape(total) * 2.0).astype(np.uint8)
    feat[3] = np.rint(state[8].reshape(total) * 2.0).astype(np.uint8)
    legal = np.packbits(state[6].reshape(total) > 0.5)
    if last_move is None or last_move == PASS_MOVE or last_move[0] < 0:
        last = -1
    else:
        last = int(last_move[0]) * n + int(last_move[1])
    if ko_point is None or ko_point[0] < 0:
        ko = -1
    else:
        ko = int(ko_point[0]) * n + int(ko_point[1])
    policy = np.asarray(policy, dtype=np.float32).reshape(-1)
    pidx = np.flatnonzero(policy > 0).astype(np.uint16)
    pp = policy[pidx].astype(np.float16)
    if pidx.size == 0:
        legal_idx = np.flatnonzero(state[6].reshape(total) > 0.5)
        pidx = legal_idx.astype(np.uint16)
        pp = np.full(pidx.size, 1.0 / max(pidx.size, 1), dtype=np.float16)
    return {
        "board": board,
        "player": int(player),
        "last": last,
        "ko": ko,
        "legal": legal,
        "feat": feat,
        "pidx": pidx,
        "pp": pp,
    }


def pack_game(samples, moves, winner, score_diff, ownership, board_size=BOARD_SIZE,
              komi=KOMI, hollow=0, own_mask=1):
    """把一局紧凑样本打包成 savez 用的 dict。

    komi 是该局真实贴目；hollow 是中心洞边长（0=实心）；
    own_mask=0 表示这局只训练 policy/win，不训练 ownership。
    """
    n = int(board_size)
    hollow = validate_hollow(n, hollow)
    t = len(samples)
    if t == 0:
        raise ValueError("空对局不能保存")
    boards = np.stack([s["board"] for s in samples]).astype(np.int8)
    players = np.array([s["player"] for s in samples], dtype=np.int8)
    lasts = np.array([s["last"] for s in samples], dtype=np.int16)
    kos = np.array([s.get("ko", -1) for s in samples], dtype=np.int16)
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
        "kos": kos,
        "legals": legals,
        "feats": feats,
        "pol_len": lens,
        "pol_idx": pidx,
        "pol_p": pp,
        "moves": move_codes,
        "winner": np.int8(winner),
        "score_diff": np.float32(score_diff),
        "ownership": np.asarray(ownership, dtype=np.int8).reshape(n, n),
        "komi": np.float32(komi),
        "hollow": np.int16(hollow),
        "own_mask": np.uint8(own_mask),
    }


def payload_arrays(payload, board_size=BOARD_SIZE):
    """把 pack_game 的 dict 转成 npz 的键值；贴目/归属掩码按 payload 落盘。"""
    n = int(board_size)
    return {
        "version": np.int16(DATA_VERSION),
        "board_size": np.int16(n),
        "komi": np.float32(payload.get("komi", KOMI)),
        "hollow": np.int16(payload.get("hollow", 0)),
        "own_mask": np.uint8(payload.get("own_mask", 1)),
        "boards": payload["boards"],
        "players": payload["players"],
        "lasts": payload["lasts"],
        "kos": payload.get("kos", np.full(len(payload["boards"]), -1, dtype=np.int16)),
        "legals": payload["legals"],
        "feats": payload["feats"],
        "pol_len": payload["pol_len"],
        "pol_idx": payload["pol_idx"],
        "pol_p": payload["pol_p"],
        "moves": payload["moves"],
        "winner": payload["winner"],
        "score_diff": payload["score_diff"],
        "ownership": payload["ownership"],
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

    def __init__(self, max_size=200000, data_dir="data/", board_size=BOARD_SIZE):
        self.max_size = int(max_size)
        self.data_dir = data_dir
        self.board_size = int(board_size)
        self.data = []
        self.files = 0
        self.last_skipped = 0
        self.last_files = 0
        self._boundary_cache = {}
        os.makedirs(data_dir, exist_ok=True)

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
        hollow = validate_hollow(self.board_size, payload.get("hollow", 0))
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
                s["board"], s["player"], s["last"], s.get("ko", -1),
                s["legal"], s["feat"], s["pidx"], s["pp"], own, winner,
                komi=komi, hollow=hollow, own_mask=own_mask,
                source=source, index=start + j))

    def sample(self, batch_size):
        batch = random.sample(self.data, min(int(batch_size), len(self.data)))
        b = len(batch)
        n = self.board_size
        total = n * n
        states = np.zeros((b, INPUT_CHANNELS, n, n), dtype=np.float32)
        policies = np.zeros((b, total + 1), dtype=np.float32)
        values = np.zeros((b,), dtype=np.float32)
        own = np.zeros((b, n, n), dtype=np.int8)
        kos = np.zeros((b,), dtype=np.int16)
        own_masks = np.zeros((b,), dtype=np.uint8)
        eye3 = 1.0 / 3.0
        eye2 = 0.5
        for j, s in enumerate(batch):
            board = s.board.reshape(n, n)
            p = s.player
            kos[j] = s.ko
            own_masks[j] = s.own_mask
            key = (n, s.hollow)
            bd = self._boundary_cache.get(key)
            if bd is None:
                bd = boundary_plane(n, s.hollow)
                self._boundary_cache[key] = bd
            states[j, 0] = board > 0
            states[j, 1] = board < 0
            if s.last >= 0:
                states[j, 2, s.last // n, s.last % n] = 1.0
            states[j, 3] = p * s.komi / 14.0
            states[j, 4] = s.feat[0].reshape(n, n) * eye3
            states[j, 5] = s.feat[1].reshape(n, n) * eye3
            states[j, 6] = np.unpackbits(s.legal, count=total).reshape(n, n)
            states[j, 7] = s.feat[2].reshape(n, n) * eye2
            states[j, 8] = s.feat[3].reshape(n, n) * eye2
            states[j, CH_BOUNDARY] = bd
            policies[j, s.pidx] = s.pp.astype(np.float32)
            if s.winner != 0:
                values[j] = 1.0 if s.winner == p else -1.0
            own[j] = s.own * p
        return states, policies, values, own, kos, own_masks

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
        self.files += 1
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
        self.last_files = len(files)
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
        n = self._load_file(filename)
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
            hollow = int(d["hollow"]) if "hollow" in d.files else 0
            hollow = validate_hollow(bs, hollow)
            own_mask = int(d["own_mask"]) if "own_mask" in d.files else 1
            if own_mask not in (0, 1):
                raise ValueError(f"own_mask {own_mask} 必须是 0/1")
            boards = d["boards"]
            players = d["players"]
            lasts = d["lasts"]
            kos = d["kos"] if "kos" in d.files else np.full(len(boards), -1, dtype=np.int16)
            legals = d["legals"]
            feats = d["feats"]
            lens = d["pol_len"].astype(np.int64)
            pidx = d["pol_idx"]
            pp = d["pol_p"]
            winner = int(d["winner"])
            own = d["ownership"].astype(np.int8)
        off = 0
        count = 0
        for i in range(len(boards)):
            k = int(lens[i])
            if len(self.data) >= self.max_size:
                break
            s = Sample(boards[i], players[i], lasts[i], kos[i], legals[i], feats[i],
                       pidx[off:off + k], pp[off:off + k], own, winner,
                       komi=komi, hollow=hollow, own_mask=own_mask,
                       source=os.path.normcase(os.path.abspath(filename)), index=i)
            self.data.append(s)
            off += k
            count += 1
        return count

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]