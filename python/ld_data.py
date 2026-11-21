"""死活题 → 两个 npz 棋局（黑先黑胜 / 白先白胜）。

一题 = 一张盘面 + 两个关键点（黑杀点、白活点，可以同一点）。落成两个标准
npz，和自对弈产物完全同格式，`replay.TrainingData` 直接能读：

    <name>_B.npz   黑先，黑必胜，策略 one-hot 在黑关键点
    <name>_W.npz   白先，白必胜，策略 one-hot 在白关键点

标签约定（与题目规则一致）：
  * 贴目 0，无历史（没有上一手），无提子记录；
  * 价值用胜负：行棋方必胜 → value=+1（加载时由 ``winner == player`` 推出）；
  * 归属头不监督：``own_mask=0``，``ownership`` 全 0；
  * 目差不用：``score_diff=0``（宏观领地项权重为 0，本来也不参与）。
"""
import os

import numpy as np

import replay
import shapes
from game import GoGame

LD_BOARD = 13
LD_KOMI = 0.0


def build_samples(mask, stones, black_pt, white_pt, board_size=LD_BOARD,
                  komi=LD_KOMI):
    """返回 (黑先样本, 白先样本)。局面非法时抛 ValueError。"""
    n = int(board_size)
    hole = None if mask is None or not np.asarray(mask).any() else np.asarray(mask, bool)
    game = GoGame(n, komi=komi, hole_mask=hole, start_board=stones)
    total = n * n

    def one(point, player):
        state = game.get_canonical_state(player=player)
        policy = np.zeros(total + 1, dtype=np.float32)
        if point is None:
            policy[total] = 1.0                 # 没标点就退化成一整手 pass
        else:
            policy[int(point[0]) * n + int(point[1])] = 1.0
        return replay.state_to_sample(state, policy, int(player), last_move=None)

    return one(black_pt, 1), one(white_pt, -1)


def build_npz(name, mask, stones, black_pt, white_pt, out_dir=None,
              board_size=LD_BOARD, komi=LD_KOMI):
    """写 ``<name>_B.npz`` / ``<name>_W.npz``，返回两个路径。

    ``out_dir`` 默认是 ``shapes.LD_DIR``（go/ld/）。局面非法时抛 ValueError，
    不会留下半成品文件。
    """
    if black_pt is None or white_pt is None:
        raise ValueError("缺少关键点（黑 %s / 白 %s），不生成 npz"
                         % (black_pt, white_pt))
    n = int(board_size)
    out_dir = shapes.LD_DIR if out_dir is None else str(out_dir)
    a, b = build_samples(mask, stones, black_pt, white_pt, n, komi)
    hole = None if mask is None or not np.asarray(mask).any() else np.asarray(mask, bool)
    own = np.zeros((n, n), dtype=np.int8)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for suffix, sample, winner in (("B", a, 1), ("W", b, -1)):
        payload = replay.pack_game([sample], [], winner, 0.0, own, n,
                                   komi=komi, own_mask=0, hole=hole,
                                   resigned=False)
        path = os.path.join(out_dir, "%s_%s.npz" % (name, suffix))
        np.savez_compressed(path, **replay.payload_arrays(payload, n))
        paths.append(path)
    return paths


def build_all(ld_dir=None, board_size=LD_BOARD, verbose=True):
    """把目录下所有 .txt 题重建成 npz（跳过 '_' 开头的模板）。返回 (ok, failed)。"""
    ld_dir = shapes.LD_DIR if ld_dir is None else str(ld_dir)
    ok, failed = [], []
    for fn in sorted(os.listdir(ld_dir)):
        if not fn.endswith(".txt") or fn.startswith("_"):
            continue
        path = os.path.join(ld_dir, fn)
        name = fn[:-4]
        try:
            mask, stones, bpt, wpt = shapes.load_ld_file(path, board_size=board_size)
            if bpt is None or wpt is None:
                raise ValueError("缺少关键点（黑 %s / 白 %s）" % (bpt, wpt))
            paths = build_npz(name, mask, stones, bpt, wpt, out_dir=ld_dir,
                              board_size=board_size)
            ok.append((name, paths))
            if verbose:
                print("[ok] %-24s → %s" % (name, ", ".join(os.path.basename(p) for p in paths)))
        except Exception as exc:                       # noqa: BLE001
            failed.append((name, str(exc)))
            if verbose:
                print("[!!] %-24s %s" % (name, exc))
    return ok, failed
