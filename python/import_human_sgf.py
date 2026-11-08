"""把 Human/ 下的 SGF 转成 replay v5 的紧凑训练数据（hollow=0）。

只收 19 路、无让子、有明确胜负、主线逐手合法的对局。
policy 用实战着法 one-hot；value 用 SGF 结果；
ownership 全掩码（own_mask=0），不参与损失。
贴目按局写入，支持 0 / 5.5 / 6.5 / 7.5 等不同值。
"""
import argparse
import csv
import glob
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict, deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from game import GoGame
from hyperparams import PROJECT_ROOT, is_half_step_komi
from replay import pack_game, payload_arrays, state_to_sample


class SkipGame(Exception):
    pass


def tokenize(text):
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in "();":
            yield c
            i += 1
            continue
        if c.isalpha():
            j = i
            while j < n and (text[j].isalpha() or text[j].isdigit() or text[j] == "_"):
                j += 1
            ident = text[i:j]
            i = j
            vals = []
            while i < n and text[i].isspace():
                i += 1
            while i < n and text[i] == "[":
                i += 1
                buf = []
                while i < n:
                    if text[i] == "\\":
                        i += 1
                        if i < n:
                            buf.append(text[i])
                            i += 1
                    elif text[i] == "]":
                        i += 1
                        break
                    else:
                        buf.append(text[i])
                        i += 1
                vals.append("".join(buf))
                while i < n and text[i].isspace():
                    i += 1
            yield ident, vals
            continue
        i += 1


class Node:
    __slots__ = ("props", "children")

    def __init__(self):
        self.props = {}
        self.children = []


def parse_collection(text):
    roots = []
    stack = []
    cur = None
    for tok in tokenize(text):
        if tok == "(":
            node = Node()
            if cur is not None:
                cur.children.append(node)
            stack.append((cur, node))
            cur = node
        elif tok == ")":
            if stack:
                parent, node = stack.pop()
                if parent is None:
                    roots.append(node)
                cur = parent
        elif tok == ";":
            node = Node()
            if cur is not None:
                cur.children.append(node)
            cur = node
        else:
            ident, vals = tok
            if cur is None:
                cur = Node()
            cur.props.setdefault(ident, []).extend(vals)
    return roots


def get_prop(node, key, default=""):
    vals = node.props.get(key)
    return vals[0] if vals else default


def real_root(tree):
    while not tree.props and tree.children:
        tree = tree.children[0]
    return tree


def mainline(root):
    out = []
    node = root
    while True:
        out.append(node)
        if not node.children:
            break
        node = node.children[0]
    return out


def norm_komi(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    # 这批 SGF 把 7.5 写成 750、6.5 写成 650；也兼容标准的 7.5。
    if x >= 100.0 and abs(x - round(x)) < 1e-9:
        x /= 100.0
    return x


def parse_result(value):
    m = re.match(r"^\s*([BW])\s*\+", value or "")
    if not m:
        return None, 0.0
    winner = 1 if m.group(1) == "B" else -1
    score = 0.0
    sm = re.match(r"^\s*[BW]\s*\+\s*([0-9]+(?:\.[0-9]+)?)", value)
    if sm:
        score = float(sm.group(1)) * winner
    return winner, score


def parse_coord(text, n=19):
    if not text or text == "tt":
        return -1, -1
    if len(text) != 2:
        return None
    c = ord(text[0]) - ord("a")
    r = ord(text[1]) - ord("a")
    if not (0 <= r < n and 0 <= c < n):
        return None
    return r, c


def extract_moves(root, n=19):
    moves = []
    for node in mainline(root):
        cols = [c for c in ("B", "W") if c in node.props]
        if not cols:
            continue
        if len(cols) != 1:
            raise SkipGame("node_has_both_colors")
        vals = node.props[cols[0]]
        if len(vals) != 1:
            raise SkipGame("move_has_multiple_values")
        coord = parse_coord(vals[0], n)
        if coord is None:
            raise SkipGame("bad_coord")
        moves.append((cols[0], coord))
    return moves


def read_sgf(path, min_moves, max_moves):
    try:
        text = open(path, "r", encoding="utf-8", errors="replace").read()
    except OSError as exc:
        raise SkipGame("read_error") from exc
    roots = parse_collection(text)
    if len(roots) != 1:
        raise SkipGame("multi_game")
    root = real_root(roots[0])
    props = root.props
    size = get_prop(root, "SZ", "19").split(":")[0]
    if size != "19":
        raise SkipGame("not_19")
    if any(k in props for k in ("AB", "AW", "AE")):
        raise SkipGame("setup_stones")
    try:
        handicap = int(float(get_prop(root, "HA", "0") or 0))
    except ValueError as exc:
        raise SkipGame("bad_handicap") from exc
    if handicap != 0:
        raise SkipGame("handicap")
    komi = norm_komi(get_prop(root, "KM", ""))
    if komi is None or not np.isfinite(komi) or komi < 0.0 or komi > 100.0:
        raise SkipGame("bad_komi")
    # 贴目必须是 x.5：整数贴目（大量让子局/老棋谱是 0）会让终局目差落到
    # 分数头两个箱之间，整局直接不要。
    if not is_half_step_komi(komi):
        raise SkipGame("komi_not_half")
    winner, score_diff = parse_result(get_prop(root, "RE", ""))
    if winner is None:
        raise SkipGame("no_result")
    moves = extract_moves(root)
    if len(moves) < min_moves:
        raise SkipGame("too_short")
    if len(moves) > max_moves:
        raise SkipGame("too_long")
    return {
        "path": path,
        "rank": os.path.basename(os.path.dirname(path)),
        "komi": float(komi),
        "result": get_prop(root, "RE", ""),
        "winner": int(winner),
        "score_diff": float(score_diff),
        "moves": moves,
    }


def make_payload(meta, board_size=19):
    n = board_size
    game = GoGame(n, komi=meta["komi"])
    samples = []
    played = []
    total = n * n + 1
    for color, (r, c) in meta["moves"]:
        if game.game_over:
            break
        expected = 1 if color == "B" else -1
        if game.current_player != expected:
            raise SkipGame("color_order")
        if not game.is_legal_move(r, c):
            raise SkipGame("illegal_move")
        _legal_moves, legal_mask = game.get_legal_moves_and_mask()
        state = game.get_canonical_state(legal_mask=legal_mask)
        policy = np.zeros(total, dtype=np.float32)
        policy[n * n if (r < 0 or c < 0) else r * n + c] = 1.0
        samples.append(state_to_sample(
            state, policy, game.current_player, game.last_move))
        game.make_move(r, c)
        played.append((r, c, expected))
    if len(samples) < 20:
        raise SkipGame("too_few_samples")
    # 引擎已终局但 SGF 还有非 pass 着法：说明规则截断，放弃这局。
    if len(played) < len(meta["moves"]):
        for _color, (r, c) in meta["moves"][len(played):]:
            if (r, c) != (-1, -1):
                raise SkipGame("ended_early")
    own = np.zeros((n, n), dtype=np.int8)
    return pack_game(samples, played, meta["winner"], meta["score_diff"], own,
                     board_size=n, komi=meta["komi"], own_mask=0)


def game_hash(meta):
    h = hashlib.sha1(("%.4f|%s|" % (meta["komi"], meta["result"])).encode("utf-8"))
    for color, (r, c) in meta["moves"]:
        h.update(("%s%d,%d;" % (color, r, c)).encode("ascii"))
    return h.hexdigest()


def rank_key(rank):
    m = re.match(r"^(\d+)p$", rank)
    return (1, int(m.group(1))) if m else (0, -1)


def ordered_candidates(metas):
    groups = defaultdict(lambda: defaultdict(deque))
    for meta in metas:
        groups[meta["rank"]][meta["komi"]].append(meta)
    ranks = sorted(groups, key=rank_key, reverse=True)
    komis = sorted({m["komi"] for m in metas})
    for rank in ranks:
        by_komi = groups[rank]
        for k in by_komi:
            by_komi[k] = deque(sorted(by_komi[k], key=lambda m: m["path"]))
        while any(by_komi[k] for k in komis):
            for k in komis:
                if by_komi[k]:
                    yield by_komi[k].popleft()


def safe_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", stem)


def write_npz(path, payload, board_size):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload_arrays(payload, board_size))
    os.replace(tmp, path)


def load_manifest(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]


def main():
    parser = argparse.ArgumentParser(description="Convert human SGF games to compact npz")
    parser.add_argument("--human", default=os.path.join(PROJECT_ROOT, "Human"))
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data_human19"))
    parser.add_argument("--limit", type=int, default=200,
                        help="fresh run: number of games to write")
    parser.add_argument("--total", type=int, default=0,
                        help="append run: target total games in --out")
    parser.add_argument("--append", action="store_true",
                        help="add to an existing --out; never overwrite .npz")
    parser.add_argument("--min-moves", type=int, default=20)
    parser.add_argument("--max-moves", type=int, default=420)
    args = parser.parse_args()

    existed = os.path.exists(args.out)
    if existed and not args.append:
        raise SystemExit("输出目录已存在；要追加请加 --append，拒绝覆盖: %s" % args.out)
    if not existed:
        os.makedirs(args.out)

    existing_paths = sorted(glob.glob(os.path.join(args.out, "*.npz"))) if existed else []
    existing_names = {os.path.basename(p) for p in existing_paths}
    if args.append:
        if args.total <= 0:
            raise SystemExit("追加模式必须给 --total")
        if len(existing_paths) > args.total:
            raise SystemExit("已有 %d 局，超过目标 %d" % (len(existing_paths), args.total))
        target_new = args.total - len(existing_paths)
        manifest = os.path.join(args.out, "manifest_%d.csv" % args.total)
    else:
        target_new = args.limit
        manifest = os.path.join(args.out, "manifest.csv")
    if os.path.exists(manifest):
        raise SystemExit("清单已存在，拒绝覆盖: %s" % manifest)

    files = sorted(glob.glob(os.path.join(args.human, "**", "*.sgf"), recursive=True))
    if not files:
        raise SystemExit("没有找到 SGF: %s" % args.human)

    metas = []
    skipped = Counter()
    for path in files:
        try:
            metas.append(read_sgf(path, args.min_moves, args.max_moves))
        except SkipGame as exc:
            skipped[str(exc)] += 1
        except Exception as exc:  # noqa: BLE001 - 单个坏文件不能中断整批
            skipped["parse_error"] += 1

    print("SGF total=%d candidates=%d skipped=%s" % (len(files), len(metas), dict(skipped)))
    if not metas:
        raise SystemExit("没有可用对局")

    old_rows = []
    if args.append:
        old_rows = [r for r in load_manifest(os.path.join(args.out, "manifest.csv"))
                    if r.get("npz") in existing_names]
        print("existing=%d target_new=%d" % (len(existing_paths), target_new))
        if target_new <= 0:
            print("nothing to add")
            return

    seen = set()
    new_rows = []
    accepted = 0
    skipped_existing = 0
    reasons = Counter()
    for meta in ordered_candidates(metas):
        if accepted >= target_new:
            break
        name = "human19_%s_%s.npz" % (meta["rank"], safe_stem(meta["path"]))
        out_path = os.path.join(args.out, name)
        if os.path.exists(out_path):
            skipped_existing += 1
            continue
        try:
            payload = make_payload(meta, 19)
            h = game_hash(meta)
            if h in seen:
                raise SkipGame("duplicate")
            suffix = 1
            while os.path.exists(out_path):
                out_path = os.path.join(args.out, "human19_%s_%s_%d.npz"
                                        % (meta["rank"], safe_stem(meta["path"]), suffix))
                suffix += 1
            write_npz(out_path, payload, 19)
            seen.add(h)
            accepted += 1
            new_rows.append({
                "npz": os.path.basename(out_path),
                "sgf": os.path.relpath(meta["path"], PROJECT_ROOT),
                "rank": meta["rank"],
                "komi": "%.1f" % meta["komi"],
                "result": meta["result"],
                "winner": "B" if meta["winner"] == 1 else "W",
                "score_diff": "%.2f" % meta["score_diff"],
                "samples": len(payload["boards"]),
            })
            if accepted % 20 == 0:
                print("accepted %d/%d (total %d)" % (
                    accepted, target_new, len(existing_paths) + accepted))
        except SkipGame as exc:
            reasons[str(exc)] += 1
        except Exception as exc:  # noqa: BLE001
            reasons["convert_error"] += 1

    rows = old_rows + new_rows
    with open(manifest, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["npz", "sgf", "rank", "komi", "result",
                                               "winner", "score_diff", "samples"])
        writer.writeheader()
        writer.writerows(rows)

    by_rank = Counter(r["rank"] for r in rows)
    by_komi = Counter(r["komi"] for r in rows)
    samples = sum(int(r["samples"]) for r in rows)
    total = len(existing_paths) + accepted
    print("done new=%d total=%d samples=%d out=%s" % (accepted, total, samples, args.out))
    print("by_rank=%s by_komi=%s" % (dict(sorted(by_rank.items())), dict(sorted(by_komi.items()))))
    print("skip_existing=%d skip_after_candidates=%s" % (skipped_existing, dict(reasons)))
    if accepted < target_new:
        print("WARNING: new %d < target_new %d" % (accepted, target_new))


if __name__ == "__main__":
    main()
