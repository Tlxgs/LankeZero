"""无界面对弈：人类 vs AI，或 AI vs AI。

用法示例：
  python play_headless.py
  python play_headless.py --board-size 13 --black human --white ai --sims 300
  python play_headless.py --black ai --white ai --games 5 --sgf eval.sgf
  python play_headless.py --help

默认与 gui.py 对弈界面一致：19 路、贴目 7.5、AI 贪心落子。
"""
import argparse
import os
import random
from datetime import datetime

import torch

from hyperparams import (BOARD_SIZE, BOARD_SIZES, KOMI, MODEL_PATH, NUM_SIMULATIONS,
                         BATCH_SIZE_MCTS, C_PUCT, ALPHA, VIRTUAL_LOSS, MCTS_CAP,
                         GUI_TEMPERATURE)
import shapes
from game import GoGame, PASS_MOVE
from model import PolicyValueNet
from evaluator import GoEvaluator
from mcts import MCTS


GO_COLUMNS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
SGF_COLUMNS = "abcdefghijklmnopqrstuvwxyz"


def parse_args():
    p = argparse.ArgumentParser(
        description="Headless Go: human vs AI or AI vs AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
人类输入：
  row col   0-based 坐标，例如 3 3
  D4        Go 坐标（列 A-T 跳过 I，行号从下往上 1 起）
  pass      停一手
  quit      退出

示例：
  python play_headless.py --board-size 13
  python play_headless.py --board-size 14 --shape square:6
  python play_headless.py --black ai --white ai --games 3 --sims 100
  python play_headless.py --board-size 13 --shape heart13 \
      --black ai --white ai --games 1 --sgf heart.sgf

形状：solid（默认）/ square:H（居中方块洞）/ ring / shapes/ 下的文件名。
""")
    p.add_argument("--board-size", type=int, choices=BOARD_SIZES, default=BOARD_SIZE)
    p.add_argument("--shape", default="solid",
                   help="棋盘形状：solid / square:H / ring / shapes/<名字>.txt")
    p.add_argument("--komi", type=float, default=KOMI)
    p.add_argument("--black", choices=("human", "ai"), default="human")
    p.add_argument("--white", choices=("human", "ai"), default="ai")
    p.add_argument("--games", type=int, default=1,
                   help="对局数量；人类参与时通常设 1")
    p.add_argument("--sims", type=int, default=NUM_SIMULATIONS)
    p.add_argument("--mcts-batch", type=int, default=BATCH_SIZE_MCTS)
    p.add_argument("--c-puct", type=float, default=C_PUCT)
    p.add_argument("--temperature", type=float, default=GUI_TEMPERATURE)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--vloss", type=int, default=VIRTUAL_LOSS)
    p.add_argument("--dir-eps", type=float, default=0.0,
                   help="对弈默认 0，避免随机噪声")
    p.add_argument("--dir-alpha", type=float, default=0.03)
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sgf", default=None, help="可选：保存 SGF 到该路径")
    p.add_argument("--no-board", action="store_true", help="不打印棋盘")
    return p.parse_args()


def player_name(player):
    return "黑" if player == 1 else "白"


def coord_text(r, c, n):
    if r < 0 or c < 0:
        return "pass"
    return "%s%d" % (GO_COLUMNS[c], n - r)


def parse_human_move(text, n):
    s = text.strip()
    if not s:
        return None
    low = s.lower()
    if low in ("quit", "q", "exit"):
        return "quit"
    if low in ("pass", "p", "停一手"):
        return PASS_MOVE
    if low in ("help", "h", "?"):
        return "help"
    parts = s.replace(",", " ").split()
    if len(parts) == 2 and all(x.lstrip("-").isdigit() for x in parts):
        r, c = int(parts[0]), int(parts[1])
        if 0 <= r < n and 0 <= c < n:
            return (r, c)
        raise ValueError("坐标超出 0..%d" % (n - 1))
    # Go 坐标：D4 / q16
    part = parts[0].upper()
    if len(part) >= 2 and part[0] in GO_COLUMNS:
        try:
            row = int(part[1:])
            c = GO_COLUMNS.index(part[0])
            r = n - row
        except ValueError:
            raise ValueError("无法识别坐标: %s" % s)
        if 0 <= r < n and 0 <= c < n:
            return (r, c)
        raise ValueError("坐标超出棋盘: %s" % s)
    raise ValueError("无法识别: %s（用 row col / D4 / pass / quit）" % s)


def print_board(game):
    print()
    print(game)
    print("X=黑  O=白  |  当前: %s" % player_name(game.current_player))


def ai_move(mcts, game, n, sims, temperature):
    policy, sample = mcts.get_move_probs(game, temp=temperature, target_visits=sims)
    if not sample:
        moves, _ = game.get_legal_moves_and_mask()
        if not moves:
            return None
        return moves[0]
    if temperature > 0.1:
        moves = list(sample.keys())
        weights = [sample[m] for m in moves]
        return random.choices(moves, weights=weights, k=1)[0]
    return max(sample, key=sample.get)


def sgf_coord(move):
    if move == PASS_MOVE or move[0] < 0:
        return ""
    return SGF_COLUMNS[move[1]] + SGF_COLUMNS[move[0]]


def result_text(game):
    fp = float(game.final_points)
    if game.winner == 1:
        return "黑胜 %.1f 目" % abs(fp)
    if game.winner == -1:
        return "白胜 %.1f 目" % abs(fp)
    return "和棋"


def result_sgf(game):
    fp = float(game.final_points)
    if game.winner == 1:
        return "B+%.1f" % abs(fp) if fp != 0 else "B+"
    if game.winner == -1:
        return "W+%.1f" % abs(fp) if fp != 0 else "W+"
    return "0"


def write_sgf(path, game, moves, n, komi, hole_mask=None, start_stones=None):
    comment = ""
    setup = ""
    if hole_mask is not None and hole_mask.any():
        comment = "C[SHAPE %s]" % shapes.to_text(hole_mask).replace("\n", "|")
    if start_stones is not None and start_stones.any():
        ab = [sgf_coord((r, c))
              for r in range(n) for c in range(n) if start_stones[r, c] == 1]
        aw = [sgf_coord((r, c))
              for r in range(n) for c in range(n) if start_stones[r, c] == -1]
        if ab:
            setup += "AB[%s]" % "][".join(ab)
        if aw:
            setup += "AW[%s]" % "][".join(aw)
    lines = ["(;GM[1]FF[4]CA[UTF-8]",
             "SZ[%d]KM[%.1f]RU[Chinese]PB[Black]PW[White]DT[%s]%s%s"
             % (n, komi, datetime.now().strftime("%Y-%m-%d"), comment, setup),
             "RE[%s]" % result_sgf(game)]
    for color, (r, c) in moves:
        lines.append(";%s[%s]" % (color, sgf_coord((r, c))))
    lines.append(")")
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines) + "\n")


def one_game(args, mcts):
    n = args.board_size
    game = GoGame(n, komi=args.komi, hole_mask=args.hole_mask,
                  start_board=args.start_stones)
    mcts.reset()
    moves = []   # (B/W, (r,c))
    humans = {1: args.black == "human", -1: args.white == "human"}

    print("=" * 60)
    print("新对局: %d 路 | 形状 %s | 可下 %d 点 | 贴目 %.1f | 黑=%s 白=%s"
          % (n, args.shape, shapes.playable_area(args.hole_mask),
             args.komi, args.black, args.white))
    if not args.no_board:
        print_board(game)

    while not game.game_over:
        player = game.current_player
        human = humans[player]
        if human:
            prompt = "%s方落子 (row col / D4 / pass / quit): " % player_name(player)
            try:
                text = input(prompt)
            except EOFError:
                print("\n输入结束，退出")
                return None
            try:
                parsed = parse_human_move(text, n)
            except ValueError as exc:
                print("输入错误: %s" % exc)
                continue
            if parsed == "quit":
                print("已退出，本局未完成")
                return None
            if parsed == "help":
                print("row col: 0-based，例如 3 3；D4: Go 坐标；pass；quit")
                continue
            if parsed is None:
                continue
            move = parsed
        else:
            move = ai_move(mcts, game, n, args.sims, args.temperature)
            if move is None:
                print("AI 找不到合法着法，按 pass 处理")
                move = PASS_MOVE
            print("%s方 AI: %s" % (player_name(player), coord_text(move[0], move[1], n)))

        if not game.is_legal_move(move[0], move[1]):
            if human:
                print("非法着法，请重试")
                continue
            print("AI 返回非法着法，退出")
            return None

        color = "B" if player == 1 else "W"
        moves.append((color, move))
        game.make_move(move[0], move[1])
        mcts.update_root(game, move)
        if not args.no_board:
            print_board(game)

    game.terminal_analysis()
    print()
    print("终局: %s | 手数 %d" % (result_text(game), game.move_count))
    return game, moves


def main():
    args = parse_args()
    if args.games <= 0:
        raise SystemExit("--games 必须 > 0")
    if args.sims <= 0:
        raise SystemExit("--sims 必须 > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warning] CUDA 不可用，改用 CPU", flush=True)
        args.device = "cpu"
    try:
        args.hole_mask, args.start_stones = shapes.spec_from_name(
            args.board_size, args.shape)
    except (ValueError, OSError) as exc:
        raise SystemExit("--shape 非法: %s" % exc)
    random.seed(args.seed)

    model = PolicyValueNet.load_model(args.model, device=args.device)
    evaluator = GoEvaluator(model, device=args.device, max_batch=args.mcts_batch)
    area = shapes.playable_area(args.hole_mask)
    cap = max(MCTS_CAP, int(args.sims * area * 1.2))
    mcts = MCTS(
        evaluator=evaluator, num_simulations=args.sims,
        batch_size=args.mcts_batch, c_puct=args.c_puct,
        temperature=args.temperature, alpha=args.alpha,
        dirichlet_alpha=args.dir_alpha, dirichlet_epsilon=args.dir_eps,
        board_size=args.board_size, virtual_loss=args.vloss, cap=cap)

    for gi in range(1, args.games + 1):
        if args.games > 1:
            print("\n##### 第 %d/%d 局 #####" % (gi, args.games))
        result = one_game(args, mcts)
        if result is None:
            break
        game, moves = result
        if args.sgf:
            path = args.sgf
            if args.games > 1:
                stem, ext = os.path.splitext(path)
                path = "%s_%d%s" % (stem, gi, ext or ".sgf")
            write_sgf(path, game, moves, args.board_size, args.komi,
                      args.hole_mask, args.start_stones)
            print("SGF 已保存: %s" % path)


if __name__ == "__main__":
    main()
