"""本机围棋推理/搜索基准（只读 model.pt）。

用法：
  ..\\venv\\Scripts\\python.exe bench.py --sims 300 --batch 16 --moves 5
"""
import argparse
import time

import numpy as np
import torch

from model import PolicyValueNet
from evaluator import GoEvaluator
from game import GoGame
from mcts import MCTS
from hyperparams import BOARD_SIZES, INPUT_CHANNELS, CH_BOUNDARY, MODEL_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--board-size", type=int, default=19, choices=BOARD_SIZES)
    ap.add_argument("--moves", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()

    model = PolicyValueNet.load_model(args.model, device=args.device)
    ev = GoEvaluator(model, device=args.device, max_batch=args.batch,
                     graph=not args.no_graph)
    n = args.board_size
    x = np.zeros((args.batch, INPUT_CHANNELS, n, n), dtype=np.float32)
    x[:, 0, 3, 3] = 1.0
    x[:, 6] = 1.0
    x[:, CH_BOUNDARY, 0, :] = 1.0
    x[:, CH_BOUNDARY, n - 1, :] = 1.0
    x[:, CH_BOUNDARY, :, 0] = 1.0
    x[:, CH_BOUNDARY, :, n - 1] = 1.0
    mask = np.ones((args.batch, n * n + 1), dtype=np.float32)
    for _ in range(3):
        ev.evaluate(x, mask)
    t = time.time()
    for _ in range(args.repeats):
        ev.evaluate(x, mask)
    infer = (time.time() - t) / args.repeats
    print("inference batch=%d graph=%s: %.4f s/batch (%.1f pos/s)"
          % (args.batch, ev._graph_enabled, infer, args.batch / infer))

    mcts = MCTS(evaluator=ev, num_simulations=args.sims,
                batch_size=args.batch, board_size=n)
    game = GoGame(n)
    for _ in range(2):
        mcts.get_move_probs(game, temp=1.0, target_visits=min(30, args.sims))
    times = []
    for _ in range(args.moves):
        t = time.time()
        policy, sample = mcts.get_move_probs(game, temp=1.0, target_visits=args.sims)
        times.append(time.time() - t)
        mv = max(sample, key=sample.get)
        game.make_move(*mv) if mv != (-1, -1) else game.make_move(-1, -1)
        mcts.update_root(game, mv)
    print("mcts sims=%d batch=%d: %s"
          % (args.sims, args.batch, " ".join("%.3f" % v for v in times)))
    print("mcts mean: %.3f s/move" % (sum(times) / len(times)))


if __name__ == "__main__":
    main()