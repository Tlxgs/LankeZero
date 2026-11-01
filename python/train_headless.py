"""无界面训练入口：自对弈 + 训练，或只训练已有数据。

用法示例：
  python train_headless.py --board-size 13 --games 100 --sims 300
  python train_headless.py --board-size 13 --train-only --train-batches 1000
  python train_headless.py --help

默认值与 train_gui 一致；模型始终是唯一 model.pt，训练后原地原子更新。
"""
import argparse
import os
import random
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch

from hyperparams import (BOARD_SIZES, NUM_SIMULATIONS, BATCH_SIZE, BATCH_SIZE_MCTS, C_PUCT,
                         ALPHA, VIRTUAL_LOSS, DIRICHLET_ALPHA, DIRICHLET_EPSILON,
                         CANDIDATE_MASS, MIN_CANDIDATES, LEARNING_RATE,
                         WEIGHT_DECAY, OPTIMIZER_NAME, MOMENTUM, TRAIN_GAMES,
                         TRAIN_STEPS_PER_GAME, SAVE_INTERVAL, TRAINING_DATA_MAX,
                         MODEL_PATH, LOG_DIR, PROJECT_ROOT, EXPLORATION_MODE,
                         validate_hollow, playable_area)
from model import PolicyValueNet, count_parameters
from train import SelfPlayTrainer
from mcts import MCTS
from selfplay import play_one_game


def default_data_dir(board_size):
    n = int(board_size)
    return "data" if n == 19 else "data%d" % n


def build_parser():
    epilog = """\
示例：
  # 13 路自对弈 100 局，使用 data13/ 和 model.pt
  python train_headless.py --board-size 13 --games 100 --sims 300

  # 13 路镂空 7×7 自对弈；混合数据仍写同一个 data13/
  python train_headless.py --board-size 13 --hollow 7 --games 100

  # 只训练已有数据，不生成新对局
  python train_headless.py --board-size 13 --train-only --train-batches 1000

  # 先试跑 1 个 batch，不写 model.pt
  python train_headless.py --board-size 13 --train-only --train-batches 1 --no-save-model

  # 新建一个 b6c96 小模型（不会覆盖已有 model.pt），再训练
  python train_headless.py --model model_b6c96.pt --init-model \
      --channels 96 --blocks 6 --head-channels 64 --board-size 13
"""
    p = argparse.ArgumentParser(
        description="Headless Go 训练：自对弈 + 训练，或只训练已有数据。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog)
    # 对局 / 搜索
    g = p.add_argument_group("对局与搜索")
    g.add_argument("--board-size", type=int, choices=BOARD_SIZES, default=13)
    g.add_argument("--hollow", type=int, default=0,
                   help="中心镂空边长（0=实心；3/5/7/9...，至少留两圈）")
    g.add_argument("--games", type=int, default=TRAIN_GAMES,
                   help="自对弈完成的局数")
    g.add_argument("--sims", type=int, default=NUM_SIMULATIONS,
                   help="每手 MCTS 模拟数")
    g.add_argument("--mcts-batch", type=int, default=BATCH_SIZE_MCTS)
    g.add_argument("--c-puct", type=float, default=C_PUCT)
    g.add_argument("--temperature", type=float, default=0.5)
    g.add_argument("--alpha", type=float, default=ALPHA,
                   help="胜率/目差混合系数，0=纯胜率，1=纯目差")
    g.add_argument("--vloss", type=int, default=VIRTUAL_LOSS)
    g.add_argument("--dir-alpha", type=float, default=DIRICHLET_ALPHA)
    g.add_argument("--dir-eps", type=float, default=DIRICHLET_EPSILON)
    g.add_argument("--candidate-mass", type=float, default=CANDIDATE_MASS)
    g.add_argument("--min-candidates", type=int, default=MIN_CANDIDATES)
    g.add_argument("--exploration", dest="exploration", action="store_true",
                   default=EXPLORATION_MODE,
                   help="四角随机开局（hollow>0 时自动关闭）")
    g.add_argument("--no-exploration", dest="exploration", action="store_false")

    # 数据 / 模型
    g = p.add_argument_group("数据与模型")
    g.add_argument("--data-dir", default=None,
                   help="默认按棋盘尺寸取 data<N>/（19 路为 data/）")
    g.add_argument("--model", default=MODEL_PATH, help="唯一权重路径")
    g.add_argument("--init-model", action="store_true",
                   help="model 不存在时，按 --channels/--blocks 新建")
    g.add_argument("--channels", type=int, default=None, help="新建模型的主干通道数")
    g.add_argument("--blocks", type=int, default=None, help="新建模型的残差块数")
    g.add_argument("--head-channels", type=int, default=64, help="新建模型的头部通道数")
    g.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    g.add_argument("--load-data", dest="load_data", action="store_true", default=True)
    g.add_argument("--no-load-data", dest="load_data", action="store_false")
    g.add_argument("--save-data", dest="save_data", action="store_true", default=True)
    g.add_argument("--no-save-data", dest="save_data", action="store_false")
    g.add_argument("--load-mode", choices=("newest", "random"), default="newest")
    g.add_argument("--max-data", type=int, default=TRAINING_DATA_MAX,
                   help="内存样本上限；1000 局 19 路约 20 万条")
    g.add_argument("--no-save-model", action="store_true",
                   help="只试跑，不写 model.pt（也不写 periodic 保存）")
    g.add_argument("--tensorrt", action="store_true",
                   help="尝试使用已缓存的 TensorRT REFIT 引擎（不匹配自动回退）")
    g.add_argument("--trt-engine-dir", default=None,
                   help="TensorRT 引擎缓存目录（默认 trt_engines/）")

    # 训练
    g = p.add_argument_group("训练")
    g.add_argument("--batch", type=int, default=BATCH_SIZE)
    g.add_argument("--steps", type=int, default=TRAIN_STEPS_PER_GAME,
                   help="每局自对弈后训练多少个 batch")
    g.add_argument("--save-interval", type=int, default=SAVE_INTERVAL,
                   help="每多少局保存一次 model.pt；0=不周期保存")
    g.add_argument("--lr", type=float, default=LEARNING_RATE)
    g.add_argument("--wd", type=float, default=WEIGHT_DECAY)
    g.add_argument("--optimizer", type=str.lower, default=OPTIMIZER_NAME.lower(),
                   choices=("sgd", "adam"))
    g.add_argument("--momentum", type=float, default=MOMENTUM)
    g.add_argument("--freeze", default=None,
                   help="冻结规格，例如 '0,own' 或 'only:gpool'")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--log-file", default=None,
                   help="默认 logs/headless_<时间>.log")

    # 只训练
    g = p.add_argument_group("只训练已有数据")
    g.add_argument("--train-only", action="store_true",
                   help="不生成新对局，只训练 data-dir 里的数据")
    g.add_argument("--train-batches", type=int, default=1000,
                   help="--train-only 时的训练 batch 数")
    g.add_argument("--save-batches", type=int, default=100,
                   help="--train-only 时每多少 batch 保存一次")
    return p


class Logger:
    def __init__(self, path):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.path = path
        self.f = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, message):
        line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), message)
        print(line, flush=True)
        self.f.write(line + "\n")

    def close(self):
        self.f.close()


def _validate(args):
    if args.board_size not in BOARD_SIZES:
        raise SystemExit("--board-size 只能是 %s" % (BOARD_SIZES,))
    try:
        args.hollow = validate_hollow(args.board_size, args.hollow)
    except ValueError as exc:
        raise SystemExit("--hollow 非法: %s" % exc)
    if args.batch <= 0:
        raise SystemExit("--batch 必须 > 0")
    if args.max_data < args.batch:
        raise SystemExit("--max-data 不能小于 --batch")
    if args.sims <= 0:
        raise SystemExit("--sims 必须 > 0")
    if args.mcts_batch <= 0:
        raise SystemExit("--mcts-batch 必须 > 0")
    if not (0.0 <= args.alpha <= 1.0):
        raise SystemExit("--alpha 必须在 [0, 1] 内")
    if not (0.0 <= args.candidate_mass <= 1.0):
        raise SystemExit("--candidate-mass 必须在 [0, 1] 内")
    if args.min_candidates <= 0:
        raise SystemExit("--min-candidates 必须 > 0")
    if args.save_interval < 0:
        raise SystemExit("--save-interval 不能为负数")
    if args.train_only:
        if args.train_batches <= 0:
            raise SystemExit("--train-batches 必须 > 0")
        if args.save_batches < 0:
            raise SystemExit("--save-batches 不能为负数")
    else:
        if args.games <= 0:
            raise SystemExit("--games 必须 > 0")
        if args.steps < 0:
            raise SystemExit("--steps 不能为负数")
    if args.init_model:
        if args.channels is None or args.blocks is None:
            raise SystemExit("--init-model 需要同时给 --channels 和 --blocks")
        if args.channels <= 0 or args.blocks <= 0 or args.head_channels <= 0:
            raise SystemExit("--channels/--blocks/--head-channels 必须 > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warning] CUDA 不可用，改用 CPU", flush=True)
        args.device = "cpu"


def _save_model(trainer, args, logger, tag):
    if args.no_save_model:
        return None
    path = trainer.save_model(tag)
    logger("已保存唯一权重: %s" % path)
    return path


def main():
    args = build_parser().parse_args()
    _validate(args)

    if args.data_dir is None:
        args.data_dir = os.path.join(PROJECT_ROOT, default_data_dir(args.board_size))
    if args.log_file is None:
        args.log_file = os.path.join(
            LOG_DIR, "headless_%s.log" % datetime.now().strftime("%Y%m%d_%H%M%S"))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    logger = Logger(args.log_file)
    trainer = None
    failed = False
    try:
        logger("=" * 60)
        logger("headless 训练开始")
        logger("board=%d hollow=%d area=%d games=%d sims=%d temp=%.2f alpha=%.2f dir=%.3f/%.2f "
               "cand=%.3f/%d"
               % (args.board_size, args.hollow,
                  playable_area(args.board_size, args.hollow),
                  args.games, args.sims, args.temperature,
                  args.alpha, args.dir_alpha, args.dir_eps,
                  args.candidate_mass, args.min_candidates))
        if args.hollow and args.exploration and not args.train_only:
            logger("提示: hollow>0 时四角固定开局自动关闭，避免开局落进洞里")
        logger("model=%s data=%s device=%s batch=%d steps=%d lr=%g"
               % (args.model, args.data_dir, args.device, args.batch,
                  args.steps, args.lr))

        if not os.path.exists(args.model):
            if not args.init_model:
                raise SystemExit(
                    "模型不存在: %s\n"
                    "首次训练请加 --init-model --channels 96 --blocks 6" % args.model)
            os.makedirs(os.path.dirname(os.path.abspath(args.model)), exist_ok=True)
            model = PolicyValueNet(
                board_size=args.board_size, channels=args.channels,
                num_res_blocks=args.blocks, head_channels=args.head_channels)
            model.save_model(args.model)
            logger("新建模型: channels=%d blocks=%d head=%d | 参数 %.3f M"
                   % (args.channels, args.blocks, args.head_channels,
                      count_parameters(model) / 1e6))
        elif args.init_model:
            raise SystemExit("--init-model 目标已存在，拒绝覆盖: %s" % args.model)

        trainer = SelfPlayTrainer(
            model_path=args.model, board_size=args.board_size,
            device=args.device, num_simulations=args.sims,
            data_dir=args.data_dir, c_puct=args.c_puct,
            learning_rate=args.lr, weight_decay=args.wd,
            optimizer_name=args.optimizer, momentum=args.momentum,
            freeze_parts=args.freeze, data_max=args.max_data,
            use_tensorrt=args.tensorrt, trt_engine_dir=args.trt_engine_dir,
            log=logger)
        logger("模型: %s | 参数 %.3f M"
               % (trainer.model_path,
                  sum(p.numel() for p in trainer.model.parameters()) / 1e6))

        if args.load_data:
            trainer.load_training_data(mode=args.load_mode)
        db = trainer.data_buffer
        if getattr(db, "last_skipped", 0):
            logger("数据跳过 %d 个旧版/损坏文件" % db.last_skipped)
        logger("数据缓冲 %d 条" % len(db))

        if args.train_only:
            _train_only(trainer, args, logger)
        else:
            _selfplay(trainer, args, logger)
    except KeyboardInterrupt:
        logger("收到 Ctrl+C，正在收尾")
    except Exception as exc:
        failed = True
        logger("训练异常: %s: %s" % (type(exc).__name__, exc))
        logger(traceback.format_exc())
    finally:
        if trainer is not None:
            try:
                if failed:
                    logger("异常退出：不自动覆盖 model.pt")
                elif args.no_save_model:
                    trainer.flush_grad_accum()
                    logger("--no-save-model：未写 model.pt")
                else:
                    trainer.flush_grad_accum()
                    _save_model(trainer, args, logger, "headless")
            except Exception as exc:
                logger("保存模型失败: %s: %s" % (type(exc).__name__, exc))
        logger("训练结束: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.close()


def _make_mcts(trainer, args):
    return MCTS(
        evaluator=trainer.evaluator, num_simulations=args.sims,
        batch_size=args.mcts_batch, c_puct=args.c_puct,
        temperature=args.temperature, alpha=args.alpha,
        dirichlet_alpha=args.dir_alpha, dirichlet_epsilon=args.dir_eps,
        candidate_mass=args.candidate_mass, min_candidates=args.min_candidates,
        board_size=args.board_size, virtual_loss=args.vloss)


def _selfplay(trainer, args, logger):
    mcts = _make_mcts(trainer, args)
    games = 0
    attempts = 0
    max_attempts = max(args.games * 3, args.games + 10)
    t_start = time.time()
    last_report = [0]

    def on_move(played, n_samples):
        if played - last_report[0] >= 50:
            last_report[0] = played
            logger("  搜索 %d 手（样本 %d）" % (played, n_samples))

    while games < args.games and attempts < max_attempts:
        attempts += 1
        t0 = time.time()
        result = play_one_game(
            mcts=mcts, board_size=args.board_size, num_simulations=args.sims,
            device=args.device, c_puct=args.c_puct,
            temperature=args.temperature, exploration=args.exploration,
            on_move=on_move, hollow=args.hollow)
        if result["payload"] is None or not result["complete"]:
            logger("自对弈未正常结束，丢弃本局（%d/%d）" % (attempts, max_attempts))
            continue
        saved = None
        if args.save_data:
            saved = trainer.save_training_data(result["payload"])
        trainer.data_buffer.add_game(result, source=saved)
        trainer.game_count += 1
        games += 1

        losses = []
        for _ in range(args.steps):
            loss = trainer.train_step(args.batch)
            if loss[0] is None:
                break
            trainer.train_count += 1
            losses.append(loss)
        if args.steps == 0:
            loss_text = "跳过训练（--steps 0）"
        elif losses:
            mean = tuple(float(np.mean([x[i] for x in losses]))
                         for i in range(trainer.LOSS_N))
            loss_text = "平均 policy=%.4f own=%.4f win=%.4f total=%.4f" % (
                mean[0], mean[1], mean[3], mean[4])
        else:
            loss_text = "数据不足，本局未训练"
        logger("局 %d/%d | %d 手 | %.1fs | 缓冲 %d | %s"
               % (games, args.games, len(result["moves"]), time.time() - t0,
                  len(trainer.data_buffer), loss_text))
        if args.save_interval and games % args.save_interval == 0:
            _save_model(trainer, args, logger, "g%06d" % games)

    if games < args.games:
        raise RuntimeError("只完成 %d/%d 局（尝试 %d 次）"
                           % (games, args.games, attempts))
    logger("自对弈结束: %d 局，%.1fs" % (games, time.time() - t_start))


def _train_only(trainer, args, logger):
    if len(trainer.data_buffer) < args.batch:
        raise RuntimeError("数据不足: %d < %d"
                           % (len(trainer.data_buffer), args.batch))
    done = 0
    acc = [0.0] * trainer.LOSS_N
    acc_n = 0
    total_acc = [0.0] * trainer.LOSS_N
    total_n = 0
    for i in range(args.train_batches):
        loss = trainer.train_step(args.batch)
        if loss[0] is None:
            break
        trainer.train_count += 1
        done += 1
        for k, v in enumerate(loss):
            acc[k] += v
            total_acc[k] += v
        acc_n += 1
        total_n += 1
        if acc_n >= 10:
            mean = tuple(a / acc_n for a in acc)
            logger("batch %d/%d | 平均 policy=%.4f own=%.4f win=%.4f total=%.4f"
                   % (i + 1, args.train_batches, mean[0], mean[1],
                      mean[3], mean[4]))
            acc = [0.0] * trainer.LOSS_N
            acc_n = 0
        if args.save_batches and (i + 1) % args.save_batches == 0:
            _save_model(trainer, args, logger, "b%06d" % (i + 1))
    if acc_n:
        mean = tuple(a / acc_n for a in acc)
        logger("batch %d/%d | 平均 policy=%.4f own=%.4f win=%.4f total=%.4f"
               % (done, args.train_batches, mean[0], mean[1],
                  mean[3], mean[4]))
    if total_n:
        mean = tuple(a / total_n for a in total_acc)
        logger("只训练结束: %d batch | 全局平均 policy=%.4f own=%.4f win=%.4f total=%.4f"
               % (done, mean[0], mean[1], mean[3], mean[4]))
    else:
        logger("只训练结束: 0 batch")


if __name__ == "__main__":
    main()
