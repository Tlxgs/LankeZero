"""数据生成模块（纯生成，不训练，不 import torch / tkinter / model）。

设计目的（内存优化）：
- 数据生成 worker（mp.Process spawn 子进程）只 import 本模块 + game/mcts/
  hyperparams，不再连带加载 torch —— 每个 worker 可少占约 0.5~1GB 内存与
  CUDA 上下文。此前 play_one_game 位于 train.py（顶层 import torch），
  Windows spawn 子进程导入 train_gui 时会一并把 torch 带进来，白白浪费内存。
- eval.py 也改用本模块（评估同样只需要 ONNX 推理，不需要 torch）。
- 训练（torch）只发生在 GUI 主进程（SelfPlayTrainer），与数据生成彻底分离。
"""
import os
import gc
import random
import faulthandler

import numpy as np

from game import GoGame, PASS_MOVE
from mcts import MCTS
from hyperparams import (BOARD_SIZE, NUM_SIMULATIONS, C_PUCT, TEMPERATURE,
                         TEMPERATURE_DECAY, TEMPERATURE_ZERO_AFTER,
                         EXPLORATION_MODE, TOP_P, KOMI, ONNX_PATH,
                         temperature_schedule)


def worker_process(trainer_params, model_path, result_queue, stop_event, device='cuda'):
    """数据生成 worker：只生成对局数据，不训练、不评估。

    复用一个 MCTS（含 TRT 会话+引擎缓存）：每局仅重置搜索树；
    主进程保存新模型（model.onnx 变更）后重建会话，保证始终用最新模型下棋。
    """
    faulthandler.enable()   # 原生崩溃（TRT/ORT/numba）时打印 Python 栈到 stderr，便于定位
    onnx_path = ONNX_PATH
    if not os.path.exists(onnx_path):
        print(f"[Worker] ONNX文件不存在: {onnx_path}")
        return
    board_size = trainer_params['board_size']
    num_simulations = trainer_params['num_simulations']
    c_puct = trainer_params['c_puct']
    temperature = trainer_params['temperature']
    exploration = trainer_params['exploration_mode']
    alpha = trainer_params['alpha']
    np.random.seed()
    mcts = MCTS(c_puct=c_puct, num_simulations=num_simulations,
                temperature=temperature, alpha=alpha, onnx_path=onnx_path,
                device=device, board_size=board_size)
    last_mtime = os.path.getmtime(onnx_path)
    while not stop_event.is_set():
        mtime = os.path.getmtime(onnx_path)
        if mtime != last_mtime:      # 模型已更新 → 重建会话（引擎按新 ONNX 内容缓存/构建）
            # 先释放旧会话（TRT/ORT 原生内存）再建新的：
            # 搜索树视图存在引用环（_RootView↔MCTS），必须 gc.collect() 才真正释放，
            # 否则新旧引擎并存会造成瞬时双倍显存/内存（4GB 卡上易触发 OOM 崩溃）。
            mcts = None
            gc.collect()
            mcts = MCTS(c_puct=c_puct, num_simulations=num_simulations,
                        temperature=temperature, alpha=alpha, onnx_path=onnx_path,
                        device=device, board_size=board_size)
            last_mtime = mtime
        states, policies, players, moves, winner, score_diff, ownership = play_one_game(
            board_size=board_size,
            num_simulations=num_simulations,
            device=device,
            c_puct=c_puct,
            temperature=temperature,
            exploration=exploration,
            onnx_path=onnx_path,
            mcts=mcts
        )
        result_queue.put((states, policies, players, moves, winner, score_diff, ownership))
        # Eval 已独立到 eval.py（独立进程）：worker 只做自对弈，不再内嵌评估，
        # 避免评估阻塞训练 / 卡死拖垮 worker。Eval 数据由 eval.py 存数据目录供训练使用。


def _exploration_opening(board_size):
    """探索模式开局：四个角各放一子（黑2白2）。
    每个角从 {星位, 小目(两个之一), 三三} 中随机选一点；
    黑棋随机分布在棋盘同一侧(上/下/左/右)或对角(两条对角线之一)。
    返回打乱落子顺序的 [(r, c, player), ...]。
    """
    n = board_size - 1
    s = 3 if board_size >= 13 else 2          # 星位距边距离
    corners = {                               # 每角候选: 星位, 小目x2, 三三
        'TL': [(s, s), (s - 1, s), (s, s - 1), (s - 1, s - 1)],
        'TR': [(s, n - s), (s - 1, n - s), (s, n - s + 1), (s - 1, n - s + 1)],
        'BL': [(n - s, s), (n - s + 1, s), (n - s, s - 1), (n - s + 1, s - 1)],
        'BR': [(n - s, n - s), (n - s + 1, n - s), (n - s, n - s + 1), (n - s + 1, n - s + 1)],
    }
    if random.random() < 0.5:                 # 同侧：上/下/左/右 随机
        side = random.choice(('top', 'bottom', 'left', 'right'))
        if side == 'top':
            black = ['TL', 'TR']
        elif side == 'bottom':
            black = ['BL', 'BR']
        elif side == 'left':
            black = ['TL', 'BL']
        else:
            black = ['TR', 'BR']
    else:                                     # 对角：主/副对角线随机
        black = ['TL', 'BR'] if random.random() < 0.5 else ['TR', 'BL']
    placements = []
    for k in black:
        r, c = random.choice(corners[k])
        placements.append((r, c, 1))
    for k in corners:
        if k not in black:
            r, c = random.choice(corners[k])
            placements.append((r, c, -1))
    random.shuffle(placements)
    placements.sort(key=lambda p: p[2], reverse=True)  # 黑先落、白后落 → 黑棋先行
    return placements


def play_one_game(board_size=BOARD_SIZE, num_simulations=NUM_SIMULATIONS,
                  device='cuda', c_puct=C_PUCT, temperature=TEMPERATURE,
                  exploration=EXPLORATION_MODE, onnx_path=ONNX_PATH, model_b=None,
                  mcts=None, mcts_b=None, temperature_zero_after=TEMPERATURE_ZERO_AFTER):
    """生成一局游戏，返回 (states, policies, players, moves, winner, score_diff, ownership)
    model_b: 白方模型路径（None=黑白同模型）；用于 当前vs最佳 评估对局
    mcts/mcts_b: 可复用的 MCTS 实例（None=每局新建；mcts_b 对应 model_b）。
    temperature_zero_after: 训练温度调度参数（见 hyperparams.temperature_schedule）；
    第 zero_after 手起温度恒为0（贪心）。评估对局传 None 保持旧的纯衰减行为。
    调用方须保证 onnx_path 与 mcts 一致，并在模型文件更新后自行重建 mcts
    （内部按局面指纹自动重建搜索树，跨局复用安全）。评估可整轮复用同一对
    mcts/mcts_b，避免每局重建 TRT/ORT 会话（显著提速）。"""
    if mcts is None:
        mcts = MCTS(c_puct=c_puct, num_simulations=NUM_SIMULATIONS,
                    temperature=temperature, onnx_path=onnx_path, device=device,
                    board_size=board_size)
    if model_b is not None and mcts_b is None:
        mcts_b = MCTS(c_puct=c_puct, num_simulations=NUM_SIMULATIONS,
                      temperature=temperature, onnx_path=model_b, device=device,
                      board_size=board_size)
    komi = KOMI   # 固定贴目：value纯胜负需要一致基准（随机5~10会制造跨局标签矛盾）
    game = GoGame(board_size, komi=komi)
    states, policies, players = [], [], []
    moves = []
    move_count = 0
    mcts.root = None
    if exploration:                           # 探索模式：四角各放一子（星位/小目/三三）
        for r, c, pl in _exploration_opening(board_size):
            game.make_move(r, c)
            moves.append((r, c, pl))
    while not game.game_over:
        temp = temperature_schedule(temperature, move_count, TEMPERATURE_DECAY,
                                    temperature_zero_after)
        cur_mcts = mcts if game.current_player == 1 else (mcts_b or mcts)
        move_probs = cur_mcts.get_move_probs(game, temp, num_simulations)
        state = game.get_canonical_state()
        policy = np.zeros(board_size * board_size + 1, dtype=np.float32)
        for move, prob in move_probs.items():
            if move == PASS_MOVE:
                idx = board_size * board_size
            else:
                idx = move[0] * board_size + move[1]
            policy[idx] = prob
        states.append(state)
        policies.append(policy)
        players.append(game.current_player)


        if move_probs:
            items = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
            moves_list = [m for m, _ in items]
            probs = np.array([p for _, p in items], dtype=np.float32)
            cum_probs = np.cumsum(probs)
            cutoff_idx = np.searchsorted(cum_probs, TOP_P) + 1
            cutoff_idx = min(cutoff_idx, len(moves_list))
            top_moves = moves_list[:cutoff_idx]
            top_probs = probs[:cutoff_idx]
            top_probs = top_probs / np.sum(top_probs)   # 归一化
            selected = random.choices(top_moves, weights=top_probs, k=1)[0]
            player_before = game.current_player
            if selected == PASS_MOVE:
                game.make_move(-1, -1)
                moves.append((-1, -1, player_before))
            else:
                game.make_move(selected[0], selected[1])
                moves.append((selected[0], selected[1], player_before))
            cur_mcts.update_root(game, selected)
        move_count += 1

    # 终局：安全点捕获法判定死子 → 更新 final_points/winner，返回绝对归属图（整局共享标签）
    ownership = game.terminal_analysis()
    return states, policies, players, moves, game.winner, game.final_points, ownership
