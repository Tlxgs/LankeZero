"""本机单进程自对弈：MCTS + GoEvaluator，直接产出紧凑训练样本。

不再有 multiprocessing worker / ONNX 模型热更新：训练线程和搜索共用同一个模型，
每局结束后才反向传播，因此一局内的权重不会变化。
"""
import random

import numpy as np

from game import GoGame, PASS_MOVE
from hyperparams import (BOARD_SIZE, NUM_SIMULATIONS, C_PUCT, TEMPERATURE,
                         EXPLORATION_MODE, TOP_P, KOMI, board_temperature_zero, MODEL_PATH,
                         temperature_schedule)
from mcts import MCTS
from replay import state_to_sample, pack_game


_AUTO_TEMPERATURE_ZERO = object()


def _top_p_keep(items, top_p):
    """按概率降序取累计概率达到 top_p 的最小点集（至少保留最高的一个点）。"""
    cum = 0.0
    kept = []
    for move, prob in items:
        kept.append((move, prob))
        cum += prob
        if cum >= top_p:
            break
    return kept


def _exploration_opening(board_size):
    """探索模式开局：四角各一子，角上随机星位（4-4）/ 小目 / 三三。

    星位是固定的 4-4（0-based 3,3），不随棋盘尺寸漂移；9 路也用它。
    三三 = (2,2)，小目 = (2,3)/(3,2)。
    """
    n = board_size - 1
    s = 3
    corners = {
        "TL": [(s, s), (s - 1, s), (s, s - 1), (s - 1, s - 1)],
        "TR": [(s, n - s), (s - 1, n - s), (s, n - s + 1), (s - 1, n - s + 1)],
        "BL": [(n - s, s), (n - s + 1, s), (n - s, s - 1), (n - s + 1, s - 1)],
        "BR": [(n - s, n - s), (n - s + 1, n - s), (n - s, n - s + 1), (n - s + 1, n - s + 1)],
    }
    if random.random() < 0.5:
        side = random.choice(("top", "bottom", "left", "right"))
        if side == "top":
            black = ["TL", "TR"]
        elif side == "bottom":
            black = ["BL", "BR"]
        elif side == "left":
            black = ["TL", "BL"]
        else:
            black = ["TR", "BR"]
    else:
        black = ["TL", "BR"] if random.random() < 0.5 else ["TR", "BL"]
    placements = []
    for k in black:
        r, c = random.choice(corners[k])
        placements.append((r, c, 1))
    for k in corners:
        if k not in black:
            r, c = random.choice(corners[k])
            placements.append((r, c, -1))
    random.shuffle(placements)
    placements.sort(key=lambda p: p[2], reverse=True)
    return placements


def _dense_target(policy_probs, board_size):
    """访问计数 -> top-p 稀疏目标（稠密 362 维，给 state_to_sample 再压成稀疏）。"""
    kept = _top_p_keep(sorted(policy_probs.items(), key=lambda x: x[1], reverse=True), TOP_P)
    mass = sum(p for _, p in kept)
    out = np.zeros(board_size * board_size + 1, dtype=np.float32)
    if mass <= 0.0:
        return out
    for move, prob in kept:
        idx = board_size * board_size if move == PASS_MOVE else move[0] * board_size + move[1]
        out[idx] = prob / mass
    return out


def play_one_game(mcts=None, evaluator=None, board_size=None,
                  num_simulations=NUM_SIMULATIONS, device="cuda",
                  c_puct=C_PUCT, temperature=TEMPERATURE,
                  exploration=EXPLORATION_MODE, model_path=None,
                  temperature_zero_after=_AUTO_TEMPERATURE_ZERO,
                  max_moves=None, on_move=None, hollow=0):
    """下一局，返回训练样本、回放着法和终局标签。

    mcts 可跨局复用；每局开始只 reset 根，不重新加载模型。
    四角随机开局的 4 手只写入回放 moves，不生成训练样本；第一个样本是
    四角已占后的局面。max_moves 仅用于基准/冒烟测试：截断局不产生终局
    ownership，也不作为训练数据。

    temperature_zero_after 默认按可落子面积取 board_temperature_zero(n, hollow)：
    实心 7/9/11/13/15/17/19/21/23 路分别约为 11/18/27/37/50/64/79/97/116 手；
    显式传 None 表示全程不切贪心（评估对局用）。
    """
    if board_size is None:
        if mcts is not None:
            board_size = mcts.board_size
        elif evaluator is not None:
            board_size = evaluator.n
        else:
            board_size = BOARD_SIZE
    board_size = int(board_size)
    if temperature_zero_after is _AUTO_TEMPERATURE_ZERO:
        temperature_zero_after = board_temperature_zero(board_size, hollow)
    if mcts is not None and mcts.board_size != board_size:
        raise ValueError("mcts.board_size=%d != board_size=%d" % (mcts.board_size, board_size))
    if mcts is None:
        if evaluator is None:
            from model import PolicyValueNet
            from evaluator import GoEvaluator
            mp = model_path or MODEL_PATH
            model = PolicyValueNet.load_model(mp, device=device)
            evaluator = GoEvaluator(model, device=device, max_batch=16)
        mcts = MCTS(evaluator=evaluator, num_simulations=num_simulations,
                    c_puct=c_puct, temperature=temperature, device=device,
                    board_size=board_size)
    game = GoGame(board_size, komi=KOMI, hollow=hollow)
    mcts.reset()
    samples = []
    moves = []
    played = 0
    # 四角固定开局只对实心棋盘生效；镂空盘的四角星位很可能落在洞里。
    if exploration and game.hollow == 0:
        for r, c, pl in _exploration_opening(board_size):
            game.make_move(r, c)
            moves.append((r, c, pl))
    truncated = max_moves is not None
    aborted = False
    while not game.game_over and not (truncated and played >= max_moves):
        temp = temperature_schedule(temperature, played, temperature_zero_after)
        policy_probs, sample_probs = mcts.get_move_probs(game, temp, num_simulations)
        legal_moves, legal_mask = game.get_legal_moves_and_mask()
        state = game.get_canonical_state(legal_mask=legal_mask)
        target = _dense_target(policy_probs, board_size)
        samples.append(state_to_sample(state, target, game.current_player,
                                       game.last_move, game.ko_point))
        if not sample_probs:
            aborted = True
            break
        top = _top_p_keep(sorted(sample_probs.items(), key=lambda x: x[1], reverse=True), TOP_P)
        top_moves = [m for m, _ in top]
        top_probs = np.array([p for _, p in top], dtype=np.float32)
        top_probs /= top_probs.sum()
        selected = random.choices(top_moves, weights=top_probs, k=1)[0]
        player_before = game.current_player
        if selected == PASS_MOVE:
            game.make_move(-1, -1)
            moves.append((-1, -1, player_before))
        else:
            game.make_move(selected[0], selected[1])
            moves.append((selected[0], selected[1], player_before))
        mcts.update_root(game, selected)
        played += 1
        if on_move is not None:
            on_move(played, len(samples))

    if truncated or aborted:
        winner = 0
        score_diff = 0.0
        ownership = np.zeros((board_size, board_size), dtype=np.int8)
    else:
        ownership = game.terminal_analysis().astype(np.int8)
        winner = int(game.winner or 0)
        score_diff = float(game.final_points)
    payload = None
    if samples:
        payload = pack_game(samples, moves, winner, score_diff, ownership,
                            board_size, hollow=hollow)
    return {
        "samples": samples,
        "payload": payload,
        "moves": moves,
        "winner": winner,
        "score_diff": score_diff,
        "ownership": ownership,
        # 内存缓冲直接 add_game(result)，必须带上 hollow/komi；
        # 否则镂空局在本次训练缓冲里会被当成实心盘编码。
        "hollow": int(hollow),
        "komi": float(KOMI),
        "complete": not (truncated or aborted),
    }