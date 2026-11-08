"""本机单进程自对弈：MCTS + GoEvaluator，直接产出紧凑训练样本。

不再有 multiprocessing worker / ONNX 模型热更新：训练线程和搜索共用同一个模型，
每局结束后才反向传播，因此一局内的权重不会变化。
"""
import random

import numpy as np

from game import GoGame, PASS_MOVE
import shapes
from hyperparams import (BOARD_SIZE, NUM_SIMULATIONS, C_PUCT, TEMPERATURE,
                         EXPLORATION_MODE, TOP_P, board_temperature_zero_for_area,
                         MODEL_PATH, temperature_schedule)
from mcts import MCTS
from replay import state_to_sample, pack_game
from rules_adapter import GO_RULES


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
    """让子开局（探索模式）：棋盘 >= 13 路时，黑先在 1~4 个角星位随机摆子，白先走。

    13 路以下不摆：地盘小，让 1~4 子把对局直接压成让子方必胜，数据没价值。
    角星位用固定 4-4（0-based 3,3），不随尺寸漂移（13 路 = (3,3)/(3,9)/(9,3)/(9,9)）。
    返回的着法全是黑子，由 play_one_game 交给 `GoGame.setup_handicap` 一次性摆上，
    这些子只进回放、不生成训练样本；让子局贴 0 目。
    """
    n = int(board_size)
    if n < 13:
        return []
    s = 3
    corners = [(s, s), (s, n - 1 - s), (n - 1 - s, s), (n - 1 - s, n - 1 - s)]
    if len(set(corners)) != 4:
        return []
    k = random.randint(1, 4)
    return [(r, c, 1) for r, c in random.sample(corners, k)]


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
                  max_moves=None, on_move=None, hole_mask=None, rules=None,
                  komi=None, start_stones=None):
    """下一局，返回训练样本、回放着法和终局标签。

    mcts 可跨局复用；每局开始只 reset 根，不重新加载模型。
    让子开局（13 路以上，1~4 子）只写入回放 moves，不生成训练样本；第一个样本是
    让子已摆好的局面。max_moves 仅用于基准/冒烟测试：截断局不产生终局
    ownership，也不作为训练数据。

    hole_mask 是自定义棋盘形状（(n,n) bool，True=洞）；给 None 时用居中方块
    镂空 hollow。让子开局只在**整盘可下**时启用 —— 心形/圆环盘的角上
    大概率是洞。

    temperature_zero_after 默认按可落子面积取 board_temperature_zero：
    实心 7/9/11/13/15/17/19 路分别约为 11/18/27/37/50/64/79 手；
    显式传 None 表示全程不切贪心（评估对局用）。

    rules 是棋种适配对象（见 rules_adapter）；不传就是围棋。
    """
    rules = rules or GO_RULES
    if board_size is None:
        if mcts is not None:
            board_size = mcts.board_size
        elif evaluator is not None:
            board_size = evaluator.n
        else:
            board_size = BOARD_SIZE
    board_size = int(board_size)
    if hole_mask is not None:
        hole_mask = shapes.validate_mask(hole_mask, board_size)
    area = (shapes.playable_area(hole_mask) if hole_mask is not None
            else board_size * board_size)
    if temperature_zero_after is _AUTO_TEMPERATURE_ZERO:
        temperature_zero_after = board_temperature_zero_for_area(area)
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
                    board_size=board_size, rules=rules)
    # ---- 让子开局（探索模式）：>=13 路时黑先摆 1~4 个角星位，白先走，贴 0 目 ----
    opening = []
    if exploration and start_stones is None:
        solid = hole_mask is None or not hole_mask.any()
        if solid:
            opening = rules.exploration_opening(board_size)
    handicap = bool(opening) and getattr(rules, "handicap_opening", False)
    if handicap:
        komi_used = 0.0                      # 让子棋贴 0 目
    else:
        komi_used = rules.komi if komi is None else float(komi)

    game = rules.new_game(board_size, hole_mask, komi_used,
                          start_board=start_stones)
    mcts.reset()
    samples = []
    moves = []
    if opening:
        if handicap:
            game.setup_handicap([(r, c) for r, c, _p in opening])
        else:
            for r, c, _p in opening:
                game.make_move(r, c)
        moves.extend(opening)
    limit = None if max_moves is None else int(max_moves)
    aborted = False
    while not game.game_over and not (limit is not None and game.move_count >= limit):
        # 温度调度按**实际手数**（含让子）走：之前用的是本循环内的
        # 计数，导致实心盘的贪心切换点比 board_temperature_zero 算出来的晚 4 手。
        temp = temperature_schedule(temperature, game.move_count, temperature_zero_after)
        policy_probs, sample_probs = mcts.get_move_probs(game, temp, num_simulations)
        if not sample_probs:
            # 根上没有可落点（极端局面）：整局作废，不必再往里塞样本。
            aborted = True
            break
        legal_moves, legal_mask = game.get_legal_moves_and_mask()
        state = game.get_canonical_state(legal_mask=legal_mask)
        target = _dense_target(policy_probs, board_size)
        # 累计提子按行棋方视角存（己方/对方），不存绝对的黑色/白色
        p_now = game.current_player
        cap_own, cap_opp = rules.captures_of(game, p_now)
        samples.append(rules.state_to_sample(
            state, target, p_now, game.last_move,
            cap_own=cap_own, cap_opp=cap_opp))
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
        if on_move is not None:
            on_move(game.move_count, len(samples))

    # 只有「没用完手数上限、却被手数截停」才算截断局。自然终局的局即使传了
    # max_moves 也必须当完整局结算 —— 之前用 `truncated = max_moves is not None`
    # 一刀切，会把正常下完的基准局判成无效（winner=0、ownership 全 0）。
    truncated = (limit is not None and not game.game_over
                 and game.move_count >= limit)
    if truncated or aborted:
        winner = 0
        score_diff = 0.0
        ownership = np.zeros((board_size, board_size), dtype=np.int8)
    else:
        own = rules.terminal_analysis(game)
        ownership = (np.zeros((board_size, board_size), dtype=np.int8)
                     if own is None else np.asarray(own, dtype=np.int8))
        winner = int(game.winner or 0)
        score_diff = float(rules.score_of(game))
    payload = None
    if samples:
        payload = pack_game(samples, moves, winner, score_diff, ownership,
                            board_size, komi=game.komi, own_mask=rules.own_mask,
                            hole=hole_mask, start_stones=start_stones)
    return {
        "samples": samples,
        "payload": payload,
        "moves": moves,
        "winner": winner,
        "score_diff": score_diff,
        "ownership": ownership,
        # 内存缓冲直接 add_game(result)，必须带上形状掩码和 komi；
        # 否则非实心盘在本次训练缓冲里会被当成整盘编码。
        "hole": hole_mask,
        "komi": float(game.komi),
        "start_stones": start_stones,
        "complete": not (truncated or aborted),
    }