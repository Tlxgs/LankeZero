"""本机单进程自对弈：MCTS + GoEvaluator，直接产出紧凑训练样本。

不再有 multiprocessing worker / ONNX 模型热更新：训练线程和搜索共用同一个模型，
每局结束后才反向传播，因此一局内的权重不会变化。
"""
import random

import numpy as np

from game import GoGame, PASS_MOVE
import shapes
from hyperparams import (BOARD_SIZE, NUM_SIMULATIONS, C_PUCT, TEMPERATURE,
                         TOP_P, board_temperature_zero_for_area,
                         MODEL_PATH, PCR_FAST_DIV, RESIGN_THRESHOLD,
                         temperature_schedule)
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


def _training_ownership(rules, game):
    """训练用的归属标签：优先用规则自己的 Benson 版本，没有就退回终局归属。"""
    fn = getattr(rules, "training_ownership", None)
    if fn is None:
        own = rules.terminal_analysis(game)
        return (None if own is None else np.asarray(own, dtype=np.int8)), None
    return fn(game)


def play_one_game(mcts=None, evaluator=None, board_size=None,
                  num_simulations=NUM_SIMULATIONS, device="cuda",
                  c_puct=C_PUCT, temperature=TEMPERATURE,
                  model_path=None,
                  temperature_zero_after=_AUTO_TEMPERATURE_ZERO,
                  max_moves=None, on_move=None, hole_mask=None, rules=None,
                  komi=None, start_stones=None, pcr_prob=0.0,
                  pcr_fast_div=PCR_FAST_DIV, resign_threshold=None):
    """下一局，返回训练样本、回放着法和终局标签。

    mcts 可跨局复用；每局开始只 reset 根，不重新加载模型。
    max_moves 仅用于基准/冒烟测试：截断局不产生终局 ownership，也不作为训练数据。

    hole_mask 是自定义棋盘形状（(n,n) bool，True=洞）；给 None 时用居中方块
    镂空 hollow。start_stones 是自定义开局的固定棋子（1 黑 / -1 白），算局面的一部分、
    不计手数，也不单独产生样本。

    temperature_zero_after 默认按可落子面积取 board_temperature_zero：
    实心 7/9/11/13/15/17/19 路分别约为 11/18/27/37/50/64/79 手；
    显式传 None 表示全程不切贪心（评估对局用）。

    rules 是棋种适配对象（见 rules_adapter）；不传就是围棋。

    pcr_prob>0 时开启快速走子：每手以 pcr_prob 的概率做全搜索（num_simulations 次），
    否则做快搜索（num_simulations // pcr_fast_div 次）；**只有全搜索那一手的局面进训练
    样本**，快搜索的走子照常落子、照常写进 moves，只是不产生样本。
    """
    rules = rules or GO_RULES
    if resign_threshold is None:
        resign_threshold = RESIGN_THRESHOLD
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
    komi_used = rules.komi if komi is None else float(komi)

    game = rules.new_game(board_size, hole_mask, komi_used,
                          start_board=start_stones)
    mcts.reset()
    samples = []
    sample_moves = []
    moves = []
    limit = None if max_moves is None else int(max_moves)
    aborted = False
    resigned = False
    resign_thr = float(resign_threshold or 0.0)
    pcr_prob = float(pcr_prob or 0.0)
    fast_div = max(1, int(pcr_fast_div or 1))
    while not game.game_over and not (limit is not None and game.move_count >= limit):
        # 温度调度按**实际手数**走（不是本循环内的计数），否则贪心切换点会偏晚。
        temp = temperature_schedule(temperature, game.move_count, temperature_zero_after)
        # 快速走子：这一手是「全搜索」（记样本）还是「快搜索」（只推进棋局）。
        full = True
        sims = num_simulations
        if pcr_prob > 0.0 and random.random() >= pcr_prob:
            full = False
            sims = max(1, int(num_simulations) // fast_div)
        policy_probs, sample_probs = mcts.get_move_probs(game, temp, sims)
        if not sample_probs:
            # 根上没有可落点（极端局面）：整局作废，不必再往里塞样本。
            aborted = True
            break
        # 认输：只看搜索根值（快搜索次数太少、噪声大，不参与判定）。
        # 判据故意不掺启发式目差 —— 纯数子把死子算活，输了也可能算出赢。
        if full and resign_thr > 0.0:
            if (mcts.root_value() + 1.0) * 0.5 < resign_thr:
                resigned = True
                break
        top = _top_p_keep(sorted(sample_probs.items(), key=lambda x: x[1], reverse=True), TOP_P)
        top_moves = [m for m, _ in top]
        top_probs = np.array([p for _, p in top], dtype=np.float32)
        top_probs /= top_probs.sum()
        selected = random.choices(top_moves, weights=top_probs, k=1)[0]
        if full:
            legal_moves, legal_mask = game.get_legal_moves_and_mask()
            state = game.get_canonical_state(legal_mask=legal_mask)
            target = _dense_target(policy_probs, board_size)
            # 累计提子按行棋方视角存（己方/对方），不存绝对的黑色/白色
            p_now = game.current_player
            cap_own, cap_opp = rules.captures_of(game, p_now)
            samples.append(rules.state_to_sample(
                state, target, p_now, game.last_move,
                cap_own=cap_own, cap_opp=cap_opp))
            # 与 samples 一一对应的实战着法：跳过局面的手不能出现在这里，
            # 否则 pack_game 的 move_codes 会整体错位。
            sample_moves.append(selected)
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
    if resigned:
        # 认输局没有终局：用 Benson 无条件活给**确定**的部分打归属标签，
        # 死活不明的点由 own_known=False 掩掉，不参与归属损失。
        # 只记胜负：目差不知道（纯数子把死子算活，可能算出反向结果），所以不写。
        own, own_known = _training_ownership(rules, game)
        if own is None:
            ownership = np.zeros((board_size, board_size), dtype=np.int8)
            own_mask = 0
            own_known = None
        else:
            ownership = np.asarray(own, dtype=np.int8)
            own_mask = int(rules.own_mask)
        winner = -int(game.current_player)
        score_diff = 0.0
    elif truncated or aborted:
        winner = 0
        score_diff = 0.0
        ownership = np.zeros((board_size, board_size), dtype=np.int8)
        own_mask = int(rules.own_mask)
        own_known = None
    else:
        # 自然终局（双 PASS / 下满）：归属标签同样用 Benson 只标确定的部分，
        # 避免「死子算活」把死子及周围标成对方地盘（那是结算口径，不是归属真值）。
        # **目数和胜负仍用原来的纯数子**（不清死子），只是归属标签被裁剪。
        own, own_known = _training_ownership(rules, game)
        if own is None:                     # 没有归属头的棋种
            ownership = np.zeros((board_size, board_size), dtype=np.int8)
            own_mask = 0
            own_known = None
        else:
            ownership = np.asarray(own, dtype=np.int8)
            own_mask = int(rules.own_mask)
        winner = int(game.winner or 0)
        score_diff = float(rules.score_of(game))
    payload = None
    if samples:
        # 用 sample_moves（与 samples 一一对应）而不是 moves：开了快速走子后
        # 大部分局面没有样本，moves 里的手数和样本数不再相等。
        payload = pack_game(samples, sample_moves, winner, score_diff, ownership,
                            board_size, komi=game.komi, own_mask=own_mask,
                            hole=hole_mask, start_stones=start_stones,
                            resigned=resigned, own_known=own_known)
    return {
        "samples": samples,
        "payload": payload,
        "moves": moves,
        "winner": winner,
        "score_diff": score_diff,
        "ownership": ownership,
        "resigned": resigned,
        # 内存缓冲直接 add_game(result)，必须带上形状掩码、komi、归属掩码；
        # 否则非实心盘在本次训练缓冲里会被当成整盘编码，
        # 认输局会丢掉 own_known（把死活不明的点当成"中立"硬训）。
        "hole": hole_mask,
        "komi": float(game.komi),
        "start_stones": start_stones,
        "own_mask": own_mask,
        "own_known": own_known,
        "complete": not (truncated or aborted),
    }