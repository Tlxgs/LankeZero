"""运行时最小回归测试：紧凑数据、严格加载、本地 MCTS、Play GUI 依赖的接口。

运行：..\\venv\\Scripts\\python.exe tests\\test_core.py
"""
import os
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import (GoGame, PASS_MOVE, HOLE, _legal_mask_for, _board_hash,
                  _hash_from_slots, _set_hash_slots, _group_has_liberty,
                  boundary_plane)
from hyperparams import (BOARD_SIZE, BOARD_SIZES, KOMI,
                         board_temperature_zero, temperature_schedule,
                         INPUT_CHANNELS, CH_BOUNDARY, MODEL_PATH,
                         PASS_TERMINAL_LOSS, validate_hollow, hollow_choices,
                         playable_area, board_rules)
from model import PolicyValueNet
from evaluator import GoEvaluator
from trt_engine import make_evaluator, architecture_hash, engine_paths
from replay import state_to_sample, pack_game, TrainingData, set_policy_onehot
from mcts import MCTS
from selfplay import play_one_game, _exploration_opening
from train import SelfPlayTrainer


class DummyEvaluator:
    """9 路均匀策略，用来测搜索树而不加载 3.5M 网络。"""
    def __init__(self, n=9):
        self.n = n
        self.device = "cpu"
        self.model = None

    def evaluate(self, states, legal_masks):
        b = len(states)
        total = self.n * self.n + 1
        p = np.zeros((b, total), dtype=np.float32)
        p[legal_masks > 0] = 1.0
        p /= p.sum(axis=1, keepdims=True)
        return p, np.zeros(b, dtype=np.float32), np.zeros(b, dtype=np.float32)


class TestReplay(unittest.TestCase):
    def test_compact_roundtrip(self):
        g = GoGame(BOARD_SIZE)
        g.make_move(3, 3)
        g.make_move(15, 15)
        g.make_move(3, 15)
        state = g.get_canonical_state()
        moves, _mask = g.get_legal_moves_and_mask()
        mv = moves[5]
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE + 1, dtype=np.float32)
        policy[mv[0] * BOARD_SIZE + mv[1]] = 1.0
        s = state_to_sample(state, policy, g.current_player, g.last_move)
        own = np.random.randint(-1, 2, size=(BOARD_SIZE, BOARD_SIZE)).astype(np.int8)
        payload = pack_game([s], moves[:1], -1, 3.5, own)

        d = tempfile.mkdtemp()
        td = TrainingData(max_size=100, data_dir=d)
        td.save_game(payload)
        td2 = TrainingData(max_size=100, data_dir=d)
        self.assertTrue(td2.load_all())
        states, policies, values, owns, kos, masks = td2.sample(1)
        np.testing.assert_allclose(states[0], state, atol=0.0)
        np.testing.assert_allclose(policies[0], policy, atol=0.0)
        self.assertEqual(values[0], 1.0)
        np.testing.assert_array_equal(owns[0], own * g.current_player)
        self.assertEqual(int(masks[0]), 1)


    def test_variable_komi_and_masked_ownership_roundtrip(self):
        """贴目按局读取；own_mask=0 的样本不参与归属损失。"""
        g = GoGame(9, komi=5.5)
        g.make_move(2, 2)
        state = g.get_canonical_state()
        policy = np.zeros(9 * 9 + 1, dtype=np.float32)
        policy[3 * 9 + 3] = 1.0
        s = state_to_sample(state, policy, g.current_player, g.last_move, g.ko_point)
        payload = pack_game([s], [(2, 2, 1)], 1, 0.0,
                            np.zeros((9, 9), dtype=np.int8),
                            board_size=9, komi=5.5, own_mask=0)
        d = tempfile.mkdtemp()
        td = TrainingData(max_size=10, data_dir=d, board_size=9)
        td.save_game(payload)
        td2 = TrainingData(max_size=10, data_dir=d, board_size=9)
        self.assertTrue(td2.load_all())
        states, policies, _values, _owns, _kos, masks = td2.sample(1)
        np.testing.assert_allclose(states[0], state, atol=0.0)
        self.assertAlmostEqual(float(states[0, 3, 0, 0]),
                               g.current_player * 5.5 / 14.0, places=6)
        self.assertEqual(int(masks[0]), 0)
        self.assertAlmostEqual(float(td2.data[0].komi), 5.5, places=6)

    def test_buffer_keeps_whole_new_game(self):
        g = GoGame(BOARD_SIZE)
        g.make_move(3, 3)
        state = g.get_canonical_state()
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE + 1, dtype=np.float32)
        policy[0] = 1.0
        s1 = state_to_sample(state, policy, g.current_player, g.last_move)
        g.make_move(15, 15)
        state2 = g.get_canonical_state()
        s2 = state_to_sample(state2, policy, g.current_player, g.last_move)
        td = TrainingData(max_size=3, data_dir=tempfile.mkdtemp())
        td.add_game({"samples": [s1, s1, s1], "ownership": np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8), "winner": 0})
        td.add_game({"samples": [s2, s2, s2], "ownership": np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8), "winner": 0})
        self.assertEqual(len(td), 3)
        np.testing.assert_array_equal(td.data[0].board, s2["board"])

    def test_replay_9x9_roundtrip(self):
        g = GoGame(9)
        g.make_move(2, 2)
        state = g.get_canonical_state()
        moves, _mask = g.get_legal_moves_and_mask()
        mv = moves[3]
        policy = np.zeros(9 * 9 + 1, dtype=np.float32)
        policy[mv[0] * 9 + mv[1]] = 1.0
        s = state_to_sample(state, policy, g.current_player, g.last_move)
        own = np.zeros((9, 9), dtype=np.int8)
        payload = pack_game([s], moves[:1], 0, 0.0, own, board_size=9)
        d = tempfile.mkdtemp()
        td = TrainingData(max_size=10, data_dir=d, board_size=9)
        td.save_game(payload)
        td2 = TrainingData(max_size=10, data_dir=d, board_size=9)
        self.assertTrue(td2.load_all())
        states, policies, _values, _owns, _kos, _masks = td2.sample(1)
        self.assertEqual(states.shape, (1, INPUT_CHANNELS, 9, 9))
        self.assertEqual(policies.shape, (1, 82))

    def test_ko_survives_data_roundtrip(self):
        g = GoGame(9)
        g.make_move(2, 2)
        state = g.get_canonical_state()
        policy = np.zeros(82, dtype=np.float32)
        policy[0] = 1.0
        s = state_to_sample(state, policy, g.current_player, g.last_move, (2, 3))
        payload = pack_game([s], [(2, 2, 1)], 0, 0.0,
                            np.zeros((9, 9), dtype=np.int8), board_size=9)
        d = tempfile.mkdtemp()
        td = TrainingData(max_size=10, data_dir=d, board_size=9)
        td.save_game(payload)
        td2 = TrainingData(max_size=10, data_dir=d, board_size=9)
        td2.load_all()
        _s, _p, _v, _o, kos, _m = td2.sample(1)
        self.assertEqual(int(kos[0]), 2 * 9 + 3)

    def test_augment_geometry_only_no_color_swap(self):
        """D4 只做几何变换：不交换 0/1，不翻转贴目，不取反 value/own。"""
        g = GoGame(9)
        g.make_move(2, 2)          # 黑
        g.make_move(6, 6)          # 白；现在轮黑
        state = g.get_canonical_state()
        policy = np.zeros(82, dtype=np.float32)
        policy[2 * 9 + 2] = 1.0
        s = state_to_sample(state, policy, g.current_player, g.last_move, g.ko_point)
        own = (np.arange(81).reshape(9, 9) % 3 - 1).astype(np.int8)
        td = TrainingData(max_size=10, data_dir=tempfile.mkdtemp(), board_size=9)
        td.add_game({"samples": [s], "ownership": own, "winner": int(g.current_player)})
        states, policies, values, owns, _kos, own_masks = td.sample(1)
        old_randint = np.random.randint
        try:
            np.random.randint = lambda *a, **k: 3      # rot90 k=1
            out_s, out_p, out_v, out_o, out_m = SelfPlayTrainer._augment_batch(
                states, policies, values, owns, own_masks, 9)
        finally:
            np.random.randint = old_randint
        # 0/1 没有被交换：黑子仍在 ch0，白子仍在 ch1
        np.testing.assert_array_equal(out_s[0, 0], np.rot90(states[0, 0], k=1))
        np.testing.assert_array_equal(out_s[0, 1], np.rot90(states[0, 1], k=1))
        # 贴目、value、own 都不因增强变号；own 只做同一个几何变换
        np.testing.assert_array_equal(out_s[0, 3], states[0, 3])
        np.testing.assert_array_equal(out_v, values)
        np.testing.assert_array_equal(out_o[0], np.rot90(owns[0], k=1))
        np.testing.assert_array_equal(out_m, own_masks)
        # 策略空间部分用同一套旋转，pass 位保留
        expect_p = np.concatenate([
            np.rot90(policies[0, :81].reshape(9, 9), k=1).reshape(-1),
            policies[0, -1:]])
        np.testing.assert_allclose(out_p[0], expect_p)

    def test_manual_policy_relabel_writes_onehot(self):
        g = GoGame(9)
        g.make_move(2, 2)
        g.make_move(6, 6)
        state = g.get_canonical_state()
        policy = np.zeros(82, dtype=np.float32)
        policy[10] = 0.7
        policy[20] = 0.3
        s = state_to_sample(state, policy, g.current_player, g.last_move, g.ko_point)
        payload = pack_game([s], [(2, 2, 1)], 0, 0.0,
                            np.zeros((9, 9), dtype=np.int8), board_size=9)
        d = tempfile.mkdtemp()
        td = TrainingData(max_size=10, data_dir=d, board_size=9)
        path = td.save_game(payload)
        target = 4 * 9 + 4
        set_policy_onehot(path, 0, target)
        self.assertFalse(os.path.exists(path + ".bak"))

        td2 = TrainingData(max_size=10, data_dir=d, board_size=9)
        self.assertTrue(td2.load_all())
        states, policies, _v, _o, _k, _m = td2.sample(1)
        self.assertEqual(int(policies[0].argmax()), target)
        self.assertAlmostEqual(float(policies[0].sum()), 1.0, places=6)
        self.assertEqual(int(td2.data[0].index), 0)
        self.assertEqual(td2.data[0].source, os.path.normcase(os.path.abspath(path)))

    def test_previous_position_hash_allows_multi_capture_at_ko(self):
        g = GoGame(5)
        g.board[:] = 1
        g.board[1, 2] = -1
        g.board[3, 2] = -1
        g.board[2, 2] = 0
        g._st[0] = 1
        g._st[5] = g._st[10]
        h = _board_hash(g.board, 5)
        _set_hash_slots(g._st, h, h)      # 上一局面 = 当前局面，故意设成可提两子
        self.assertTrue(g.is_legal_move(2, 2))
        moves, mask = g.get_legal_moves_and_mask()
        self.assertEqual(float(mask[2 * 5 + 2]), 1.0)

    def test_previous_position_hash_blocks_simple_ko_recapture(self):
        g = GoGame(5)
        for r, c in ((0, 1), (1, 0), (2, 1)):
            g.board[r, c] = 1
        for r, c in ((0, 2), (1, 1), (1, 3), (2, 2)):
            g.board[r, c] = -1
        g._st[0] = 1
        g._st[5] = g._st[10]
        _set_hash_slots(g._st, _board_hash(g.board, 5), np.uint64(0))
        self.assertTrue(g.is_legal_move(1, 2))
        g.make_move(1, 2)
        self.assertEqual(g.ko_point, (1, 1))
        self.assertFalse(g.is_legal_move(1, 1))
        moves, _mask = g.get_legal_moves_and_mask()
        self.assertNotIn((1, 1), moves)

    def test_full_history_blocks_triple_ko_cycle(self):
        """三劫循环：第 6 手回到初始局面，简单劫（只看上一手）会放行。"""
        g = GoGame(9)
        # A：黑提白一子成劫。
        for r, c in ((0, 1), (1, 0), (2, 1)):
            g.board[r, c] = 1
        for r, c in ((0, 2), (1, 1), (1, 3), (2, 2)):
            g.board[r, c] = -1
        # B：白提黑一子成劫。
        for r, c in ((0, 7), (1, 6), (1, 8), (2, 7)):
            g.board[r, c] = 1
        for r, c in ((0, 6), (1, 5), (2, 6)):
            g.board[r, c] = -1
        # C：黑提白一子成劫。
        for r, c in ((4, 1), (5, 0), (6, 1)):
            g.board[r, c] = 1
        for r, c in ((4, 2), (5, 1), (5, 3), (6, 2)):
            g.board[r, c] = -1
        g._st[0] = 1
        g._st[5] = g._st[10]
        _set_hash_slots(g._st, _board_hash(g.board, 9), np.uint64(0))
        self.assertTrue(g.make_move(1, 2))        # 黑提 A
        self.assertTrue(g.make_move(1, 7))        # 白提 B
        self.assertTrue(g.make_move(5, 2))        # 黑提 C
        self.assertTrue(g.make_move(1, 1))        # 白回提 A
        self.assertTrue(g.make_move(1, 6))        # 黑回提 B
        # 白回提 C 会回到初始局面 p0；上一手在 B，简单劫允许，全历史应禁止。
        self.assertFalse(g.is_legal_move(5, 1))
        moves, _mask = g.get_legal_moves_and_mask()
        self.assertNotIn((5, 1), moves)

    def test_superko_capture_with_adjacent_air_is_filtered(self):
        """提子点自己有气也可能提子；不能走「有气」快速路径漏掉超级劫。"""
        g = GoGame(5)
        for r, c in ((0, 1), (1, 0), (2, 1)):
            g.board[r, c] = 1
        g.board[1, 1] = -1
        g._st[0] = 1
        g._st[5] = g._st[10]
        cur = int(_board_hash(g.board, 5))
        after = g.board.copy()
        after[1, 1] = 0
        after[1, 2] = 1
        # 历史里已经有过提子后的局面；上一手哈希则故意设成不匹配。
        g._hist[0] = _board_hash(after, 5)
        g._hist[1] = np.uint64(cur)
        g._st[17] = 2
        _set_hash_slots(g._st, np.uint64(cur), np.uint64(0))
        self.assertFalse(g.is_legal_move(1, 2))
        moves, _mask = g.get_legal_moves_and_mask()
        self.assertNotIn((1, 2), moves)

    def test_hash_slots_track_board(self):
        g = GoGame(9)
        for _ in range(30):
            if g.game_over:
                break
            moves, _mask = g.get_legal_moves_and_mask()
            mv = next((m for m in moves if m != PASS_MOVE), None)
            if mv is None:
                break
            before = _board_hash(g.board, 9)
            g.make_move(*mv)
            cur, prev = _hash_from_slots(g._st)
            self.assertEqual(int(cur), int(_board_hash(g.board, 9)))
            self.assertEqual(int(prev), int(before))

    def test_23_board_hash_and_legal_points(self):
        g = GoGame(23)
        self.assertEqual(g.get_canonical_state().shape, (INPUT_CHANNELS, 23, 23))
        self.assertEqual(len(g.get_legal_moves_and_mask()[0]), 23 * 23)
        for r, c in ((3, 3), (19, 19), (3, 19), (19, 3), (11, 11)):
            before = _board_hash(g.board, 23)
            self.assertTrue(g.is_legal_move(r, c))
            g.make_move(r, c)
            cur, prev = _hash_from_slots(g._st)
            self.assertEqual(int(cur), int(_board_hash(g.board, 23)))
            self.assertEqual(int(prev), int(before))

    def test_pass_clears_ko_and_max_move_flips_player(self):
        g = GoGame(9)
        g._st[3], g._st[4] = 2, 3
        g.make_move(-1, -1)
        self.assertIsNone(g.ko_point)

        g = GoGame(9)
        g._st[5] = g._st[11] - 1
        g.make_move(2, 2)
        self.assertTrue(g.game_over)
        self.assertEqual(g.current_player, -1)

    def test_canonical_state_custom_player(self):
        g = GoGame(9)
        g.make_move(2, 2)
        other = -g.current_player
        state = g.get_canonical_state(player=other)
        self.assertAlmostEqual(float(state[3, 0, 0]), other * KOMI / 14.0, places=6)
        mask = np.zeros(81, dtype=np.float32)
        kr, kc = g.ko_point if g.ko_point is not None else (-2, -2)
        _legal_mask_for(g.board, other, kr, kc, 9, mask)
        np.testing.assert_array_equal(state[6], mask.reshape(9, 9))

    def test_fast_legal_indices_match_tuple_path(self):
        """快速路径必须与 tuple 路径的点集、顺序、mask 完全一致。"""
        rng = random.Random(0)
        for n in (9, 13, 19):
            g = GoGame(n)
            for _step in range(40):
                idx, mask = g.get_legal_indices_and_mask()
                # Numba 按 row-major 填点，pass 在最后；flatnonzero 的顺序相同。
                np.testing.assert_array_equal(idx, np.flatnonzero(mask).astype(np.int32))
                moves, mask2 = g.get_legal_moves_and_mask()
                np.testing.assert_array_equal(mask, mask2)
                back = np.array([n * n if m == PASS_MOVE else m[0] * n + m[1]
                                 for m in moves], dtype=np.int32)
                np.testing.assert_array_equal(idx, back)
                if mask[-1] > 0:
                    self.assertEqual(int(idx[-1]), n * n)
                if not moves or g.game_over:
                    break
                cand = [m for m in moves if m != PASS_MOVE]
                if not cand:
                    break
                g.make_move(*rng.choice(cand))

    def test_save_never_overwrites(self):
        g = GoGame(BOARD_SIZE)
        g.make_move(3, 3)
        state = g.get_canonical_state()
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE + 1, dtype=np.float32)
        policy[0] = 1.0
        s = state_to_sample(state, policy, g.current_player, g.last_move)
        payload = pack_game([s], [(3, 3, 1)], 0, 0.0,
                            np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8))
        d = tempfile.mkdtemp()
        td = TrainingData(max_size=10, data_dir=d)
        p1 = td.save_game(payload)
        p2 = td.save_game(payload)
        self.assertNotEqual(p1, p2)
        self.assertEqual(len([f for f in os.listdir(d) if f.endswith(".npz")]), 2)


class TestHollow(unittest.TestCase):
    def test_boundary_encoding_and_legal_points(self):
        n, h = 13, 7
        g = GoGame(n, hollow=h)
        b = g.get_canonical_state()[CH_BOUNDARY]
        counts = {int(v): int((b == v).sum()) for v in (-1, 0, 1)}
        self.assertEqual(counts, {-1: 49, 0: 40, 1: 80})
        self.assertEqual(int((g.board == HOLE).sum()), 49)
        self.assertEqual(int(g.valid_mask.sum()), 120)
        self.assertEqual(len(g.get_legal_moves_and_mask()[0]), 120)
        self.assertTrue(np.all(b[3:10, 3:10] == -1.0))
        self.assertTrue(np.all(b[2, 2:11] == 1.0))
        self.assertTrue(np.all(b[10, 2:11] == 1.0))
        self.assertTrue(np.all(b[0, :] == 1.0))
        self.assertTrue(np.all(b[:, -1] == 1.0))
        self.assertEqual(float(b[1, 1]), 0.0)
        self.assertTrue(g.is_hole(6, 6))
        self.assertFalse(g.is_legal_move(6, 6))
        self.assertFalse(g.is_hole(2, 6))

        # 3×3 洞、5×5 内边界、9×9 外边界，至少两圈。
        g3 = GoGame(9, hollow=3)
        b3 = g3.get_canonical_state()[CH_BOUNDARY]
        counts3 = {int(v): int((b3 == v).sum()) for v in (-1, 0, 1)}
        self.assertEqual(counts3, {-1: 9, 0: 24, 1: 48})
        self.assertEqual(int(g3.valid_mask.sum()), 72)
        self.assertEqual(len(g3.get_legal_moves_and_mask()[0]), 72)

        # 9 路最大镂空 5（至少两圈）；此时可落子点全是边界，没有 0 点。
        g9 = GoGame(9, hollow=5)
        b9 = g9.get_canonical_state()[CH_BOUNDARY]
        self.assertFalse(np.any(b9 == 0.0))
        self.assertEqual(int(g9.valid_mask.sum()), 56)

    def test_hollow_liberty_and_scoring(self):
        g = GoGame(9, hollow=5)
        # 黑洞正上方一颗黑子，另外三面被白子堵住；洞不能当气。
        g.board[1, 3] = 1
        g.board[0, 3] = -1
        g.board[1, 2] = -1
        g.board[1, 4] = -1
        self.assertFalse(_group_has_liberty(g.board, 1, 3, 9))

        g = GoGame(9, hollow=5)
        d = g._merged_analysis()
        self.assertEqual(d["tb"] + d["tw"] + d["neutral"]
                         + d["black_stones"] + d["white_stones"], 56)
        self.assertTrue(np.all(d["own"][2:7, 2:7] == 0))

    def test_hollow_rules_and_validation(self):
        self.assertEqual(BOARD_SIZES, (7, 9, 11, 13, 15, 17, 19, 21, 23))
        self.assertEqual(hollow_choices(9), [0, 3, 5])
        self.assertEqual(hollow_choices(13), [0, 3, 5, 7, 9])
        self.assertEqual(hollow_choices(19), [0, 3, 5, 7, 9, 11, 13, 15])
        self.assertEqual(playable_area(13, 7), 120)
        self.assertEqual(board_rules(13, 7),
                         {"max_moves": 156, "min_moves": 72, "pass_limit": 6})
        self.assertEqual(board_rules(13, 0)["max_moves"], 220)
        self.assertEqual(board_rules(23, 0)["max_moves"], 688)
        with self.assertRaises(ValueError):
            validate_hollow(13, 11)
        with self.assertRaises(ValueError):
            validate_hollow(13, 4)
        with self.assertRaises(ValueError):
            validate_hollow(13, 1)
        with self.assertRaises(ValueError):
            validate_hollow(9, 7)

    def test_hollow_data_roundtrip(self):
        n, h = 13, 7
        g = GoGame(n, hollow=h)
        g.make_move(1, 1)
        state = g.get_canonical_state()
        moves, _mask = g.get_legal_moves_and_mask()
        mv = moves[0]
        policy = np.zeros(n * n + 1, dtype=np.float32)
        policy[n * n if mv == PASS_MOVE else mv[0] * n + mv[1]] = 1.0
        s = state_to_sample(state, policy, g.current_player, g.last_move, g.ko_point)
        payload = pack_game([s], moves[:1], 0, 0.0,
                            np.zeros((n, n), dtype=np.int8),
                            board_size=n, hollow=h)
        d = tempfile.mkdtemp()
        td = TrainingData(max_size=10, data_dir=d, board_size=n)
        td.save_game(payload)
        td2 = TrainingData(max_size=10, data_dir=d, board_size=n)
        self.assertTrue(td2.load_all())
        states, _p, _v, _o, _k, _m = td2.sample(1)
        self.assertEqual(td2.data[0].hollow, h)
        np.testing.assert_array_equal(states[0, CH_BOUNDARY], state[CH_BOUNDARY])
        self.assertEqual(float(states[0, CH_BOUNDARY, 6, 6]), -1.0)

    def test_hollow_selfplay_complete(self):
        m = MCTS(evaluator=DummyEvaluator(9), num_simulations=3,
                 batch_size=3, board_size=9, cap=200000, dirichlet_epsilon=0)
        r = play_one_game(mcts=m, board_size=9, num_simulations=3,
                          exploration=True, device="cpu", hollow=5)
        self.assertTrue(r["complete"])
        self.assertIsNotNone(r["payload"])
        self.assertEqual(int(r["payload"]["hollow"]), 5)
        # 内存缓冲直接 add_game(result)，result 也必须带 hollow/komi。
        self.assertEqual(int(r["hollow"]), 5)
        self.assertEqual(float(r["komi"]), float(KOMI))
        # 镂空盘不使用四角固定开局，所以每一手都对应一条样本。
        self.assertEqual(len(r["moves"]), len(r["samples"]))
        own = r["ownership"]
        self.assertTrue(np.all(own[2:7, 2:7] == 0))


    def test_hollow_model_masks_ownership_and_value(self):
        n, h = 9, 5
        m = PolicyValueNet(input_channels=INPUT_CHANNELS, channels=8,
                           num_res_blocks=1, head_channels=4)
        x = torch.zeros(2, INPUT_CHANNELS, n, n)
        x[:, CH_BOUNDARY] = torch.from_numpy(boundary_plane(n, h))
        valid = torch.from_numpy(boundary_plane(n, h) >= 0.0)
        with torch.inference_mode():
            _p, v, own_raw, _w = m(x)
        # 训练用的 raw logit 不再硬置 0；value 仍然只对可落子点求和。
        expected = (torch.tanh(own_raw) * valid).sum(dim=(2, 3)).flatten()
        self.assertTrue(torch.allclose(v.flatten(), expected))

        m.eval()
        _pol, val, own, _wl = m.get_policy_value_ownership(x[0].numpy())
        self.assertEqual(float(np.abs(own[2:7, 2:7]).max()), 0.0)
        self.assertAlmostEqual(float(val), float(own.sum()), places=5)

    def test_ownership_loss_mask_includes_holes(self):
        own_target = torch.tensor([[[1.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0],
                                    [0.0, 0.0, -1.0]]])
        own_mask = torch.ones((1, 1, 1), dtype=torch.bool)
        valid = torch.ones((1, 3, 3), dtype=torch.bool)
        valid[0, 1, 1] = False          # 中心是洞
        mask = SelfPlayTrainer._ownership_loss_mask(own_target, own_mask, valid)
        self.assertTrue(bool(mask[0, 0, 0]))      # 可落子点，非 0 标签
        self.assertFalse(bool(mask[0, 0, 1]))     # 可落子点，中立 0 标签
        self.assertTrue(bool(mask[0, 1, 1]))      # 洞，纳入损失
        self.assertTrue(bool(mask[0, 2, 2]))      # 可落子点，非 0 标签
        # own_mask=0（例如人类 SGF）整局不训归属，洞也不训。
        mask_off = SelfPlayTrainer._ownership_loss_mask(
            own_target, torch.zeros_like(own_mask), valid)
        self.assertFalse(bool(mask_off.any()))


class TestTensorRT(unittest.TestCase):
    def test_fallback_without_engine(self):
        n = 9
        m = PolicyValueNet(input_channels=INPUT_CHANNELS, channels=8,
                           num_res_blocks=1, head_channels=4)
        d = tempfile.mkdtemp()
        ev = make_evaluator(m, board_size=n, device="cpu", max_batch=16,
                            use_tensorrt=True, trt_engine_dir=d,
                            log=lambda _msg: None)
        self.assertIsInstance(ev, GoEvaluator)

    def test_architecture_hash_tracks_architecture(self):
        m1 = PolicyValueNet(input_channels=INPUT_CHANNELS, channels=8,
                            num_res_blocks=1, head_channels=4)
        m2 = PolicyValueNet(input_channels=INPUT_CHANNELS, channels=16,
                            num_res_blocks=1, head_channels=4)
        self.assertEqual(architecture_hash(m1), architecture_hash(m1))
        self.assertNotEqual(architecture_hash(m1), architecture_hash(m2))
        ep, mp = engine_paths(tempfile.mkdtemp(), 13, 16, "fp16")
        self.assertTrue(ep.endswith("go_13_b16_fp16_refit.engine"))
        self.assertTrue(mp.endswith("go_13_b16_fp16_refit.json"))


class TestModel(unittest.TestCase):
    def test_strict_load_rejects_old_channels(self):
        old = PolicyValueNet(input_channels=7)
        d = tempfile.mkdtemp()
        p = os.path.join(d, "old.pt")
        old.save_model(p)
        with self.assertRaises(ValueError):
            PolicyValueNet.load_model(p, device="cpu")

    def test_old_9ch_model_migrates_with_small_boundary_init(self):
        old = PolicyValueNet(input_channels=INPUT_CHANNELS - 1)
        d = tempfile.mkdtemp()
        p = os.path.join(d, "old9.pt")
        old.save_model(p)
        m = PolicyValueNet.load_model(p, device="cpu")
        self.assertEqual(m.conv_input.weight.shape[1], INPUT_CHANNELS)
        self.assertTrue(torch.allclose(m.conv_input.weight[:, :INPUT_CHANNELS - 1],
                                       old.conv_input.weight))
        new = m.conv_input.weight[:, INPUT_CHANNELS - 1]
        self.assertGreater(float(new.abs().max()), 0.0)
        self.assertLess(float(new.abs().max()), 0.06)

    def test_boundary_channel(self):
        for n in (7, 9, 13, 19, 23):
            s = GoGame(n).get_canonical_state()
            self.assertEqual(s.shape[0], INPUT_CHANNELS)
            b = s[CH_BOUNDARY]
            self.assertEqual(int(b.sum()), 4 * n - 4)
            self.assertTrue(np.all(b[0] == 1.0) and np.all(b[-1] == 1.0))
            self.assertTrue(np.all(b[:, 0] == 1.0) and np.all(b[:, -1] == 1.0))
            if n > 2:
                self.assertTrue(np.all(b[1:-1, 1:-1] == 0.0))

    def test_channels_last_forward(self):
        m = PolicyValueNet.load_model(MODEL_PATH, device="cpu")
        m = m.to(memory_format=torch.channels_last)
        for n in (7, 9, 13, 19, 23):
            x = torch.randn(2, INPUT_CHANNELS, n, n).to(memory_format=torch.channels_last)
            with torch.inference_mode():
                p, v, o, w = m(x)
            self.assertEqual(tuple(p.shape), (2, n * n + 1))
            self.assertEqual(tuple(v.shape), (2, 1))
            self.assertEqual(tuple(o.shape), (2, 1, n, n))
            self.assertEqual(tuple(w.shape), (2, 1))

    def test_evaluator_multiboard_cpu(self):
        from evaluator import GoEvaluator
        m = PolicyValueNet.load_model(MODEL_PATH, device="cpu")
        e = GoEvaluator(m, device="cpu", max_batch=2, graph=False)
        for n in (7, 9, 13, 19, 23):
            x = np.zeros((2, INPUT_CHANNELS, n, n), dtype=np.float32)
            x[:, 6] = 1.0
            mask = np.ones((2, n * n + 1), dtype=np.float32)
            p, v, w = e.evaluate(x, mask)
            self.assertEqual(p.shape, (2, n * n + 1))
            self.assertTrue(np.allclose(p.sum(axis=1), 1.0, atol=1e-5))


class TestMCTS(unittest.TestCase):
    def test_temperature_zero_per_board(self):
        expected = {7: 11, 9: 18, 11: 27, 13: 37, 15: 50,
                    17: 64, 19: 79, 21: 97, 23: 116}
        for n in BOARD_SIZES:
            self.assertEqual(board_temperature_zero(n), expected[n])
            z = expected[n]
            self.assertEqual(temperature_schedule(1.0, z - 1, z), 1.0)
            self.assertEqual(temperature_schedule(1.0, z, z), 0.0)
        self.assertEqual(board_temperature_zero(13, 7), 26)

    def test_pass_terminal_value_asymmetry(self):
        """PASS 终局按最后一手 PASS 的人结算：胜方 +1.0，败方 -0.1。"""
        m = MCTS(evaluator=DummyEvaluator(9), num_simulations=1,
                 batch_size=1, board_size=9)
        g = GoGame(9)
        g.last_move = PASS_MOVE
        g.winner = 1
        # 黑胜；白是最后 PASS 的人（失败方）⇒ 白自己的 PASS 搜索值 -0.1。
        g.current_player = 1
        self.assertAlmostEqual(m._terminal_value(g), -PASS_TERMINAL_LOSS, places=6)
        self.assertAlmostEqual(-m._terminal_value(g), PASS_TERMINAL_LOSS, places=6)
        # 黑胜；黑是最后 PASS 的人（获胜方）⇒ 黑自己的 PASS 搜索值 +1.0。
        g.current_player = -1
        self.assertEqual(m._terminal_value(g), -1.0)
        self.assertEqual(-m._terminal_value(g), 1.0)
        g.winner = 0
        self.assertEqual(m._terminal_value(g), 0.0)

    def test_exploration_opening_positions(self):
        for n in (9, 13, 19):
            for _ in range(40):
                op = _exploration_opening(n)
                self.assertEqual(len(op), 4)
                self.assertEqual(len({(r, c) for r, c, _p in op}), 4)
                for r, c, _p in op:
                    self.assertTrue(0 <= r < n and 0 <= c < n)
                    if n == 9:
                        # 9 路也不能用 2-2/2-3 这类二线点；星位固定 4-4。
                        self.assertGreaterEqual(min(r, c), 2)

    def test_smoke(self):
        m = MCTS(evaluator=DummyEvaluator(9), num_simulations=20,
                 batch_size=4, board_size=9, cap=100000)
        g = GoGame(9)
        policy, sample = m.get_move_probs(g, temp=1.0, target_visits=20)
        self.assertTrue(policy)
        self.assertAlmostEqual(sum(policy.values()), 1.0, places=5)
        self.assertIn(max(sample, key=sample.get), g.get_legal_moves_and_mask()[0])

    def test_root_candidate_pruning(self):
        g = GoGame(9)
        m = MCTS(evaluator=DummyEvaluator(9), num_simulations=10,
                 batch_size=2, board_size=9, cap=100000,
                 candidate_mass=0.1, min_candidates=8, dirichlet_epsilon=0)
        m.get_move_probs(g, temp=0.0, target_visits=10)
        self.assertLess(len(m.root.children), 81)
        self.assertGreaterEqual(len(m.root.children), 8)

        g2 = GoGame(9)
        g2._st[5] = g2._st[10]
        m2 = MCTS(evaluator=DummyEvaluator(9), num_simulations=10,
                  batch_size=2, board_size=9, cap=100000,
                  candidate_mass=0.01, min_candidates=2, dirichlet_epsilon=0)
        m2.get_move_probs(g2, temp=0.0, target_visits=10)
        self.assertIn(PASS_MOVE, m2.root.children)

    def test_root_noise_before_candidate_pruning(self):
        """噪声必须先加到完整合法着法，再做候选裁剪。"""
        g = GoGame(9)
        moves, _mask = g.get_legal_moves_and_mask()
        chosen = moves[-1]
        self.assertNotEqual(chosen, PASS_MOVE)
        noise = np.zeros(len(moves), dtype=np.float64)
        noise[-1] = 1.0
        m = MCTS(evaluator=DummyEvaluator(9), num_simulations=10,
                 batch_size=4, board_size=9, cap=100000,
                 candidate_mass=0.5, min_candidates=1,
                 dirichlet_epsilon=1.0, dirichlet_alpha=0.03)
        with patch.object(np.random, "dirichlet", return_value=noise):
            m.init_root(g)
        # 旧顺序会先保留均匀先验的前 41 个点，最后一个点进不了根。
        self.assertIn(chosen, m.root.children)
        self.assertAlmostEqual(m.root.children[chosen].prior_prob, 1.0, places=6)
        # init_root 已经混过噪声，后续 GUI 的 mix_root_noise 不能重复混。
        before = {k: v.prior_prob for k, v in m.root.children.items()}
        m.mix_root_noise()
        after = {k: v.prior_prob for k, v in m.root.children.items()}
        self.assertEqual(before, after)

    def test_pass_probe_finds_forced_win(self):
        n = 9
        g = GoGame(n)
        g.make_move(2, 2)
        g.make_move(6, 6)
        g._st[5] = g._st[10]
        g._st[6] = 1          # 白刚 pass，黑只要 pass 就终局
        g._st[1] = g._st[2] = -1
        g._st[3] = g._st[4] = -2
        g._st[0] = 1
        g._go[0] = False
        orig = GoGame._compute_winner
        def fake(self):
            self.winner = 1
            self.final_points = 100.0
        GoGame._compute_winner = fake
        try:
            m = MCTS(evaluator=DummyEvaluator(n), num_simulations=20,
                     batch_size=4, board_size=n, cap=100000, dirichlet_epsilon=0)
            m.get_move_probs(g, temp=0.0, target_visits=20)
            self.assertGreater(m.root.children[PASS_MOVE].visit_count, 0)
        finally:
            GoGame._compute_winner = orig

    def test_terminal_leaf_does_not_crash(self):
        """回归：终局叶子的 legal 必须是空数组，不能是 None。

        Numba 会为这个调用点编译 _backup_np；None 会变成 getitem(none, int64)。
        用 5 模拟跑完一整局，确保真的走到终局叶子。
        """
        m = MCTS(evaluator=DummyEvaluator(19), num_simulations=5,
                 batch_size=4, board_size=19, cap=600000)
        r = play_one_game(mcts=m, num_simulations=5, exploration=False,
                          device="cpu")
        self.assertTrue(r["complete"])
        self.assertIsNotNone(r["payload"])


if __name__ == "__main__":
    unittest.main(verbosity=2)