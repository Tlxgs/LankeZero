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

from game import (GoGame, PASS_MOVE, _legal_mask_for, _board_hash,
                  _hash_from_slots, _set_hash_slots)
from hyperparams import (BOARD_SIZE, KOMI, BOARD_TEMPERATURE_ZERO,
                         board_temperature_zero, temperature_schedule,
                         INPUT_CHANNELS, CH_BOUNDARY, MODEL_PATH)
from model import PolicyValueNet
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
        states, policies, values, owns, kos = td2.sample(1)
        np.testing.assert_allclose(states[0], state, atol=0.0)
        np.testing.assert_allclose(policies[0], policy, atol=0.0)
        self.assertEqual(values[0], 1.0)
        np.testing.assert_array_equal(owns[0], own * g.current_player)

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
        states, policies, _values, _owns, _kos = td2.sample(1)
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
        _s, _p, _v, _o, kos = td2.sample(1)
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
        states, policies, values, owns, _kos = td.sample(1)
        old_randint = np.random.randint
        try:
            np.random.randint = lambda *a, **k: 3      # rot90 k=1
            out_s, out_p, out_v, out_o = SelfPlayTrainer._augment_batch(
                states, policies, values, owns, 9)
        finally:
            np.random.randint = old_randint
        # 0/1 没有被交换：黑子仍在 ch0，白子仍在 ch1
        np.testing.assert_array_equal(out_s[0, 0], np.rot90(states[0, 0], k=1))
        np.testing.assert_array_equal(out_s[0, 1], np.rot90(states[0, 1], k=1))
        # 贴目、value、own 都不因增强变号；own 只做同一个几何变换
        np.testing.assert_array_equal(out_s[0, 3], states[0, 3])
        np.testing.assert_array_equal(out_v, values)
        np.testing.assert_array_equal(out_o[0], np.rot90(owns[0], k=1))
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
        states, policies, _v, _o, _k = td2.sample(1)
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
        for n in (9, 13, 19):
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
        for n in (9, 13, 19):
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
        for n in (9, 13, 19):
            x = np.zeros((2, INPUT_CHANNELS, n, n), dtype=np.float32)
            x[:, 6] = 1.0
            mask = np.ones((2, n * n + 1), dtype=np.float32)
            p, v, w = e.evaluate(x, mask)
            self.assertEqual(p.shape, (2, n * n + 1))
            self.assertTrue(np.allclose(p.sum(axis=1), 1.0, atol=1e-5))


class TestMCTS(unittest.TestCase):
    def test_temperature_zero_per_board(self):
        self.assertEqual(BOARD_TEMPERATURE_ZERO, {9: 15, 13: 40, 19: 80})
        for n, z in ((9, 15), (13, 40), (19, 80)):
            self.assertEqual(board_temperature_zero(n), z)
            self.assertEqual(temperature_schedule(1.0, z - 1, z), 1.0)
            self.assertEqual(temperature_schedule(1.0, z, z), 0.0)

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