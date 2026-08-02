"""
自我对弈训练模块
"""
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import random
import os
import pickle
import glob
from datetime import datetime
from game import GoGame, PASS_MOVE,SCALE,upgrade_state_channels
from mcts import MCTS
from model import PolicyValueNet

# ==================== 超参数 ====================
BOARD_SIZE = 13
NUM_SIMULATIONS = 200
C_PUCT = 1.5
TEMPERATURE = 0.8
TEMPERATURE_DECAY = 0.99
TOP_P = 0.9
EXPLORATION_MODE = False   # 是否在开局随机落子
BATCH_SIZE = 384
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.0001

DATA_DIR = 'data/'
MODEL_PATH = 'model.pt'
ONNX_PATH = 'model.onnx'

class TrainingData:
    def __init__(self, max_size=80000, data_dir=DATA_DIR,board_size=BOARD_SIZE):
        self.max_size = max_size
        self.data_dir = data_dir
        self.board_size = board_size
        self.data = deque(maxlen=max_size)
        os.makedirs(data_dir, exist_ok=True)
    @staticmethod
    def _compute_value(winner, player, score_diff, scale=SCALE):
        if score_diff is not None:
            diff = score_diff if player == 1 else -score_diff
            return np.tanh(diff / scale)
        else:
            return 0.0 if winner == 0 else (1.0 if winner == player else -1.0)

    @staticmethod
    def _compute_ownership_view(ownership_abs, player):
        if ownership_abs is None:
            return None
        own = np.zeros_like(ownership_abs)
        own[ownership_abs == player] = 1
        own[ownership_abs == -player] = -1
        return own

    def add(self, state, policy, value, ownership=None):
        self.data.append((state, policy, value, ownership))

    def add_game(self, states, policies, players, winner, score_diff, ownership_abs=None):
        for state, policy, player in zip(states, policies, players):
            value = self._compute_value(winner, player, score_diff)
            own_view = self._compute_ownership_view(ownership_abs, player)
            self.add(state, policy, value, own_view)

    # sample: 返回4元组
    def sample(self, batch_size):
        batch = random.sample(list(self.data), min(batch_size, len(self.data)))
        states, policies, values, ownerships = zip(*batch)
        bs = len(batch)
        h, w = states[0].shape[1], states[0].shape[2]
        own = np.zeros((bs, h, w), dtype=np.int64) 
        for i, o in enumerate(ownerships):
            if o is not None:
                own[i] = o.astype(np.int64)
        return (np.array(states, dtype=np.float32),
                np.array(policies, dtype=np.float32),
                np.array(values, dtype=np.float32), own)


    def __len__(self):
        return len(self.data)

    def save_game(self, game_data):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(self.data_dir, f"data_{timestamp}.pkl")
        states, policies, players, winner, score_diff,ownership = game_data   # 注意解包个数
        info = {
            'states': states,
            'policies': policies,
            'players': players,
            'winner': winner,
            'score_diff': score_diff,   # 新增
            'timestamp': datetime.now().isoformat(),
            'num_moves': len(states),
            'ownership':ownership
        }
        with open(filepath, 'wb') as f:
            pickle.dump(info, f)

    def load_all(self):
        files = glob.glob(os.path.join(self.data_dir, "data_*.pkl"))
        if not files:
            return False
        files.sort(key=os.path.getmtime, reverse=True)
        all_data = []
        for f in files:
            try:
                with open(f, 'rb') as fp:
                    info = pickle.load(fp)
                states = info['states']
                policies = info['policies']
                players = info['players']
                winner = info['winner']
                ownership_abs = info.get('ownership', None) 
                score_diff = info.get('score_diff', None)   # 兼容旧数据
                
                for i, (s, p, pl) in enumerate(zip(states, policies, players)):
                    if s.shape[0] == 4:                    # 旧数据 4→6 通道升级
                        s = upgrade_state_channels(s, pl)  # 用 players 里的真实玩家，精确重建气数
                    value = self._compute_value(winner, pl, score_diff)
                    own_view = self._compute_ownership_view(ownership_abs, pl)
                    all_data.append((s, p, value, own_view))
                if len(all_data) >= self.max_size:
                    break
            except Exception as e:
                print(f"加载 {f} 失败: {e}")
        self.data.clear()
        for item in all_data[:self.max_size]:
            self.data.append(item)
        print(f"[数据] 加载了 {len(self.data)} 条数据")
        return True


def play_one_game(board_size=BOARD_SIZE, num_simulations=NUM_SIMULATIONS,
                  device='cuda', c_puct=C_PUCT, temperature=TEMPERATURE,
                  exploration=EXPLORATION_MODE, onnx_path=ONNX_PATH):
    """生成一局游戏，返回 (states, policies, players, moves, winner)"""
    mcts = MCTS(c_puct=c_puct, num_simulations=num_simulations,
                temperature=temperature, onnx_path=onnx_path, device=device,
                board_size=board_size)
    """mu = 7.0      # 均值
    sigma = 7.0   # 标准差
    komi = random.gauss(mu, sigma)
    """
    komi = random.randint(0,25)*0.5
    game = GoGame(board_size,komi = komi)
    states, policies, players = [], [], []
    moves = []
    move_count = 0
    mcts.root = None

    while not game.game_over:
        temp = max(0.1, temperature * (TEMPERATURE_DECAY ** move_count))
        move_probs = mcts.get_move_probs(game, temp,num_simulations)
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
            mcts.update_root(game, selected)
        move_count += 1

    ownership = game.compute_ownership_map()   # 整局所有局面共享同一张终局归属图
    return states, policies, players, moves, game.winner, game.final_points, ownership


class SelfPlayTrainer:
    def __init__(self, model_path=MODEL_PATH, board_size=BOARD_SIZE,
                 device='cuda', num_simulations=NUM_SIMULATIONS,
                 data_dir=DATA_DIR, c_puct=C_PUCT,
                 learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
        self.model_path = model_path
        self.board_size = board_size
        self.device = device
        self.num_simulations = num_simulations
        self.data_dir = data_dir
        self.c_puct = c_puct
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        os.makedirs(data_dir, exist_ok=True)

        self.model = PolicyValueNet.load_model(model_path).to(device)
        self.data_buffer = TrainingData(data_dir=data_dir, board_size = board_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate, weight_decay=weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: 1.0
        )
        self.game_count = 0
        self.train_count = 0
    def _augment_batch(self, states, policies, ownerships):
        """数据增强：对策略向量只变换棋盘部分，pass维度保持不变。
        使用完整的 D4 对称群（8 种变换），state / policy / ownership 使用完全一致的坐标变换。
        """
        batch_size = states.shape[0]
        board_size = self.board_size
        n = board_size * board_size

        transforms = [
            # 1. 恒等
            (lambda x: x,
            lambda p: p),

            # 2. 上下翻转（垂直镜像）: (r,c) -> (n-1-r, c)
            (lambda x: np.flip(x, axis=1),
            lambda p: np.concatenate(
                [p[:n].reshape(board_size, board_size)[::-1, :].flatten(), p[-1:]])),

            # 3. 左右翻转（水平镜像）: (r,c) -> (r, n-1-c)
            (lambda x: np.flip(x, axis=2),
            lambda p: np.concatenate(
                [p[:n].reshape(board_size, board_size)[:, ::-1].flatten(), p[-1:]])),

            # 4. 旋转 90°
            (lambda x: np.rot90(x, k=1, axes=(1, 2)),
            lambda p: np.concatenate(
                [np.rot90(p[:n].reshape(board_size, board_size), k=1).flatten(), p[-1:]])),

            # 5. 旋转 180°
            (lambda x: np.rot90(x, k=2, axes=(1, 2)),
            lambda p: np.concatenate(
                [np.rot90(p[:n].reshape(board_size, board_size), k=2).flatten(), p[-1:]])),

            # 6. 旋转 270°
            (lambda x: np.rot90(x, k=3, axes=(1, 2)),
            lambda p: np.concatenate(
                [np.rot90(p[:n].reshape(board_size, board_size), k=3).flatten(), p[-1:]])),

            # 7. 主对角线镜像（转置）
            (lambda x: np.transpose(x, (0, 2, 1)),
            lambda p: np.concatenate(
                [p[:n].reshape(board_size, board_size).T.flatten(), p[-1:]])),

            # 8. 副对角线镜像
            (lambda x: np.flip(np.transpose(x, (0, 2, 1)), axis=(1, 2)),
            lambda p: np.concatenate(
                [np.flip(p[:n].reshape(board_size, board_size).T, axis=(0, 1)).flatten(), p[-1:]])),
        ]

        # 归属图 (H,W) 与棋盘使用完全相同的8种几何变换（一一对应上面的state变换）
        own_funcs = [
            lambda x: x,
            lambda x: np.flip(x, axis=0),               # 对应变换2 (state flip axis=1)
            lambda x: np.flip(x, axis=1),               # 对应变换3 (state flip axis=2)
            lambda x: np.rot90(x, k=1),                 # 对应变换4
            lambda x: np.rot90(x, k=2),                 # 对应变换5
            lambda x: np.rot90(x, k=3),                 # 对应变换6
            lambda x: x.T,                              # 对应变换7
            lambda x: np.flip(x.T, axis=(0, 1)),        # 对应变换8
        ]

        aug_states = []
        aug_policies = []
        aug_ownerships = []
        for i in range(batch_size):
            s, p, o = states[i], policies[i], ownerships[i]
            t_idx = np.random.randint(len(transforms))
            func_s, func_p = transforms[t_idx]
            aug_states.append(func_s(s))
            aug_policies.append(func_p(p))
            aug_ownerships.append(own_funcs[t_idx](o))
        return (np.array(aug_states, dtype=np.float32),
                np.array(aug_policies, dtype=np.float32),
                np.array(aug_ownerships, dtype=np.int64))



    def train_step(self, batch_size=BATCH_SIZE):
        if len(self.data_buffer) < batch_size:
            return None, None, None, None
        states, policies, values, ownerships = self.data_buffer.sample(batch_size)
        states, policies, ownerships = self._augment_batch(states, policies, ownerships)
        states_t = torch.FloatTensor(states).to(self.device)
        policies_t = torch.FloatTensor(policies).to(self.device)
        values_t = torch.FloatTensor(values).unsqueeze(1).to(self.device)

        self.model.train()
        policy_logits, pred_values, own_logits = self.model(states_t)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policies_t * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(pred_values, values_t)
        probs = F.softmax(policy_logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        # 领地归属辅助损失（单通道tanh回归；0=中立/无归属样本被忽略）
        own_pred = torch.tanh(own_logits.squeeze(1))            # (B,H,W) ∈(-1,1)
        own_target = torch.from_numpy(ownerships).to(self.device).float()
        mask = own_target != 0
        if mask.any():
            own_loss = F.mse_loss(own_pred[mask], own_target[mask])
        else:
            own_loss = torch.zeros((), device=self.device)

        # 归属头求和 → 黑白绝对目差（不含贴目）；价值头是当前玩家视角、含贴目
        has_own = (own_target != 0).any(dim=(1, 2))             # (B,) 有归属标签的样本
        if has_own.any():
            komi_ch = states_t[has_own, 3, 0, 0]                # = player*komi/14
            player = torch.sign(komi_ch) 
            komi = komi_ch.abs() * 14.0
            own_sum = own_pred[has_own].sum(dim=(1, 2))         
            score_own = own_sum - komi * player
            cons_target = torch.tanh(score_own / SCALE)         # 与价值头同一压缩尺度
            cons_loss = F.mse_loss(pred_values[has_own].squeeze(1), cons_target)
        else:
            cons_loss = torch.zeros((), device=self.device)


        entropy_weight = 0.15
        value_weight = 1.0
        own_weight = 0.9                                        # 归属标签损失权重
        cons_weight = 0.3                                       # 自洽正则权重（调小点）
        total_loss = policy_loss + value_weight * value_loss \
                - entropy_weight * entropy \
                + own_weight * own_loss +cons_weight*cons_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return policy_loss.item(), value_loss.item(), entropy.item(), total_loss.item()

    def save_model(self):
        self.model.save_model(self.model_path)
        self.model.export_onnx(ONNX_PATH)

    def load_training_data(self, filename=None):
        if filename is not None:
            return self.data_buffer.load_single(filename)
        else:
            return self.data_buffer.load_all()

    def save_training_data(self, game_data):
        self.data_buffer.save_game(game_data)

    def update_hyperparameters(self, **kwargs):
        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']
            for g in self.optimizer.param_groups:
                g['lr'] = self.learning_rate
        if 'weight_decay' in kwargs:
            self.weight_decay = kwargs['weight_decay']
            for g in self.optimizer.param_groups:
                g['weight_decay'] = self.weight_decay
        if 'c_puct' in kwargs:
            self.c_puct = kwargs['c_puct']
        if 'num_simulations' in kwargs:
            self.num_simulations = kwargs['num_simulations']