"""
自我对弈训练模块
"""
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import os
# 显存碎片控制：训练与 worker 的 TRT 引擎共享 4GB 显存，降低碎片可避免"小分配失败"崩溃
from hyperparams import PYTORCH_CUDA_ALLOC_CONF
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", PYTORCH_CUDA_ALLOC_CONF)

# ---- 训练/自对弈性能优化（全局生效）----
torch.backends.cudnn.benchmark = True          # cuDNN 自动选择最优卷积算法
torch.set_float32_matmul_precision('high')     # 矩阵乘允许 TF32（torch 内部）
import random
import pickle
import glob
from datetime import datetime
from game import PASS_MOVE, SCALE
from model import PolicyValueNet

# ==================== 超参数（统一配置见 hyperparams.py） ====================
from hyperparams import (BOARD_SIZE, NUM_SIMULATIONS, C_PUCT,
                         BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE,
                         WEIGHT_DECAY, MOMENTUM, OPTIMIZER_NAME, WARMUP_STEPS,
                         ENTROPY_WEIGHT, OWN_WEIGHT, WIN_WEIGHT, GRAD_CLIP_NORM,
                         DATA_DIR, MODEL_PATH, ONNX_PATH,
                         TRAINING_DATA_MAX)

class TrainingData:
    def __init__(self, max_size=TRAINING_DATA_MAX, data_dir=DATA_DIR,board_size=BOARD_SIZE):
        self.max_size = max_size
        self.data_dir = data_dir
        self.board_size = board_size
        self.data = deque(maxlen=max_size)
        os.makedirs(data_dir, exist_ok=True)
    @staticmethod
    def _compute_value(winner, player, score_diff=None, scale=SCALE):
        # 纯胜负标签 +1/-1/0：目差信息由 own 头与搜索层负责，value 头只学胜负
        return 0.0 if winner == 0 else (1.0 if winner == player else -1.0)

    @staticmethod
    def _compute_ownership_view(ownership_abs, player):
        if ownership_abs is None:
            return None
        own = np.zeros_like(ownership_abs,dtype=np.int8)
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

    # sample: 返回4元组（states/policies/values(±1胜负)/ownership）
    def sample(self, batch_size):
        batch = random.sample(list(self.data), min(batch_size, len(self.data)))
        states, policies, values, ownerships = zip(*batch)
        bs = len(batch)
        h, w = states[0].shape[1], states[0].shape[2]
        own = np.zeros((bs, h, w), dtype=np.int8) 
        for i, o in enumerate(ownerships):
            if o is not None:
                own[i] = o.astype(np.int64)
        return (np.array(states, dtype=np.float32),
                np.array(policies, dtype=np.float32),
                np.array(values, dtype=np.float32), own)


    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # 支持 list(training_data) 遍历 / td[idx] 按索引访问
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def save_game(self, game_data):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(self.data_dir, f"data_{timestamp}.pkl")
        states, policies, players, winner, score_diff, ownership = game_data  # 注意解包个数
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

    def load_all(self, mode='newest'):
        """批量加载 data/*.pkl 直到填满缓冲区（TRAINING_DATA_MAX）。
        mode='newest'：按文件修改时间倒序取最新（默认，原逻辑）；
        mode='random'：data 目录下随机抽取（洗牌后顺序加载）。"""
        files = glob.glob(os.path.join(self.data_dir, "data_*.pkl"))
        if not files:
            return False
        if mode == 'random':
            random.shuffle(files)
        else:
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
                ownership_abs = info['ownership']
                score_diff = info['score_diff']

                
                for i, (s, p, pl) in enumerate(zip(states, policies, players)):
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
        print(f"[数据] 加载了 {len(self.data)} 条数据（模式: {mode}）")
        return True

    def load_single(self, filename):
        """加载单个 pkl 数据文件（train_gui"加载数据"按钮），合并进现有缓冲。"""
        with open(filename, 'rb') as fp:
            info = pickle.load(fp)
        added = 0
        for s_, p, pl in zip(info['states'], info['policies'], info['players']):
            value = self._compute_value(info['winner'], pl, info['score_diff'])
            own_view = self._compute_ownership_view(info['ownership'], pl)
            self.data.append((s_, p, value, own_view))
            added += 1
        print(f"[数据] 已从 {os.path.basename(filename)} 加载 {added} 条数据")
        return added > 0



# play_one_game / _exploration_opening / worker_process 已迁至 selfplay.py
# （数据生成模块，纯生成不加载 torch）：worker 与 eval 只 import selfplay，
# 避免 spawn 子进程连带加载 torch 造成多余内存占用。

class SelfPlayTrainer:
    def __init__(self, model_path=MODEL_PATH, board_size=BOARD_SIZE,
                 device='cuda', num_simulations=NUM_SIMULATIONS,
                 data_dir=DATA_DIR, c_puct=C_PUCT,
                 learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                 optimizer_name=OPTIMIZER_NAME, momentum=MOMENTUM,
                 freeze_parts=None, reset_bn=False):
        """freeze_parts: 冻结参数规格（None/''=不冻结），格式见 PolicyValueNet.freeze_parts，
        例如 '0,1,2,3,5,7'（冻结这些残差块）或 '0,own,policy'（冻结块0+归属头+策略头）。"""
        self.model_path = model_path
        self.board_size = board_size
        self.device = device
        self.num_simulations = num_simulations
        self.data_dir = data_dir
        self.c_puct = c_puct
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name.lower()
        self.momentum = momentum
        os.makedirs(data_dir, exist_ok=True)

        self.model = PolicyValueNet.load_model(model_path, reset_bn=reset_bn).to(device)
        self.frozen_params_count = 0
        if freeze_parts:
            # 在创建优化器之前冻结：优化器只挂载可训练参数（Adam/SGD 均不更新冻结层）
            self.frozen_params_count = self.model.freeze_parts(freeze_parts)
            print('[训练] 冻结参数: %s（%d 个参数）' % (freeze_parts, self.frozen_params_count))
        self.data_buffer = TrainingData(data_dir=data_dir, board_size = board_size)
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: 1.0
        )
        self.game_count = 0
        self.train_count = 0
        self.warmup_steps = WARMUP_STEPS   # 学习率warmup步数（前N步从0线性升至目标lr，SGD/Adam均生效）
        self.grad_accum_steps = GRAD_ACCUM_STEPS  # 梯度累积：每 N 个 micro-batch 才做一次 optimizer.step()
        self._accum_steps_done = 0                # 当前累积周期内已完成反向的 micro-batch 数
        self._opt_steps = 0                       # 实际 optimizer.step() 次数（warmup 调度基准）
        

    def _build_optimizer(self):
        """按优化器类型创建优化器（Adam/SGD，SGD含动量）。
        只挂载 requires_grad=True 的参数：冻结层不参与更新（见 freeze_parts）。
        KataGo 惯例：所有 1D 参数（bias + BatchNorm 的 weight/bias）不参与 weight decay——
        BN 的 affine 缩放会被后续层抵消，对其正则化无意义，且会持续缩小有效表达。"""
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        groups = [
            {'params': decay, 'weight_decay': self.weight_decay, 'wd_enabled': True},
            {'params': no_decay, 'weight_decay': 0.0, 'wd_enabled': False},
        ]
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(groups, lr=self.learning_rate, momentum=self.momentum)
        return torch.optim.Adam(groups, lr=self.learning_rate)
    @staticmethod
    def _augment_batch(states, policies, values, ownerships, board_size):
        """数据增强：对策略向量只变换棋盘部分，pass维度保持不变。
        使用完整的 D4 对称群（8 种变换），state / policy / ownership 使用完全一致的坐标变换。
        values（±1 胜负标签）为标量：8 种几何变换不变；黑白互换（视角翻转）时必须取反，
        否则一半样本标签符号错误 →胜率头被迫输出0（损失恒为1.0）。
        """
        batch_size = states.shape[0]
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
        aug_values = []
        aug_ownerships = []
        for i in range(batch_size):
            s, p, v, o = states[i], policies[i], values[i], ownerships[i]
            t_idx = np.random.randint(len(transforms))
            func_s, func_p = transforms[t_idx]
            s = func_s(s)
            p = func_p(p)
            o = own_funcs[t_idx](o)
            # 黑白互换增强（视角切换→倒贴目）：模型输入天然支持，50%概率
            if np.random.rand() < 0.5:
                s = s.copy()  # func_s可能返回视图(np.flip/rot90/transpose)，复制避免污染数据缓存
                s[0], s[1] = s[1].copy(), s[0].copy()  # 己方/对方棋子互换
                s[3] = -s[3]                            # 玩家符号+贴目取反
                s[4], s[5] = s[5].copy(), s[4].copy()  # 己方/对方危急度互换
                o = -o                                  # 归属标签取反（当前视角→对手视角）
                v = -v                                  # 胜率标签取反（当前视角→对手视角）
            aug_states.append(s)
            aug_policies.append(p)
            aug_values.append(v)
            aug_ownerships.append(o)
        return (np.array(aug_states, dtype=np.float32),
                np.array(aug_policies, dtype=np.float32),
                np.array(aug_values, dtype=np.float32),
                np.array(aug_ownerships, dtype=np.int8))



    def train_step(self, batch_size=BATCH_SIZE):
        if len(self.data_buffer) < batch_size:
            return None, None, None, None, None
        states, policies, values, ownerships = self.data_buffer.sample(batch_size)
        states, policies, values, ownerships = SelfPlayTrainer._augment_batch(
            states, policies, values, ownerships, self.board_size)
        states_t = torch.FloatTensor(states).to(self.device)
        policies_t = torch.FloatTensor(policies).to(self.device)

        self.model.train()
        policy_logits, pred_values, own_logits, win_logits = self.model(states_t)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policies_t * log_probs).sum(dim=1).mean()
        probs = F.softmax(policy_logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        # 领地损失（唯一价值监督：价值 = 领地求和，KataGo式）
        own_pred = torch.tanh(own_logits.squeeze(1))            # (B,H,W) ∈(-1,1)
        own_target = torch.from_numpy(ownerships).to(self.device).float()
        mask = own_target != 0
        if mask.any():
            own_loss = F.mse_loss(own_pred[mask], own_target[mask])
        else:
            own_loss = torch.zeros((), device=self.device)


        # 胜率头损失（BCE风格）：tanh(z)=2·sigmoid(2z)−1，直接对 logits 算（数值稳定）
        # 标签 y=(values+1)/2 ∈{0,1}（±1胜负；和棋0→0.5，朝向50%）
        win_target = (torch.from_numpy(values).to(self.device).float() + 1.0) / 2.0
        win_loss = F.binary_cross_entropy_with_logits(2.0 * win_logits.squeeze(1), win_target)

        entropy_weight = ENTROPY_WEIGHT
        own_weight = OWN_WEIGHT
        total_loss = (policy_loss - entropy_weight * entropy
                      + own_weight * own_loss + WIN_WEIGHT * win_loss)

        # ---- 梯度累积 ----
        # 每 grad_accum_steps 个 micro-batch 才做一次 optimizer.step()；
        # loss 除以累积步数再反向 → 累积梯度≈平均梯度（等效 batch = grad_accum_steps × batch_size）。
        # BN 的 running stats 按 micro-batch 正常更新，与常规训练一致。
        if self._accum_steps_done == 0:
            self.optimizer.zero_grad()
        (total_loss / self.grad_accum_steps).backward()
        self._accum_steps_done += 1
        if self._accum_steps_done >= self.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
            self.optimizer.step()
            self._accum_steps_done = 0
            self._opt_steps += 1
            self._apply_warmup()
        # 释放 torch 缓存分配器持有的空闲显存给驱动（worker 的 TRT 引擎构建/加载需要显存）
        torch.cuda.empty_cache()
        return (policy_loss.item(), own_loss.item(), entropy.item(),
                win_loss.item(), total_loss.item())

    def _apply_warmup(self):
        """学习率 warmup：按实际 optimizer.step() 次数从 0 线性升至目标 lr
        （梯度累积下以真实步数为准；accum=1 时与原先按 train_count 调度等价）。"""
        if self.warmup_steps > 0 and self._opt_steps <= self.warmup_steps:
            scale = self._opt_steps / self.warmup_steps
            for g in self.optimizer.param_groups:
                g['lr'] = self.learning_rate * scale

    def flush_grad_accum(self):
        """梯度累积收尾：累积周期不满（余下 < grad_accum_steps 个 micro-batch）时
        把已累积的梯度也执行一次 step，避免训练末尾/重建优化器时丢失梯度更新。"""
        if getattr(self, '_accum_steps_done', 0) > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
            self.optimizer.step()
            self._accum_steps_done = 0
            self._opt_steps += 1
            self._apply_warmup()

    def save_model(self):
        # 原子保存：先写 .tmp 再 os.replace，避免 worker 评估/推理线程读到半截文件
        # （torch.load 报 "PytorchStreamReader failed reading file data/N" = 读到了写入中的 checkpoint）
        tmp_pt = self.model_path + '.tmp'
        self.model.save_model(tmp_pt)
        os.replace(tmp_pt, self.model_path)
        tmp_onnx = ONNX_PATH + '.tmp'
        self.model.export_onnx(tmp_onnx)
        os.replace(tmp_onnx, ONNX_PATH)
        # 保存后 worker 会立即用新 onnx 重建 TRT 引擎：先把 torch 空闲显存还给驱动
        torch.cuda.empty_cache()

    def load_training_data(self, filename=None, mode='newest'):
        if filename is not None:
            return self.data_buffer.load_single(filename)
        else:
            return self.data_buffer.load_all(mode=mode)

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
                if g.get('wd_enabled', True):   # 只更新衰减组；BN/bias 组恒为 0（KataGo 惯例）
                    g['weight_decay'] = self.weight_decay
        rebuild = False
        if 'optimizer_name' in kwargs and kwargs['optimizer_name'].lower() != self.optimizer_name:
            self.optimizer_name = kwargs['optimizer_name'].lower()
            rebuild = True
        if 'momentum' in kwargs and self.optimizer_name == 'sgd':
            self.momentum = kwargs['momentum']
            rebuild = True
        if rebuild:  # 切换优化器/动量 → 重建（保留当前lr/wd）
            self.flush_grad_accum()   # 先结算未完成的累积梯度（否则换优化器后梯度丢失）
            self.optimizer = self._build_optimizer()
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1.0)
        if 'c_puct' in kwargs:
            self.c_puct = kwargs['c_puct']
        if 'num_simulations' in kwargs:
            self.num_simulations = kwargs['num_simulations']

