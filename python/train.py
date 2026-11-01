"""本机单进程训练：SelfPlayTrainer + 紧凑回放 + 唯一 model.pt。

旧 pkl / ONNX / 多 worker 训练路径已归档到 legacy/server。
训练只加载当前 10 通道架构（旧 9 通道自动小初始化迁移）；全项目只有一个权重 model.pt，保存时原地原子覆盖。
"""
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from hyperparams import (BOARD_SIZE, NUM_SIMULATIONS, C_PUCT, BATCH_SIZE,
                         GRAD_ACCUM_STEPS, LEARNING_RATE, EYE_LR, GPOOL_LR,
                         CH_OWN_EYES, CH_OPP_EYES, CH_BOUNDARY,
                         WEIGHT_DECAY, MOMENTUM,
                         OPTIMIZER_NAME, WARMUP_STEPS, ENTROPY_TARGET,
                         ENTROPY_HINGE_WEIGHT, OWN_WEIGHT, WIN_WEIGHT,
                         GRAD_CLIP_NORM, DATA_DIR, MODEL_PATH, TRAINING_DATA_MAX,
                         PYTORCH_CUDA_ALLOC_CONF, MCTS_CAP, BATCH_SIZE_MCTS)
from model import PolicyValueNet, GlobalPoolBias
from replay import TrainingData
from trt_engine import make_evaluator

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", PYTORCH_CUDA_ALLOC_CONF)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


class SelfPlayTrainer:
    LOSS_FIELDS = ("policy", "own", "entropy", "win", "total")
    LOSS_N = len(LOSS_FIELDS)

    def __init__(self, model_path=MODEL_PATH, board_size=BOARD_SIZE,
                 device="cuda", num_simulations=NUM_SIMULATIONS,
                 data_dir=DATA_DIR, c_puct=C_PUCT,
                 learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                 optimizer_name=OPTIMIZER_NAME, momentum=MOMENTUM,
                 freeze_parts=None, reset_norm=False, data_max=None,
                 evaluator=None, use_tensorrt=False, trt_engine_dir=None,
                 log=None):
        self.board_size = board_size
        self.device = device
        self.num_simulations = num_simulations
        self.data_dir = data_dir
        self.c_puct = c_puct
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name.lower()
        self.momentum = momentum
        self.use_tensorrt = bool(use_tensorrt)
        self.trt_engine_dir = trt_engine_dir
        self._log = log or (lambda message: print(message, flush=True))
        os.makedirs(data_dir, exist_ok=True)
        self.model_path = os.path.abspath(model_path)

        self.model = PolicyValueNet.load_model(self.model_path, reset_norm=reset_norm).to(device)
        self.frozen_params_count = 0
        if freeze_parts:
            self.frozen_params_count = self.model.freeze_parts(freeze_parts)
            print("[训练] 冻结参数: %s（%d 个参数）" % (freeze_parts, self.frozen_params_count))
        self.data_buffer = TrainingData(max_size=data_max or TRAINING_DATA_MAX,
                                        data_dir=data_dir, board_size=board_size)
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: 1.0)
        self.game_count = 0
        self.train_count = 0
        self.warmup_steps = WARMUP_STEPS
        self.grad_accum_steps = max(1, int(GRAD_ACCUM_STEPS))
        self._accum_steps_done = 0
        self._opt_steps = 0
        self.evaluator = evaluator or make_evaluator(
            self.model, board_size=board_size, device=device,
            max_batch=BATCH_SIZE_MCTS, use_tensorrt=self.use_tensorrt,
            trt_engine_dir=trt_engine_dir, log=self._log)

        # 全项目只有一个权重：加载 model.pt，训练后原地原子覆盖 model.pt。
        self.last_saved = self.model_path
        print("[模型] 唯一权重: %s（训练后原地更新）" % self.model_path)

    def _build_optimizer(self):
        decay, no_decay = [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        groups = [
            {"params": decay, "weight_decay": self.weight_decay, "wd_enabled": True},
            {"params": no_decay, "weight_decay": 0.0, "wd_enabled": False},
        ]
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(groups, lr=self.learning_rate, momentum=self.momentum)
        return torch.optim.Adam(groups, lr=self.learning_rate)

    @staticmethod
    def _augment_batch(states, policies, values, ownerships, own_masks, board_size):
        """只做 D4 八对称几何增强，不做黑白互换。策略 pass 位保持不变。

        为什么删掉黑白互换：贴目固定给白方，棋局对黑白并不对称。
        把颜色互换后，同一局面的「当前行棋方」从欠贴目变成含贴目（或反过来），
        最优策略会随之改变；而增强却把原局面的 MCTS 策略目标原样搬过去，
        等于给一半样本灌了错误标签。贴目越大（9 路 7.5 目尤其明显）危害越大。
        D4 旋转/镜像不改变贴目归属，是严格对称，保留。
        """
        batch_size = states.shape[0]
        n = board_size * board_size
        transforms = [
            (lambda x: x, lambda p: p),
            (lambda x: np.flip(x, axis=1),
             lambda p: np.concatenate([p[:n].reshape(board_size, board_size)[::-1, :].flatten(), p[-1:]])),
            (lambda x: np.flip(x, axis=2),
             lambda p: np.concatenate([p[:n].reshape(board_size, board_size)[:, ::-1].flatten(), p[-1:]])),
            (lambda x: np.rot90(x, k=1, axes=(1, 2)),
             lambda p: np.concatenate([np.rot90(p[:n].reshape(board_size, board_size), k=1).flatten(), p[-1:]])),
            (lambda x: np.rot90(x, k=2, axes=(1, 2)),
             lambda p: np.concatenate([np.rot90(p[:n].reshape(board_size, board_size), k=2).flatten(), p[-1:]])),
            (lambda x: np.rot90(x, k=3, axes=(1, 2)),
             lambda p: np.concatenate([np.rot90(p[:n].reshape(board_size, board_size), k=3).flatten(), p[-1:]])),
            (lambda x: np.transpose(x, (0, 2, 1)),
             lambda p: np.concatenate([p[:n].reshape(board_size, board_size).T.flatten(), p[-1:]])),
            (lambda x: np.flip(np.transpose(x, (0, 2, 1)), axis=(1, 2)),
             lambda p: np.concatenate([np.flip(p[:n].reshape(board_size, board_size).T, axis=(0, 1)).flatten(), p[-1:]])),
        ]
        own_funcs = [
            lambda x: x,
            lambda x: np.flip(x, axis=0),
            lambda x: np.flip(x, axis=1),
            lambda x: np.rot90(x, k=1),
            lambda x: np.rot90(x, k=2),
            lambda x: np.rot90(x, k=3),
            lambda x: x.T,
            lambda x: np.flip(x.T, axis=(0, 1)),
        ]
        out_s = np.empty_like(states, dtype=np.float32)
        out_p = np.empty_like(policies, dtype=np.float32)
        out_v = np.array(values, dtype=np.float32, copy=True)
        out_o = np.empty_like(ownerships, dtype=np.int8)
        out_m = np.array(own_masks, dtype=np.uint8, copy=True)
        for i in range(batch_size):
            t = np.random.randint(len(transforms))
            out_s[i] = transforms[t][0](states[i])
            out_p[i] = transforms[t][1](policies[i])
            out_o[i] = own_funcs[t](ownerships[i])
        return out_s, out_p, out_v, out_o, out_m

    @staticmethod
    def _ownership_loss_mask(own_target, own_mask, valid):
        """归属损失的样本掩码。

        可落子点（valid=True）只训练非 0 标签，中立空点不强行给 0；
        洞（valid=False）也纳入损失，目标固定为 0，让归属头学会输出 0。
        """
        return own_mask & (((own_target != 0) & valid) | (~valid))

    def train_step(self, batch_size=BATCH_SIZE):
        if len(self.data_buffer) < batch_size:
            return (None,) * self.LOSS_N
        states, policies, values, ownerships, _kos, own_masks = self.data_buffer.sample(batch_size)
        states, policies, values, ownerships, own_masks = self._augment_batch(
            states, policies, values, ownerships, own_masks, self.board_size)
        states_t = torch.from_numpy(states).to(self.device)
        policies_t = torch.from_numpy(policies).to(self.device)

        self.model.train()
        policy_logits, pred_values, own_logits, win_logits = self.model(states_t)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policies_t * log_probs).sum(dim=1).mean()
        probs = F.softmax(policy_logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        own_logit = own_logits.squeeze(1)
        own_target = torch.from_numpy(ownerships).to(self.device).float()
        own_mask = torch.from_numpy(own_masks).to(self.device).bool().view(-1, 1, 1)
        # 边界通道 -1 = 洞。可落子点只训练非 0 标签；洞内也参与损失，目标 0。
        valid = torch.from_numpy(states[:, CH_BOUNDARY] >= 0.0).to(self.device).bool()
        mask = self._ownership_loss_mask(own_target, own_mask, valid)
        if not mask.any():
            own_loss = torch.zeros((), device=self.device)
        else:
            # 洞内目标强制 0，避免旧数据/异常标签把洞教出非 0 归属。
            own_loss_target = torch.where(valid, own_target,
                                          torch.zeros_like(own_target))
            own_loss = F.binary_cross_entropy_with_logits(
                2.0 * own_logit[mask], (own_loss_target[mask] + 1.0) / 2.0)

        win_target = (torch.from_numpy(values).to(self.device).float() + 1.0) / 2.0
        win_loss = F.binary_cross_entropy_with_logits(2.0 * win_logits.squeeze(1), win_target)
        excess = (entropy - ENTROPY_TARGET).clamp(min=0.0)
        entropy_penalty = ENTROPY_HINGE_WEIGHT * excess * excess
        total_loss = policy_loss + entropy_penalty + OWN_WEIGHT * own_loss + WIN_WEIGHT * win_loss

        if self._accum_steps_done == 0:
            self.optimizer.zero_grad()
        (total_loss / self.grad_accum_steps).backward()
        self._accum_steps_done += 1
        if self._accum_steps_done >= self.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
            self._boost_eye_grads()
            self._boost_gpool_grads()
            self.optimizer.step()
            self._accum_steps_done = 0
            self._opt_steps += 1
            self._apply_warmup()
        return (policy_loss.item(), own_loss.item(), entropy.item(),
                win_loss.item(), total_loss.item())

    def _boost_eye_grads(self):
        if EYE_LR <= 0 or self.learning_rate <= 0:
            return
        w = self.model.conv_input.weight
        if w.grad is None:
            return
        mult = EYE_LR / self.learning_rate
        w.grad[:, CH_OWN_EYES] *= mult
        w.grad[:, CH_OPP_EYES] *= mult

    def _boost_gpool_grads(self):
        if GPOOL_LR <= 0 or self.learning_rate <= 0:
            return
        mult = GPOOL_LR / self.learning_rate
        for m in self.model.modules():
            if not isinstance(m, GlobalPoolBias):
                continue
            for p in (m.fc1.weight, m.fc1.bias, m.fc2.weight, m.fc2.bias):
                if p.grad is not None:
                    p.grad *= mult

    def _apply_warmup(self):
        if self.warmup_steps > 0 and self._opt_steps <= self.warmup_steps:
            scale = self._opt_steps / self.warmup_steps
            for g in self.optimizer.param_groups:
                g["lr"] = self.learning_rate * scale

    def flush_grad_accum(self):
        if getattr(self, "_accum_steps_done", 0) > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
            self._boost_eye_grads()
            self._boost_gpool_grads()
            self.optimizer.step()
            self._accum_steps_done = 0
            self._opt_steps += 1
            self._apply_warmup()

    def save_model(self, tag=None):
        """原地原子更新唯一的 model.pt。

        tag 只保留给旧调用方；不再生成任何新权重文件。先写 .tmp 再 os.replace，
        保存过程中崩溃也不会留下半截 model.pt。
        TensorRT 开启时，保存后只做权重 refit（几十毫秒），不重建引擎。
        """
        path = self.model_path
        tmp = path + ".tmp"
        self.model.save_model(tmp)
        os.replace(tmp, path)
        self.last_saved = path
        torch.cuda.empty_cache()
        self._refit_tensorrt()
        return path

    def _refit_tensorrt(self):
        ev = getattr(self, "evaluator", None)
        if ev is None or not hasattr(ev, "refit_from_model"):
            return
        try:
            ev.refit_from_model(self.model)
        except Exception as exc:
            self._log("[TensorRT] refit 异常: %s" % exc)

    def load_training_data(self, filename=None, mode="newest"):
        if filename is not None:
            return self.data_buffer.load_single(filename)
        return self.data_buffer.load_all(mode=mode)

    def save_training_data(self, payload):
        return self.data_buffer.save_game(payload)

    def update_hyperparameters(self, **kwargs):
        if "learning_rate" in kwargs:
            self.learning_rate = kwargs["learning_rate"]
            for g in self.optimizer.param_groups:
                g["lr"] = self.learning_rate
        if "weight_decay" in kwargs:
            self.weight_decay = kwargs["weight_decay"]
            for g in self.optimizer.param_groups:
                if g.get("wd_enabled", True):
                    g["weight_decay"] = self.weight_decay
        rebuild = False
        if "optimizer_name" in kwargs and kwargs["optimizer_name"].lower() != self.optimizer_name:
            self.optimizer_name = kwargs["optimizer_name"].lower()
            rebuild = True
        if "momentum" in kwargs and self.optimizer_name == "sgd":
            self.momentum = kwargs["momentum"]
            rebuild = True
        if rebuild:
            self.flush_grad_accum()
            self.optimizer = self._build_optimizer()
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1.0)
        if "c_puct" in kwargs:
            self.c_puct = kwargs["c_puct"]
        if "num_simulations" in kwargs:
            self.num_simulations = kwargs["num_simulations"]