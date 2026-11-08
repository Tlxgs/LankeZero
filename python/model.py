"""KataGo-lite policy/value/ownership network.

Input contract
--------------
``x``: (B, INPUT_CHANNELS, n, n) spatial planes.  These contain only
*spatially varying* features; komi/rules/global scalars are **not** planes.
``g``: (B, GLOBAL_DIM) global vector.  Today GLOBAL_DIM == 1 (signed komi),
but every layer reads the dimension from the config so it can grow later.

Output contract is unchanged:
``policy`` (B, n*n+1), ``value`` (B, 1), ``ownership_raw`` (B, 1, n, n),
``win_logit`` (B, 1).

The trunk follows KataGo's main ideas: 5x5 stem, pre-activation residual
blocks, global-pooling bias in only a few blocks, and 1x1 policy/value heads
with a global bias.  It is fully convolutional and has no size-specific
parameters, so 7..23 and hollow boards use the same weights.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparams import (BOARD_SIZE, INPUT_CHANNELS, GLOBAL_DIM,
                         CHANNELS, HEAD_CHANNELS, NUM_RES_BLOCKS,
                         INPUT_KERNEL, GPOOL_BLOCKS, DROPOUT_RATE,
                         CH_BOUNDARY, GLOBAL_INIT_STD)

# Inference/training performance flags (global, same as before).
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


class GlobalPoolBias(nn.Module):
    """KataGo-style global-pooling bias: mean+max -> FC -> per-channel bias.

    The last FC is zero-initialized, so the module starts as identity and the
    trunk can learn to use global context without destabilizing early training.
    ``pool_channels`` allows pooling a prefix of channels (KataGo's cpool).
    """

    def __init__(self, channels, hidden=None, pool_channels=None):
        super().__init__()
        self.pool_channels = int(pool_channels or channels)
        self.full_pool = (self.pool_channels == int(channels))
        h = int(hidden or max(8, channels // 4))
        self.fc1 = nn.Linear(2 * self.pool_channels, h)
        self.fc2 = nn.Linear(h, channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # Avoid a channel Slice when the whole tensor is pooled: TensorRT 8.x
        # rejects Slice nodes whose axes are not constant in a dynamic-batch graph.
        g = x if self.full_pool else x[:, :self.pool_channels]
        s = torch.cat([g.mean(dim=(2, 3)), g.amax(dim=(2, 3))], dim=1)
        return x + self.fc2(F.relu(self.fc1(s))).view(x.shape[0], -1, 1, 1)


class GlobalFeatureBias(nn.Module):
    """Global vector -> per-channel bias, broadcast over the board.

    This is the *only* path for komi/rules.  Keeping it separate from ``x``
    means a future vector (komi parity, rules, board-size scalars, ...) only
    changes GLOBAL_DIM and the encoder, never the spatial channel layout.
    """

    def __init__(self, global_dim, channels, hidden=None):
        super().__init__()
        h = int(hidden or max(8, channels // 4))
        self.fc1 = nn.Linear(int(global_dim), h)
        self.fc2 = nn.Linear(h, channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, g):
        return self.fc2(F.relu(self.fc1(g))).view(g.shape[0], -1, 1, 1)


class RMSNorm2d(nn.Module):
    """Global RMSNorm over (C,H,W), per-channel affine.

    Kept from the previous architecture: no running statistics, so training and
    inference use exactly the same normalization, and a per-channel constant
    bias (e.g. from gpool) is not removed as it would be by InstanceNorm.
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        rms = x.pow(2).mean(dim=(1, 2, 3), keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def _make_norm(channels):
    return RMSNorm2d(channels)


class PreActResBlock(nn.Module):
    """Pre-activation residual block, identity skip (no 1x1 bottleneck)."""

    def __init__(self, channels, gpool=False, dropout_rate=0.0):
        super().__init__()
        self.norm1 = _make_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gpool = GlobalPoolBias(channels) if gpool else None
        self.norm2 = _make_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        # Bake the 0.5 residual scale into conv2's initialization instead of a
        # runtime `x + 0.5*y`.  A standalone 0.5 constant becomes a TensorRT
        # CONSTANT weight, and REFIT then demands that constant on every refit.
        with torch.no_grad():
            self.conv2.weight.mul_(0.5)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        y = F.relu(self.norm1(x))
        y = self.conv1(y)
        if self.gpool is not None:
            y = self.gpool(y)
        y = F.relu(self.norm2(y))
        y = self.conv2(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return x + y


class PolicyValueNet(nn.Module):
    FORMAT_VERSION = 13
    ARCHITECTURE = 'go-katago-lite-v11'

    def __init__(self, board_size=BOARD_SIZE, input_channels=INPUT_CHANNELS,
                 global_dim=GLOBAL_DIM, channels=CHANNELS,
                 num_res_blocks=NUM_RES_BLOCKS, head_channels=None,
                 dropout_rate=DROPOUT_RATE, input_kernel=INPUT_KERNEL,
                 gpool_blocks=GPOOL_BLOCKS, boundary_channel=None):
        super().__init__()
        self.board_size = int(board_size)
        self.input_channels = int(input_channels)
        self.global_dim = int(global_dim)
        # 哪一路是「边界/有效点」通道（go=8）。只用来屏蔽洞里的归属输出；
        # 其他棋种（五子棋）把它放在自己的通道位置，没有洞时该通道恒 >= 0。
        self.boundary_channel = int(CH_BOUNDARY if boundary_channel is None
                                    else boundary_channel)
        self.channels = int(channels)
        self.num_res_blocks = int(num_res_blocks)
        self.num_blocks = self.num_res_blocks
        self.head_channels = int(head_channels or HEAD_CHANNELS)
        self.dropout_rate = float(dropout_rate)
        self.input_kernel = int(input_kernel)
        self.gpool_blocks = tuple(int(i) for i in gpool_blocks)

        pad = self.input_kernel // 2
        self.conv_input = nn.Conv2d(self.input_channels, self.channels,
                                    self.input_kernel, padding=pad, bias=False)
        self.global_bias = GlobalFeatureBias(self.global_dim, self.channels)
        # 全局特征的**直连残差**：out += g @ W。线性层的梯度与自身权重无关，
        # 所以用正常小随机初始化即可，不需要靠 0 初始化来保持恒等 ——
        # 初始幅度很小（std=GLOBAL_INIT_STD），对主干的影响可忽略。
        self.global_skip = nn.Linear(self.global_dim, self.channels, bias=False)
        nn.init.normal_(self.global_skip.weight, std=float(GLOBAL_INIT_STD))

        gp = set(self.gpool_blocks)
        self.blocks = nn.ModuleList([
            PreActResBlock(self.channels, gpool=(i + 1) in gp,
                           dropout_rate=self.dropout_rate)
            for i in range(self.num_blocks)
        ])
        self.norm_out = _make_norm(self.channels)

        # ---------------- 统一头部（三个头） ----------------
        # 一个共享 1x1 头塔把主干 128 通道变成同宽的 128 通道头特征，三个头读同一份
        # 特征。全局偏置只保留一条低秩支路 head_bias1/2(p0)，先加进头特征再 norm。
        # 每个头都是同一种形式的读出：
        #   policy : pol_h 1x1 128->48 -> pol_out 1x1 48->1（每点 1 logit）+ pol_pass(p)
        #   own    : own_h 1x1 128->48 -> own_out 1x1 48->1 + own_global(p)
        #   win    : p ->128 ->64 ->1
        self.head_width = int(self.channels)
        hw = self.head_width
        self.head_conv = nn.Conv2d(self.channels, hw, 1, bias=False)
        self.head_norm = _make_norm(hw)
        self.head_bias1 = nn.Linear(2 * hw, hw // 2)
        self.head_bias2 = nn.Linear(hw // 2, hw)

        ro = hw // 2                                # 每点读出宽度：64（128 时）
        self.pol_h = nn.Conv2d(hw, ro, 1, bias=False)
        self.pol_out = nn.Conv2d(ro, 1, 1, bias=True)
        self.pol_pass = nn.Linear(2 * hw, 1)

        self.own_h = nn.Conv2d(hw, ro, 1, bias=False)
        self.own_out = nn.Conv2d(ro, 1, 1, bias=True)
        self.own_global = nn.Linear(2 * hw, 1)

        self.win_fc1 = nn.Linear(2 * hw, hw)
        self.win_fc2 = nn.Linear(hw, hw // 2)
        self.win_fc3 = nn.Linear(hw // 2, 1)


    def forward(self, x, g=None):
        if g is None:
            g = torch.zeros((x.shape[0], self.global_dim),
                            dtype=x.dtype, device=x.device)
        out = self.conv_input(x)
        out = out + self.global_bias(g) + self.global_skip(g).view(-1, self.channels, 1, 1)
        for block in self.blocks:
            out = block(out)
        out = F.relu(self.norm_out(out))

        # ---------- 统一头部：一个共享 1x1 塔 + 一次池化 ----------
        h = self.head_conv(out)                                   # (B,128,n,n)
        p0 = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        h = F.relu(self.head_norm(h + self.head_bias2(
            F.relu(self.head_bias1(p0))).view(-1, self.head_width, 1, 1)))
        p = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)

        # 策略 / 归属（空间读出）
        p_spatial = self.pol_out(F.relu(self.pol_h(h))).reshape(h.shape[0], -1)
        p_pass = self.pol_pass(p)
        policy = torch.cat([p_spatial, p_pass], dim=1)

        own_local = self.own_out(F.relu(self.own_h(h)))
        own_global = self.own_global(p).view(-1, 1, 1, 1)
        ownership_raw = own_local + own_global
        valid = (x.select(1, self.boundary_channel).unsqueeze(1) >= 0.0).to(ownership_raw.dtype)
        ownership = torch.tanh(ownership_raw) * valid
        value = ownership.sum(dim=(2, 3))

        # 胜率（全局读出）
        w = F.relu(self.win_fc1(p))
        win_logit = self.win_fc3(F.relu(self.win_fc2(w)))
        return policy, value, ownership_raw, win_logit

    def get_policy_value_ownership(self, state, legal_mask=None, device='cpu',
                                   global_features=None):
        self.eval()
        with torch.inference_mode():
            state_tensor = torch.from_numpy(np.asarray(state)).unsqueeze(0).float().to(device)
            gt = None
            if global_features is not None:
                gt = torch.from_numpy(np.asarray(global_features)).reshape(
                    1, self.global_dim).float().to(device)
            legal_tensor = None
            if legal_mask is not None:
                legal_tensor = torch.from_numpy(np.asarray(legal_mask)).float().to(device)
            policy_logits, value, ownership, win_logit = self(state_tensor, gt)
            if legal_tensor is not None:
                policy_logits = policy_logits.masked_fill(legal_tensor == 0, -1e4)
            policy = F.softmax(policy_logits, dim=1)
            ownership = torch.tanh(ownership)
            valid = (state_tensor[:, self.boundary_channel:self.boundary_channel + 1] >= 0.0).float()
            ownership = ownership * valid
            value = value.cpu().numpy()[0, 0]
            policy = policy.cpu().numpy()[0]
            ownership = ownership.cpu().numpy()[0, 0]
            win_logit = float(win_logit.cpu().numpy()[0, 0])
        return policy, value, ownership, win_logit

    def save_model(self, path):
        torch.save({
            'format_version': self.FORMAT_VERSION,
            'architecture': self.ARCHITECTURE,
            'model_state_dict': self.state_dict(),
            'board_size': self.board_size,
            'input_channels': self.input_channels,
            'global_dim': self.global_dim,
            'channels': self.channels,
            'head_channels': self.head_channels,
            'num_res_blocks': self.num_res_blocks,
            'dropout_rate': self.dropout_rate,
            'input_kernel': self.input_kernel,
            'gpool_blocks': list(self.gpool_blocks),
            'boundary_channel': self.boundary_channel,
        }, path)

    def freeze_parts(self, spec):
        """Freeze a subset of parameters (training only; inference unaffected).

        Tokens: integer N = blocks[N]; 'gpool' = all GlobalPoolBias branches;
        'own' = value tower + ownership outputs; 'policy' = policy head;
        'win' = win MLP.  Prefix 'only:' turns the list into an allow-list.
        """
        names = set()
        substrs = []
        only = False
        if isinstance(spec, str):
            tokens = [t.strip() for t in spec.split(',') if t.strip()]
        else:
            tokens = list(spec)
        for t in tokens:
            if t.startswith('only:'):
                only = True
                t = t[len('only:'):]
            if t == 'gpool':
                substrs.append('.gpool.')
            elif t == 'own':
                names.update(['own_out', 'own_global'])
            elif t == 'policy':
                names.update(['pol_out', 'pol_pass'])
            elif t == 'win':
                names.update(['win_fc1', 'win_fc2'])
            elif t == 'head':
                names.update(['head_conv', 'head_norm', 'head_bias1', 'head_bias2'])
            else:
                try:
                    idx = int(t)
                except (TypeError, ValueError):
                    raise ValueError('未知冻结令牌 %r（支持：残差块序号 N、gpool、head、own、'
                                     'policy、win，可加 only: 前缀）' % t)
                if not (0 <= idx < len(self.blocks)):
                    raise ValueError('冻结块序号 %d 越界（当前共 %d 个残差块）'
                                     % (idx, len(self.blocks)))
                names.add('blocks.%d' % idx)
        n_frozen = 0
        for name, param in self.named_parameters():
            listed = (any(name == n or name.startswith(n + '.') for n in names)
                      or any(s in name for s in substrs))
            frozen = (not listed) if only else listed
            param.requires_grad = not frozen
            if frozen:
                n_frozen += param.numel()
        return n_frozen

    def reset_norm(self):
        """Reset all RMSNorm affine parameters to weight=1, bias=0."""
        n = 0
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, RMSNorm2d):
                    m.weight.data.fill_(1.0)
                    m.bias.data.fill_(0.0)
                    n += 1
        print(f'[模型] 已重置 {n} 个 RMSNorm 层（weight=1, bias=0）')
        return n

    @staticmethod
    def load_model(path, device='cpu', reset_norm=False):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        arch = checkpoint.get('architecture')
        if arch != PolicyValueNet.ARCHITECTURE:
            raise ValueError(
                '权重架构 %r 与当前架构 %r 不兼容：\n%s\n'
                '旧版 model.pt 请先用 tools/upgrade_model.py 升级；'
                '12 通道的老模型已备份在 legacy/model_v1_12ch.pt。'
                % (arch, PolicyValueNet.ARCHITECTURE, path))
        model = PolicyValueNet(
            board_size=checkpoint.get('board_size', BOARD_SIZE),
            input_channels=checkpoint.get('input_channels', INPUT_CHANNELS),
            global_dim=checkpoint.get('global_dim', GLOBAL_DIM),
            channels=checkpoint.get('channels', CHANNELS),
            num_res_blocks=checkpoint.get('num_res_blocks', NUM_RES_BLOCKS),
            head_channels=checkpoint.get('head_channels', HEAD_CHANNELS),
            dropout_rate=checkpoint.get('dropout_rate', DROPOUT_RATE),
            input_kernel=checkpoint.get('input_kernel', INPUT_KERNEL),
            gpool_blocks=checkpoint.get('gpool_blocks', GPOOL_BLOCKS),
            boundary_channel=checkpoint.get('boundary_channel', CH_BOUNDARY),
        )
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f'权重与当前架构不兼容：{path}\n{e}\n'
                '新架构不接受旧 block/通道布局，请从当前 model.pt 重新训练。') from e
        with torch.no_grad():
            norm_means = [m.weight.data.mean().item() for m in model.modules()
                          if isinstance(m, RMSNorm2d)]
        if norm_means:
            norm_mean = float(np.mean(norm_means))
            if norm_mean < 0.8:
                print(f'[模型] 提示: 归一化 weight 均值={norm_mean:.2f}'
                      '（<0.8，疑似被 weight decay 衰减；可用 reset_norm=True 重置）')
        if reset_norm:
            model.reset_norm()
        model.to(device)
        return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PolicyValueNet()
    print(f"参数量: {count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")
    print(f"结构: 空间{INPUT_CHANNELS}通道 + 全局{GLOBAL_DIM}维 / "
          f"{CHANNELS}通道×{NUM_RES_BLOCKS}块 / gpool={GPOOL_BLOCKS}")
    x = torch.randn(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    g = torch.randn(1, GLOBAL_DIM)
    p, v, o, wl, sc = model(x, g)
    print(f"策略输出: {p.shape}, 价值输出: {v.shape}, 归属输出: {o.shape}, "
          f"胜率logit: {wl.shape}, 分数logits: {sc.shape}")

    state = np.random.randn(INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float32)
    gv = np.zeros(GLOBAL_DIM, dtype=np.float32)
    policy, value, ownership, win_logit = model.get_policy_value_ownership(
        state, global_features=gv)
    print(f"策略维度: {policy.shape}, 价值: {value:.4f}, 归属: {ownership.shape}, "
          f"胜率logit: {win_logit:.4f}")
