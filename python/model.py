import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---- 推理/训练性能优化（全局生效）----
torch.backends.cudnn.benchmark = True 
torch.set_float32_matmul_precision('high')

# 超参数（统一配置见 hyperparams.py）
from hyperparams import (BOARD_SIZE, INPUT_CHANNELS, CHANNELS, HEAD_CHANNELS, NUM_RES_BLOCKS,
                         DROPOUT_RATE, CH_BOUNDARY)


class GlobalPoolBias(nn.Module):
    """全盘统计量 → 逐通道偏置 → 广播回每个空间位置（KataGo 的 gpool，论文第二强技巧 1.60×）。

    统计量只取 **mean + max** 两个：棋盘固定 19 路，KataGo 第三个「按尺寸缩放的均值」是为跨
    9~19 路泛化用的，对我们是纯冗余（见 AGENT.md 的 KataGo 对照表）。

    为什么以前不能用：旧归一化用「减逐通道均值」，会把逐通道常数偏置的**共同部分**抵消掉
    （实测梯度被压小约 100 倍），而推理时又不抵消 ⇒ 训推不一致。换成跨通道求 RMS 的
    RMSNorm 后没有减均值这一步，加性偏置的梯度路径才通（_dl/_check_gpool_grad.py 可复验）。

    末层零初始化 ⇒ 初始输出与不接本模块时**逐位相同** ⇒ 旧权重可无痛续训。梯度不会因此
    断掉：这是残差加法，主路径的梯度不经过 fc，而 fc 自身的梯度 ∝ 上游梯度 × 池化统计量，
    两者都非零。注意与「末层零初始化会把前层梯度掐断」那种情形不同 —— 那种是**唯一通路**
    被断开，这里只是旁路初值为零。
    """

    def __init__(self, channels, hidden=None):
        super().__init__()
        h = hidden or max(1, channels // 4)
        self.fc1 = nn.Linear(2 * channels, h)
        self.fc2 = nn.Linear(h, channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        s = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)   # (B, 2C)
        return x + self.fc2(F.relu(self.fc1(s))).view(b, c, 1, 1)


class RMSNorm2d(nn.Module):
    """LayerNorm 风格的 RMSNorm：跨 (C,H,W) 求均方根，逐通道仿射。

    刻意**不**按通道求 RMS —— 那等价于 InstanceNorm，会把「整通道同一个常数」在归一化那一步
    抹掉。跨通道求 RMS 时，单通道的常数只与整体尺度竞争，不会被单独消掉。

    没有 running 统计量 ⇒ 训练与推理行为一致（不存在两套统计量的差异）。
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        rms = x.pow(2).mean(dim=(1, 2, 3), keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, channels, dropout_rate):
        super().__init__()
        self.norm1 = RMSNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = RMSNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bottleneck = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm_bottle = RMSNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.gpool = GlobalPoolBias(channels)

    def forward(self, x):
        res = x
        out = self.norm1(x)
        out = F.leaky_relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        res = self.bottleneck(res)
        res = self.norm_bottle(res)
        out = self.dropout(out)
        out = res + out
        # 放在残差相加之后：偏置进入残差流，交给后续层去扩散利用（KataGo 也是加进 block 输出）。
        # 不在归一化「之前」注入，是为了让偏置少被归一化一次。
        return self.gpool(out)


class PolicyValueNet(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, input_channels=INPUT_CHANNELS,
                 channels=CHANNELS, num_res_blocks=NUM_RES_BLOCKS,
                 head_channels=None, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.head_channels = head_channels if head_channels else HEAD_CHANNELS
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels

        # 输入卷积
        self.conv_input = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm_input = RMSNorm2d(channels)

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResBlock(channels, dropout_rate) for _ in range(num_res_blocks)
        ])

        # ---------- 三个头（KataGo 风格轻量塔）----------
        # 共享头塔：归属头与胜率头共用第一层 3×3 卷积（对应 KataGo ownership/value head 共用 conv tower）
        self.head_conv = nn.Conv2d(channels, self.head_channels, kernel_size=3, padding=1, bias=False)
        self.head_norm = RMSNorm2d(self.head_channels)

        # 策略头（独立 2 层 3×3 塔）：逐点 logit + 全局池化 pass logit
        self.policy_conv = nn.Conv2d(channels, self.head_channels, kernel_size=3, padding=1, bias=False)
        self.policy_norm = RMSNorm2d(self.head_channels)
        self.policy_conv2 = nn.Conv2d(self.head_channels, self.head_channels, kernel_size=3, padding=1, bias=False)
        self.policy_norm2 = RMSNorm2d(self.head_channels)
        self.conv_policy_out = nn.Conv2d(self.head_channels, 1, kernel_size=1, bias=True)
        self.fc_policy_pass = nn.Linear(self.head_channels, 1, bias=True)

        # 归属头（共享塔 + 1 层加深；KataGo：逐点本地归属 + 全局偏置）
        self.own_conv = nn.Conv2d(self.head_channels, self.head_channels, kernel_size=3, padding=1, bias=False)
        self.own_norm = RMSNorm2d(self.head_channels)
        self.conv_own_out = nn.Conv2d(self.head_channels, 1, kernel_size=1, bias=True)
        self.fc_own_global = nn.Linear(self.head_channels, 1, bias=True)

        # 胜率头（共享塔 + 1 层加深；全局平均池化 → 2 层 MLP → 胜率 logit）
        self.win_conv = nn.Conv2d(self.head_channels, self.head_channels, kernel_size=3, padding=1, bias=False)
        self.win_norm = RMSNorm2d(self.head_channels)
        self.win_fc1 = nn.Linear(self.head_channels, self.head_channels, bias=True)
        self.win_fc2 = nn.Linear(self.head_channels, 1, bias=True)

    def forward(self, x):
        # 共享特征
        out = F.leaky_relu(self.norm_input(self.conv_input(x)))
        for res in self.res_blocks:
            out = res(out)

        # ---------- 策略头（独立 2 层 3×3 塔）----------
        policy = F.leaky_relu(self.policy_norm(self.policy_conv(out)))                  # (B, HEAD_CHANNELS, H, W)
        policy = F.leaky_relu(self.policy_norm2(self.policy_conv2(policy)))             # (B, HEAD_CHANNELS, H, W)
        policy_spatial = self.conv_policy_out(policy).reshape(policy.size(0), -1)  # (B, 361) 逐点 logit
        policy_pass = self.fc_policy_pass(                                      # (B, 1) pass logit（全局池化）
            F.adaptive_avg_pool2d(policy, 1).flatten(1))
        policy = torch.cat([policy_spatial, policy_pass], dim=1)                # (B, 362)

        # ---------- 共享头塔（归属头/胜率头共用第一层，KataGo 风格）----------
        head = F.leaky_relu(self.head_norm(self.head_conv(out)))                        # (B, HEAD_CHANNELS, H, W)

        # ---------- 领地头（价值 = 归属求和；共享塔 + 1 层加深，本地 + 全局偏置）----------
        own = F.leaky_relu(self.own_norm(self.own_conv(head)))                          # (B, HEAD_CHANNELS, H, W)
        own_local = self.conv_own_out(own)                                      # (B, 1, H, W) 逐点本地归属
        own_global = self.fc_own_global(                                        # (B, 1) 全局偏置（允许整体偏移）
            F.adaptive_avg_pool2d(own, 1).flatten(1))
        ownership_raw = own_local + own_global.view(-1, 1, 1, 1)                # (B, 1, H, W) 未压缩
        # 边界通道 -1 = 洞。洞内不参与 value 求和；raw logit 不再硬置 0，
        # 训练时把洞内目标也纳入归属损失（目标 0），让头自己学会输出 0。
        # hollow=0 时 valid 全 1，行为与旧模型逐位一致。
        valid = (x.select(1, CH_BOUNDARY).unsqueeze(1) >= 0.0).to(ownership_raw.dtype)
        ownership = torch.tanh(ownership_raw) * valid                           # (B, 1, H, W)，洞内 0
        value = ownership.sum(dim=(2, 3))                                       # (B, 1) 期望领地目差（当前视角）

        # ---------- 胜率头（共享塔 + 1 层加深；全局平均池化 → 2 层 MLP；tanh 后=当前玩家胜率 ±1，MCTS 用 logit）----------
        win = F.leaky_relu(self.win_norm(self.win_conv(head)))                          # (B, HEAD_CHANNELS, H, W)
        win = F.leaky_relu(self.win_fc1(F.adaptive_avg_pool2d(win, 1).flatten(1)))    # (B, HEAD_CHANNELS)
        win_logit = self.win_fc2(win)                                           # (B, 1) logit

        return policy, value, ownership_raw, win_logit

    def get_policy_value_ownership(self, state, legal_mask=None, device='cpu'):
        self.eval()
        with torch.inference_mode():
            state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)
            legal_tensor = None
            if legal_mask is not None:
                legal_tensor = torch.from_numpy(legal_mask).float().to(device)
            policy_logits, value, ownership, win_logit = self(state_tensor)
            if legal_tensor is not None:
                policy_logits = policy_logits.masked_fill(legal_tensor == 0, -1e4)
            policy = F.softmax(policy_logits, dim=1)
            ownership = torch.tanh(ownership)                # (1,1,H,W) ∈(-1,1)
            valid = (state_tensor[:, CH_BOUNDARY:CH_BOUNDARY + 1] >= 0.0).float()
            ownership = ownership * valid                    # 洞内强制 0
            value = value.cpu().numpy()[0, 0]
            policy = policy.cpu().numpy()[0]
            ownership = ownership.cpu().numpy()[0, 0]
            win_logit = float(win_logit.cpu().numpy()[0, 0])   # 胜率头 logit（tanh 后 = 当前玩家胜率 ±1）
        return policy, value, ownership, win_logit

    def save_model(self, path):
        torch.save({
            'format_version': self.FORMAT_VERSION,
            'architecture': self.ARCHITECTURE,
            'model_state_dict': self.state_dict(),
            'board_size': self.board_size,
            'input_channels': self.input_channels,
            'channels': self.channels,
            'head_channels': self.head_channels,
            'num_res_blocks': self.num_res_blocks,
            'dropout_rate': self.dropout_rate,
        }, path)

    def freeze_parts(self, spec):
        """按规格冻结部分网络参数（仅训练时生效，推理/导出不受影响）。
        spec: 逗号分隔的令牌——
          整数 N    = res_blocks[N]（从 0 起）；
          'gpool'   = 各残差块内的全盘偏置分支 GlobalPoolBias；
          'own'     = 归属头（head_conv/head_norm + own_conv/own_norm + conv_own_out + fc_own_global；
                      注意 head_conv/head_norm 为共享头塔，与胜率头共用，只冻它会让胜率头输入
                      变成静态特征，而 win_conv/win_fc* 仍可训练）；
          'policy'  = 策略头（policy_conv/policy_norm/policy_conv2/policy_norm2 + conv_policy_out + fc_policy_pass）；
          'win'     = 胜率头（win_conv/win_norm + win_fc1/win_fc2，共享头塔不受影响）。

        任一令牌带前缀 'only:' 即把整条规格切成**白名单**语义：只有列出的部分可训练，其余全冻。
        'only:gpool' = 只训 gpool，主干与三个头一律不动（若不给前缀则恰好相反 —— 只冻 gpool）。

        示例：'0,1,2,3,5,7'（冻结这些残差块）、'0,own,policy'（冻结块0+归属头+策略头）、
        'only:gpool'（只训 gpool）。
        未知令牌/越界序号抛 ValueError。返回冻结的参数量。"""
        names = set()
        # gpool 嵌在每个残差块内部（res_blocks.N.gpool.*），没有顶层共同前缀，按子串认。
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
                names.update(['head_conv', 'head_norm', 'own_conv', 'own_norm',
                              'conv_own_out', 'fc_own_global'])
            elif t == 'policy':
                names.update(['policy_conv', 'policy_norm', 'policy_conv2', 'policy_norm2',
                              'conv_policy_out', 'fc_policy_pass'])
            elif t == 'win':
                names.update(['win_conv', 'win_norm', 'win_fc1', 'win_fc2'])
            else:
                try:
                    idx = int(t)
                except (TypeError, ValueError):
                    raise ValueError('未知冻结令牌 %r（支持：残差块序号 N、gpool、own、policy、win，'
                                     '可加 only: 前缀）' % t)
                if not (0 <= idx < len(self.res_blocks)):
                    raise ValueError('冻结块序号 %d 越界（当前共 %d 个残差块）' % (idx, len(self.res_blocks)))
                names.add('res_blocks.%d' % idx)
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
        """重新初始化全部归一化层：weight=1、bias=0。
        归一化的 affine 缩放会被后续层抵消，对其施加 weight decay 无意义，且会持续缩小有效表达
        （KataGo 惯例：这类 1D 参数不参与衰减）；长期训练后被衰减的可随时用本方法恢复。
        返回重置的归一化层数。"""
        n = 0
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, RMSNorm2d):
                    m.weight.data.fill_(1.0)
                    m.bias.data.fill_(0.0)
                    n += 1
        print(f'[模型] 已重置 {n} 个 RMSNorm 层（weight=1, bias=0）')
        return n

    FORMAT_VERSION = 2
    ARCHITECTURE = 'go-cnn-v1'

    @staticmethod
    def load_model(path, device='cpu', reset_norm=False):
        """加载当前架构，并支持从上一版 9 通道权重增量迁移到 10 通道。

        新增的边界通道（CH_BOUNDARY=9）接在输入末尾：旧 9 通道权重原样复制，
        新通道用 std=0.01 的小初始化，不破坏已有特征；数据无需改动，
        边界通道由棋盘尺寸和 hollow 在 sample/canonical_state 里即时生成。
        除 9→10 外，通道数或键不匹配仍直接报错。
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        sd = dict(checkpoint['model_state_dict'])
        ckpt_in = int(checkpoint.get('input_channels', -1))
        if ckpt_in not in (INPUT_CHANNELS, INPUT_CHANNELS - 1):
            raise ValueError(
                f'权重输入通道 {ckpt_in} 与当前代码 {INPUT_CHANNELS} 不兼容；'
                '只支持上一版 9 通道 → 当前 10 通道的迁移。')
        if ckpt_in != INPUT_CHANNELS:
            old_w = sd.get('conv_input.weight')
            if old_w is None or old_w.shape[1] != ckpt_in:
                raise ValueError('旧权重缺少可迁移的 conv_input.weight')
            new_w = torch.zeros((old_w.shape[0], INPUT_CHANNELS,
                                 old_w.shape[2], old_w.shape[3]),
                                dtype=old_w.dtype, device=old_w.device)
            new_w[:, :ckpt_in].copy_(old_w)
            nn.init.normal_(new_w[:, ckpt_in:], mean=0.0, std=0.01)
            sd['conv_input.weight'] = new_w
            print('[模型] 已把 %d 通道 conv_input 迁移到 %d 通道，新通道 std=0.01'
                  % (ckpt_in, INPUT_CHANNELS))
        model = PolicyValueNet(
            board_size=checkpoint.get('board_size', BOARD_SIZE),
            input_channels=INPUT_CHANNELS,
            channels=checkpoint.get('channels', CHANNELS),
            num_res_blocks=checkpoint.get('num_res_blocks', NUM_RES_BLOCKS),
            head_channels=checkpoint.get('head_channels', HEAD_CHANNELS),
            dropout_rate=checkpoint.get('dropout_rate', DROPOUT_RATE),
        )
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f'权重与当前架构不兼容：{path}\n{e}\n'
                '除 9→10 通道迁移外，不再跳过不匹配的键。') from e
        # 归一化层健康提示：weight decay 会持续衰减它们的 affine 缩放（KataGo 惯例：这类 1D 参数不参与衰减）。
        # 若 mean 明显 <1（如 <0.8），说明已被衰减，建议 reset_norm=True 重置后再训练。
        with torch.no_grad():
            norm_means = [m.weight.data.mean().item() for m in model.modules()
                          if isinstance(m, RMSNorm2d)]
        if norm_means:
            norm_mean = float(np.mean(norm_means))
            if norm_mean < 0.8:
                print(f'[模型] 提示: 归一化 weight 均值={norm_mean:.2f}（<0.8，疑似被 weight decay 衰减；可用 reset_norm=True 重置）')
        if reset_norm:
            model.reset_norm()
        model.to(device)
        return model



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PolicyValueNet()
    print(f"参数量: {count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")
    print(f"结构: 输入{INPUT_CHANNELS}通道 / {CHANNELS}通道×{NUM_RES_BLOCKS}残差块")
    x = torch.randn(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    p, v, o, wl = model(x)
    print(f"策略输出: {p.shape}, 价值输出: {v.shape}, 归属输出: {o.shape}, 胜率logit: {wl.shape}")

    state = np.random.randn(INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float32)
    policy, value, ownership, win_logit = model.get_policy_value_ownership(state)
    print(f"策略维度: {policy.shape}, 价值: {value:.4f}, 归属: {ownership.shape}, 胜率logit: {win_logit:.4f}")
