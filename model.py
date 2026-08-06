import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---- 推理/训练性能优化（全局生效）----
torch.backends.cudnn.benchmark = True 
torch.set_float32_matmul_precision('high')

# 超参数（统一配置见 hyperparams.py）
from hyperparams import BOARD_SIZE, INPUT_CHANNELS, CHANNELS, HEAD_CHANNELS, NUM_RES_BLOCKS, DROPOUT_RATE
class ResBlock(nn.Module):
    def __init__(self, channels, dropout_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bottleneck = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn_bottle = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        res = x
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        res = self.bottleneck(res)
        res = self.bn_bottle(res)
        out = self.dropout(out)
        return res + out


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
        self.bn_input = nn.BatchNorm2d(channels)

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResBlock(channels, dropout_rate) for _ in range(num_res_blocks)
        ])

        # ---------- 三个头（KataGo 风格轻量塔，每个头约 0.1M 参数）----------
        # 共享头塔：归属头与胜率头共用第一层 3×3 卷积（对应 KataGo ownership/value head 共用 conv tower）
        self.head_conv = nn.Conv2d(channels, self.head_channels, kernel_size=3, padding=1, bias=False)
        self.head_bn = nn.BatchNorm2d(self.head_channels)

        # 策略头（独立 2 层 3×3 塔）：逐点 logit + 全局池化 pass logit
        self.policy_conv = nn.Conv2d(channels, self.head_channels, kernel_size=3, padding=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(self.head_channels)
        self.policy_conv2 = nn.Conv2d(self.head_channels, self.head_channels, kernel_size=3, padding=1, bias=False)
        self.policy_bn2 = nn.BatchNorm2d(self.head_channels)
        self.conv_policy_out = nn.Conv2d(self.head_channels, 1, kernel_size=1, bias=True)
        self.fc_policy_pass = nn.Linear(self.head_channels, 1, bias=True)

        # 归属头（共享塔 + 1 层加深；KataGo：逐点本地归属 + 全局偏置）
        self.own_conv = nn.Conv2d(self.head_channels, self.head_channels, kernel_size=3, padding=1, bias=False)
        self.own_bn = nn.BatchNorm2d(self.head_channels)
        self.conv_own_out = nn.Conv2d(self.head_channels, 1, kernel_size=1, bias=True)
        self.fc_own_global = nn.Linear(self.head_channels, 1, bias=True)

        # 胜率头（共享塔 + 1 层加深；全局平均池化 → 2 层 MLP → 胜率 logit）
        self.win_conv = nn.Conv2d(self.head_channels, self.head_channels, kernel_size=3, padding=1, bias=False)
        self.win_bn = nn.BatchNorm2d(self.head_channels)
        self.win_fc1 = nn.Linear(self.head_channels, self.head_channels, bias=True)
        self.win_fc2 = nn.Linear(self.head_channels, 1, bias=True)

    def forward(self, x):
        # 共享特征
        out = F.relu(self.bn_input(self.conv_input(x)))
        for res in self.res_blocks:
            out = res(out)

        # ---------- 策略头（独立 2 层 3×3 塔）----------
        policy = F.relu(self.policy_bn(self.policy_conv(out)))                  # (B, HEAD_CHANNELS, H, W)
        policy = F.relu(self.policy_bn2(self.policy_conv2(policy)))             # (B, HEAD_CHANNELS, H, W)
        policy_spatial = self.conv_policy_out(policy).view(policy.size(0), -1)  # (B, 361) 逐点 logit
        policy_pass = self.fc_policy_pass(                                      # (B, 1) pass logit（全局池化）
            F.adaptive_avg_pool2d(policy, 1).flatten(1))
        policy = torch.cat([policy_spatial, policy_pass], dim=1)                # (B, 362)

        # ---------- 共享头塔（归属头/胜率头共用第一层，KataGo 风格）----------
        head = F.relu(self.head_bn(self.head_conv(out)))                        # (B, HEAD_CHANNELS, H, W)

        # ---------- 领地头（价值 = 归属求和；共享塔 + 1 层加深，本地 + 全局偏置）----------
        own = F.relu(self.own_bn(self.own_conv(head)))                          # (B, HEAD_CHANNELS, H, W)
        own_local = self.conv_own_out(own)                                      # (B, 1, H, W) 逐点本地归属
        own_global = self.fc_own_global(                                        # (B, 1) 全局偏置（允许整体偏移）
            F.adaptive_avg_pool2d(own, 1).flatten(1))
        ownership_raw = own_local + own_global.view(-1, 1, 1, 1)                # (B, 1, H, W) 未压缩
        ownership = torch.tanh(ownership_raw)                                   # (B, 1, H, W) ∈(-1,1)
        value = ownership.sum(dim=(2, 3))                                       # (B, 1) 期望领地目差（当前视角）

        # ---------- 胜率头（共享塔 + 1 层加深；全局平均池化 → 2 层 MLP；tanh 后=当前玩家胜率 ±1，MCTS 用 logit）----------
        win = F.relu(self.win_bn(self.win_conv(head)))                          # (B, HEAD_CHANNELS, H, W)
        win = F.relu(self.win_fc1(F.adaptive_avg_pool2d(win, 1).flatten(1)))    # (B, HEAD_CHANNELS)
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
            value = value.cpu().numpy()[0, 0]
            policy = policy.cpu().numpy()[0]
            ownership = ownership.cpu().numpy()[0, 0]
            win_logit = float(win_logit.cpu().numpy()[0, 0])   # 胜率头 logit（tanh 后 = 当前玩家胜率 ±1）
        return policy, value, ownership, win_logit

    def save_model(self, path):
        torch.save({
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
          整数 N    = 冻结 res_blocks[N]（从 0 起）；
          'own'     = 冻结归属头（head_conv/head_bn + own_conv/own_bn + conv_own_out + fc_own_global；
                      注意 head_conv/head_bn 为共享头塔，与胜率头共用，冻结后胜率头输入为静态特征，
                      但 win_conv/win_fc* 仍可训练）；
          'policy'  = 冻结策略头（policy_conv/policy_bn/policy_conv2/policy_bn2 + conv_policy_out + fc_policy_pass）；
          'win'     = 冻结胜率头（win_conv/win_bn + win_fc1/win_fc2，共享头塔不受影响）；

        示例：'0,1,2,3,5,7'（冻结这些残差块）或 '0,own,policy'（冻结块0+归属头+策略头）。
        未知令牌/越界序号抛 ValueError。返回冻结的参数量。"""
        names = set()
        if isinstance(spec, str):
            tokens = [t.strip() for t in spec.split(',') if t.strip()]
        else:
            tokens = list(spec)
        for t in tokens:
            if t == 'own':
                names.update(['head_conv', 'head_bn', 'own_conv', 'own_bn',
                              'conv_own_out', 'fc_own_global'])
            elif t == 'policy':
                names.update(['policy_conv', 'policy_bn', 'policy_conv2', 'policy_bn2',
                              'conv_policy_out', 'fc_policy_pass'])
            elif t == 'win':
                names.update(['win_conv', 'win_bn', 'win_fc1', 'win_fc2'])

            else:
                try:
                    idx = int(t)
                except (TypeError, ValueError):
                    raise ValueError('未知冻结令牌 %r（支持：残差块序号 N、own、policy、win）' % t)
                if not (0 <= idx < len(self.res_blocks)):
                    raise ValueError('冻结块序号 %d 越界（当前共 %d 个残差块）' % (idx, len(self.res_blocks)))
                names.add('res_blocks.%d' % idx)
        n_frozen = 0
        for name, param in self.named_parameters():
            frozen = any(name == n or name.startswith(n + '.') for n in names)
            param.requires_grad = not frozen
            if frozen:
                n_frozen += param.numel()
        return n_frozen

    def reset_bn(self):
        """重新初始化全部 BatchNorm 层：weight=1、bias=0、running_mean=0、running_var=1、
        num_batches_tracked=0。BN 的 affine 缩放会被后续层抵消，对其施加 weight decay 无意义，
        且会持续缩小有效表达（KataGo 惯例：BN/bias 不参与衰减）；长期训练后被衰减的 BN 可用本方法恢复。
        注意：训练中 BN 使用 batch 统计，重置后几个 batch 即恢复；eval 使用 running 统计，重置后短期不准。
        返回重置的 BN 层数。"""
        n = 0
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.0)
                    m.bias.data.fill_(0.0)
                    m.reset_running_stats()
                    n += 1
        print(f'[模型] 已重置 {n} 个 BatchNorm 层（weight=1, bias=0, running stats 归零）')
        return n

    @staticmethod
    def load_model(path, device='cpu', reset_bn=False):
        """加载任意配置的模型 checkpoint：
        channels / num_res_blocks / head_channels 跟随 checkpoint（支持加载不同规模模型，
        如不同配置的模型）；形状不匹配的键按过滤跳过（对应层随机初始化）。
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        sd = checkpoint['model_state_dict']
        model = PolicyValueNet(
            board_size=checkpoint.get('board_size', BOARD_SIZE),
            input_channels=INPUT_CHANNELS,                    # 输入通道固定 6（不变）
            channels=checkpoint.get('channels', CHANNELS),    # 跟随 checkpoint（任意通道数）
            num_res_blocks=checkpoint.get('num_res_blocks', NUM_RES_BLOCKS),  # 跟随 checkpoint（任意块数）
            head_channels=checkpoint.get('head_channels', HEAD_CHANNELS),      # 跟随 checkpoint
            dropout_rate=checkpoint.get('dropout_rate', DROPOUT_RATE),
        )
        # 只装载形状完全匹配的键：结构/通道不一致的键跳过并提示（不抛错）
        state = model.state_dict()
        compatible = {k: v for k, v in sd.items() if k in state and state[k].shape == v.shape}
        skipped = [k for k in sd if k not in compatible]
        if skipped:
            print(f'[模型] 跳过形状不兼容键 {len(skipped)} 个（结构/通道不匹配，对应层随机初始化）')
        model.load_state_dict(compatible, strict=False)
        # BN 健康提示：weight decay 会持续衰减 BN 的 affine 缩放（KataGo 惯例：BN/bias 不参与衰减）。
        # 若 mean 明显 <1（如 <0.8），说明已被衰减，建议 reset_bn=True 重置后再训练。
        with torch.no_grad():
            bn_means = [m.weight.data.mean().item() for m in model.modules()
                        if isinstance(m, nn.BatchNorm2d)]
        if bn_means:
            bn_mean = float(np.mean(bn_means))
            if bn_mean < 0.8:
                print(f'[模型] 提示: BN weight 均值={bn_mean:.2f}（<0.8，疑似被 weight decay 衰减；可用 reset_bn=True 重置）')
        if reset_bn:
            model.reset_bn()
        model.to(device)
        return model




    def export_onnx(self, path):
        self.eval()
        original_device = next(self.parameters()).device
        if original_device.type != 'cpu':
            self.to('cpu')

        dummy_input = torch.randn(1, self.input_channels, self.board_size, self.board_size)
        torch.onnx.export(
            self,
            dummy_input,
            path,
            input_names=['input'],
            output_names=['policy', 'value', 'ownership', 'win_logit'],
            dynamic_axes={'input': {0: 'batch'}, 'policy': {0: 'batch'},
                           'value': {0: 'batch'}, 'ownership': {0: 'batch'},
                           'win_logit': {0: 'batch'}},
            opset_version=11
        )
        if original_device.type != 'cpu':
            self.to(original_device)
        print(f"[ONNX] 模型已导出至 {path}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PolicyValueNet()
    print(f"参数量: {count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")
    x = torch.randn(1, 6, 19, 19)
    p, v, o, wl = model(x)
    print(f"策略输出: {p.shape}, 价值输出: {v.shape}, 归属输出: {o.shape}, 胜率logit: {wl.shape}")
    
    state = np.random.randn(6, 19, 19).astype(np.float32)
    policy, value, ownership, win_logit = model.get_policy_value_ownership(state)
    print(f"策略维度: {policy.shape}, 价值: {value:.4f}, 归属: {ownership.shape}, 胜率logit: {win_logit:.4f}")
