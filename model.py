import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---- 推理/训练性能优化（全局生效）----
torch.backends.cudnn.benchmark = True 
torch.set_float32_matmul_precision('high')

# 超参数（统一配置见 hyperparams.py）
from hyperparams import BOARD_SIZE, INPUT_CHANNELS, CHANNELS, NUM_RES_BLOCKS, DROPOUT_RATE
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


class HeadResBlock(nn.Module):
    """头部专用的残差块，支持通道数变化"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        residual = self.projection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return F.relu(residual + out)


class PolicyValueNet(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, input_channels=INPUT_CHANNELS,
                 channels=CHANNELS, num_res_blocks=NUM_RES_BLOCKS,
                 dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels

        # 输入卷积
        self.conv_input = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResBlock(channels, dropout_rate) for _ in range(num_res_blocks)
        ])

        # 策略头
        self.policy_head1 = HeadResBlock(channels, channels, dropout_rate)
        self.policy_head2 = HeadResBlock(channels, channels//2, dropout_rate)
        self.policy_head3 = HeadResBlock(channels//2, channels//4, dropout_rate)
        
        self.conv_policy_out = nn.Conv2d(channels // 4, 1, kernel_size=1, bias=True)
        self.gap_pass = nn.AdaptiveAvgPool2d(1)
        self.fc_policy_pass = nn.Linear(channels // 4, 1, bias=True)

        # ---- 领地头（KataGo式：价值=领地求和，无独立价值头）----
        self.own_head1 = HeadResBlock(channels, channels, dropout_rate)          # 72→72
        self.own_head2 = HeadResBlock(channels, channels, dropout_rate)          # 72→72
        self.own_head3 = HeadResBlock(channels, channels // 2, dropout_rate)     # 72→36
        self.conv_own_out = nn.Conv2d(channels // 2, 1, kernel_size=1, bias=True)  # 逐点领地

        # ---------- 胜率头（第一层与归属头共用 own_head1；MaxPool2d 多次池化 →1×1，输出过 tanh 为当前玩家胜率 ±1）----------
        self.win_pool1 = nn.MaxPool2d(2)                                                # 19→9（输入为共享的 own_head1 输出）
        self.win_head2 = HeadResBlock(channels, channels, dropout_rate)                  # 9×9
        self.win_pool2 = nn.MaxPool2d(2)                                                # 9→4
        self.win_head3 = HeadResBlock(channels, channels, dropout_rate)                  # 4×4
        self.win_pool3 = nn.MaxPool2d(2)                                                # 4→2
        self.win_fc = nn.Linear(channels, 1)            # 2×2 →1×1 全局池化 → 胜率 logit

    def forward(self, x):
        # 共享特征
        out = F.relu(self.bn_input(self.conv_input(x)))
        for res in self.res_blocks:
            out = res(out)

        # ---------- 策略头 ----------
        policy = self.policy_head1(out)
        policy = self.policy_head2(policy)
        policy = self.policy_head3(policy)
        
        policy_spatial = self.conv_policy_out(policy)
        policy_spatial = policy_spatial.view(policy.size(0), -1)
        
        policy_pass_flat = self.gap_pass(policy)
        policy_pass_flat = policy_pass_flat.view(policy.size(0), -1)
        policy_pass = self.fc_policy_pass(policy_pass_flat)
        
        policy = torch.cat([policy_spatial, policy_pass], dim=1)

        # ---------- 领地头（价值 = 领地求和）----------
        own_feat = self.own_head1(out)           # (B, channels, H, W) 归属头/胜率头共用第一层
        own = self.own_head2(own_feat)
        own = self.own_head3(own)                # (B, channels//2, H, W)

        ownership_raw = self.conv_own_out(own)   # (B, 1, H, W) 逐点领地（未压缩）
        ownership = torch.tanh(ownership_raw)    # (B, 1, H, W) ∈(-1,1) 期望归属
        value = ownership.sum(dim=(2, 3))        # (B, 1) 期望领地目差（当前视角，无界）

        # ---------- 胜率头（第一层=归属头 own_head1 共享；MaxPool2d →1×1，tanh 后为当前玩家胜率 ±1；MCTS 用 logit 线性混合）----------
        win = self.win_pool1(own_feat)               # 共享 own_head1 输出 → 19→9
        win = self.win_pool2(self.win_head2(win))    # 9→4
        win = self.win_pool3(self.win_head3(win))    # 4→2
        win = F.adaptive_avg_pool2d(win, 1).flatten(1)   # 2×2 →1×1 → (B,C)
        win_logit = self.win_fc(win)                 # (B,1) logit

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
            'num_res_blocks': self.num_res_blocks,
            'dropout_rate': self.dropout_rate,
        }, path)

    def freeze_parts(self, spec):
        """按规格冻结部分网络参数（仅训练时生效，推理/导出不受影响）。
        spec: 逗号分隔的令牌——
          整数 N    = 冻结 res_blocks[N]（从 0 起）；
          'own'     = 冻结归属头（own_head1/2/3 + conv_own_out，注意 own_head1 与胜率头共享，会一并冻结）；
          'policy'  = 冻结策略头（policy_head1/2/3 + conv_policy_out + gap_pass + fc_policy_pass）；

        示例：'0,1,2,3,5,7'（冻结这些残差块）或 '0,own,policy'（冻结块0+归属头+策略头）。
        未知令牌/越界序号抛 ValueError。返回冻结的参数量。"""
        names = set()
        if isinstance(spec, str):
            tokens = [t.strip() for t in spec.split(',') if t.strip()]
        else:
            tokens = list(spec)
        for t in tokens:
            if t == 'own':
                names.update(['own_head1', 'own_head2', 'own_head3', 'conv_own_out'])
            elif t == 'policy':
                names.update(['policy_head1', 'policy_head2', 'policy_head3',
                              'conv_policy_out', 'gap_pass', 'fc_policy_pass'])

            else:
                try:
                    idx = int(t)
                except (TypeError, ValueError):
                    raise ValueError('未知冻结令牌 %r（支持：残差块序号 N、own、policy）' % t)
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

    @staticmethod
    def load_model(path, device='cpu'):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        sd = checkpoint['model_state_dict']
        old_in = sd['conv_input.weight'].shape[1]          # 旧权重实际输入通道数
        model = PolicyValueNet(
            board_size=checkpoint.get('board_size', BOARD_SIZE),
            input_channels=INPUT_CHANNELS,                 # 强制当前通道数（6），不再读旧值
            channels=checkpoint.get('channels', CHANNELS),
            num_res_blocks=checkpoint.get('num_res_blocks', NUM_RES_BLOCKS),
            dropout_rate=checkpoint.get('dropout_rate', DROPOUT_RATE),
        )
        if old_in != model.input_channels:
            # 兼容旧权重：旧通道装载到前 old_in 个，新增通道权重置零（初始输出与旧模型等价）
            sd = dict(sd)
            w = sd.pop('conv_input.weight')
            new_w = torch.zeros(model.conv_input.weight.shape, dtype=w.dtype)
            k = min(old_in, model.input_channels)
            new_w[:, :k] = w[:, :k]
            sd['conv_input.weight'] = new_w
        # 领地头/胜率头（own_head*/win_*）如旧权重缺失则保持随机初始化；
        # 仅加载匹配的主干/策略头参数（strict=False 忽略旧价值头多余键）
        model.load_state_dict(sd, strict=False)
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
