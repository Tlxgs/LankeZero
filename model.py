import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BOARD_SIZE = 13
INPUT_CHANNELS = 4
CHANNELS = 72
NUM_RES_BLOCKS = 9
DROPOUT_RATE = 0.1
from game import SCALE
class ResBlock(nn.Module):
    def __init__(self, channels, dropout_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return residual + out


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
        
        self.conv_policy_out = nn.Conv2d(channels // 4, 1, kernel_size=1, bias=False)
        self.gap_pass = nn.AdaptiveAvgPool2d(1)
        self.fc_policy_pass = nn.Linear(channels // 4, 1, bias=False)

        # 价值头
        self.value_head1 = HeadResBlock(channels, channels, dropout_rate)
        self.value_head2 = HeadResBlock(channels, channels//2, dropout_rate)
        self.value_head3 = HeadResBlock(channels//2, channels//4, dropout_rate)

        # ---- 领地归属辅助头：在 HeadResBlock 和输出之间增加一层卷积 ----
        self.own_head = nn.Sequential(
            HeadResBlock(channels, channels // 4, dropout_rate),          # 通道降维
            nn.Conv2d(channels // 4, channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),                   # 输出归属图
        )

        # 融合 ownership 的卷积块（3×3 加深）
        self.merge_conv = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 价值输出层
        self.conv_value_out = nn.Conv2d(channels//4, 1, kernel_size=1, bias=False)
        self.fc_value1 = nn.Linear(board_size * board_size, 64)
        self.dropout_value = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc_value_out = nn.Linear(64, 1)

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

        # ---------- 价值头 ----------
        value = self.value_head1(out)                         # (B, channels, H, W)
        ownership = self.own_head(value)                      # (B, 1, H, W)  多了中间卷积层

        # 将 ownership 作为额外特征图拼接到通道维度
        value = torch.cat([value, ownership], dim=1)          # (B, channels+1, H, W)
        value = self.merge_conv(value)                        # (B, channels, H, W)

        value = self.value_head2(value)
        value = self.value_head3(value)
        
        value = self.conv_value_out(value)
        value = value.view(value.size(0), -1)
        value = F.relu(self.fc_value1(value))
        value = self.dropout_value(value)
        value = torch.tanh(self.fc_value_out(value))
        
        return policy, value, ownership

    def get_policy_value(self, state, legal_mask=None, device='cpu', alpha=0):
       
        policy, value, ownership = self.get_policy_value_ownership(state, legal_mask, device)
        if alpha == 0:
            return policy,value
        komi_ch = state[3, 0, 0] 
        player = 1 if komi_ch > 0 else -1
        komi = abs(komi_ch) * SCALE
        total = np.sum(ownership)
        own_value = np.tanh((total - komi * player) / SCALE)
        
        mixed_value = (1 - alpha) * value + alpha * own_value
        return policy, mixed_value
    def get_policy_value_ownership(self, state, legal_mask=None, device='cpu'):
        self.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)
            legal_tensor = None
            if legal_mask is not None:
                legal_tensor = torch.from_numpy(legal_mask).float().to(device)
            policy_logits, value, ownership = self(state_tensor)
            if legal_tensor is not None:
                policy_logits = policy_logits.masked_fill(legal_tensor == 0, -1e4)
            policy = F.softmax(policy_logits, dim=1)
            ownership = torch.tanh(ownership)                # (1,1,H,W) ∈(-1,1)
            value = value.cpu().numpy()[0, 0]
            policy = policy.cpu().numpy()[0]
            ownership = ownership.cpu().numpy()[0, 0]
        return policy, value, ownership

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'board_size': self.board_size,
            'input_channels': self.input_channels,
            'channels': self.channels,
            'num_res_blocks': self.num_res_blocks,
            'dropout_rate': self.dropout_rate,
        }, path)

    @staticmethod
    def load_model(path, device='cpu'):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = PolicyValueNet(
            board_size=checkpoint.get('board_size', BOARD_SIZE),
            input_channels=checkpoint.get('input_channels', INPUT_CHANNELS),
            channels=checkpoint.get('channels', CHANNELS),
            num_res_blocks=checkpoint.get('num_res_blocks', NUM_RES_BLOCKS),
            dropout_rate=checkpoint.get('dropout_rate', DROPOUT_RATE),
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
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
            output_names=['policy', 'value', 'ownership'],
            dynamic_axes={'input': {0: 'batch'}},
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
    x = torch.randn(1, 4, 13, 13)
    p, v, o = model(x)
    print(f"策略输出: {p.shape}, 价值输出: {v.shape}, 归属输出: {o.shape}")
    
    state = np.random.randn(4, 13, 13).astype(np.float32)
    policy, value = model.get_policy_value(state)
    print(f"策略维度: {policy.shape}, 价值: {value:.4f}")