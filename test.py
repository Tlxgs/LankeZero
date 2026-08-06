#!/usr/bin/env python3
"""
模型参数健康检查工具
用法：
    python check_model.py --model model.pt [--verbose] [--output report.txt]
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import torch

from model import PolicyValueNet


def compute_stats(tensor):
    """计算一个 Tensor 的统计量（先展平再计算范数）。"""
    arr = tensor.detach().cpu().numpy().ravel()  # 展平为一维
    return {
        "shape": tensor.shape,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "l1": float(np.linalg.norm(arr, ord=1)),
        "l2": float(np.linalg.norm(arr, ord=2)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "numel": arr.size,
    }


def group_by_prefix(name):
    """将参数名分组（例如 res_blocks.0.conv1.weight 归到 res_blocks.0）"""
    parts = name.split('.')
    if parts[0] == 'res_blocks' and len(parts) >= 2:
        return '.'.join(parts[:2])
    else:
        return parts[0]


def check_model(model_path, verbose=False):
    device = torch.device('cpu')
    model = PolicyValueNet.load_model(model_path, device=device)
    model.eval()

    print(f"\n{'='*60}")
    print(f"模型健康检查: {model_path}")
    print(f"参数总量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")

    # 按分组统计
    group_stats = defaultdict(list)
    total_stats = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            stats = compute_stats(param)
            total_stats.append(stats)
            group = group_by_prefix(name)
            group_stats[group].append((name, stats))

    # 整体统计（所有参数绝对值）
    all_vals = np.concatenate([
        p.detach().cpu().numpy().ravel() for p in model.parameters() if p.requires_grad
    ])
    print("全局摘要（所有可训练参数）:")
    print(f"  参数总数: {len(all_vals):,}")
    print(f"  绝对值均值: {np.mean(all_vals):.6f}")
    print(f"  绝对值标准差: {np.std(all_vals):.6f}")
    print(f"  绝对值 L2 范数: {np.linalg.norm(all_vals):.6f}")
    print(f"  绝对值最小值: {np.min(all_vals):.6e}")
    print(f"  绝对值最大值: {np.max(all_vals):.6e}")
    print("")

    # 按组报告
    print("按模块分组统计（每组取所有参数的平均 L2 范数 / 平均 std）:")
    for group, items in group_stats.items():
        l2s = [s['l2'] for _, s in items]
        stds = [s['std'] for _, s in items]
        print(f"  {group}: 层数={len(items)}, 平均L2={np.mean(l2s):.4f}, 平均std={np.mean(stds):.6f}")

    if verbose:
        print("\n详细参数统计:")
        print(f"{'参数名':<45} {'形状':<15} {'均值':>10} {'标准差':>10} {'L1范数':>10} {'L2范数':>10} {'最小值':>10} {'最大值':>10}")
        print("-" * 120)
        for name, param in model.named_parameters():
            if param.requires_grad:
                s = compute_stats(param)
                print(f"{name:<45} {str(s['shape']):<15} {s['mean']:10.6f} {s['std']:10.6f} {s['l1']:10.2f} {s['l2']:10.2f} {s['min']:10.4e} {s['max']:10.4e}")
    else:
        # 非详细模式，至少打印各残差块的头信息
        print("\n各残差块统计（仅显示 L2 范数）:")
        for group, items in group_stats.items():
            if group.startswith('res_blocks'):
                l2s = [s['l2'] for _, s in items]
                print(f"  {group}: L2范数: {l2s}")

    print("\n健康检查完成。")


def main():
    parser = argparse.ArgumentParser(description="模型参数健康检查")
    parser.add_argument("--model", required=True, help="模型 .pt 文件路径")
    parser.add_argument("--verbose", action="store_true", help="打印所有参数的详细统计")
    parser.add_argument("--output", help="将结果写入文件（同时打印到屏幕）")
    args = parser.parse_args()

    if args.output:
        sys.stdout = open(args.output, 'w', encoding='utf-8')

    check_model(args.model, args.verbose)

    if args.output:
        sys.stdout.close()
        print(f"报告已保存至 {args.output}")


if __name__ == "__main__":
    main()