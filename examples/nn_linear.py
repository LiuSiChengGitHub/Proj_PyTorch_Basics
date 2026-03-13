"""
=============================================================================
nn.Linear 学习脚本
=============================================================================
学习目标：
1. 理解全连接层（线性层）的 in_features 和 out_features 含义
2. 观察 Linear 的权重与偏置的形状
3. 理解如何用 nn.Flatten 把图片展平后接入全连接层

说明：
- nn.Linear(in_features, out_features) 等价于 y = xW^T + b
- in_features：输入特征数量（即输入向量的维度）
- out_features：输出特征数量（即神经元个数）
- 权重 W 的形状：(out_features, in_features)
- 偏置 b 的形状：(out_features,)
=============================================================================
"""
import os

import torch
import torchvision
from torch import nn
from torchvision import transforms


def run_small_linear_demo():
    """小张量示例：直接观察 Linear 的输出形状与参数。"""
    # 2 个样本，每个有 5 个特征
    x = torch.randn(2, 5)  # shape: (batch, in_features)

    linear = nn.Linear(in_features=5, out_features=3)
    output = linear(x)

    print("=" * 60)
    print("1. 小张量示例")
    print("=" * 60)
    print(f"输入 x shape: {x.shape}")
    print(f"Linear 参数: in_features=5, out_features=3")
    print(f"输出 shape: {output.shape}")
    print(f"权重 W shape: {linear.weight.shape}")  # (out_features, in_features)
    print(f"偏置 b shape: {linear.bias.shape}")    # (out_features,)
    print()


def run_flatten_then_linear_demo():
    """演示将图片展平后送入全连接层。"""
    # 模拟一个 batch 的 CIFAR-10 图片：(batch=4, C=3, H=32, W=32)
    imgs = torch.randn(4, 3, 32, 32)

    flatten = nn.Flatten()         # 展平除 batch 维以外的所有维度
    linear = nn.Linear(3072, 10)   # 3*32*32=3072 个输入特征，10 类输出

    flattened = flatten(imgs)      # (4, 3072)
    output = linear(flattened)     # (4, 10)

    print("=" * 60)
    print("2. Flatten -> Linear 示例")
    print("=" * 60)
    print(f"原始图片 shape: {imgs.shape}")
    print(f"Flatten 后 shape: {flattened.shape}")
    print(f"Linear 输出 shape: {output.shape}")
    # 输出是 logits（原始得分），还没有经过 softmax
    # (4, 10) 表示 4 张图片各自对应 10 个类别的得分
    print()


def run_cifar10_demo():
    """CIFAR-10 示例：展平真实图片并通过全连接层。"""
    dataset_root = os.path.join("..", "datasets")

    try:
        dataset = torchvision.datasets.CIFAR10(
            root=dataset_root,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )
    except RuntimeError as exc:
        print("=" * 60)
        print("3. CIFAR-10 示例（可选）")
        print("=" * 60)
        print(f"读取 CIFAR-10 失败: {exc}")
        print(f"请确认数据集是否位于相对路径: {dataset_root}")
        return

    img, label = dataset[0]
    img = img.unsqueeze(0)  # (1, 3, 32, 32)

    flatten = nn.Flatten()
    linear = nn.Linear(3 * 32 * 32, 10)

    flattened = flatten(img)
    output = linear(flattened)

    print("=" * 60)
    print("3. CIFAR-10 示例")
    print("=" * 60)
    print(f"输入 shape: {img.shape}")
    print(f"展平后 shape: {flattened.shape}")
    print(f"Linear 输出 shape: {output.shape}")
    print(f"输出 logits: {output.detach()}")
    print()


def main():
    run_small_linear_demo()
    run_flatten_then_linear_demo()
    run_cifar10_demo()


if __name__ == "__main__":
    main()
