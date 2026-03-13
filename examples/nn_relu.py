"""
=============================================================================
nn.ReLU / nn.Sigmoid 学习脚本
=============================================================================
学习目标：
1. 理解非线性激活函数的作用：让神经网络能拟合非线性关系
2. 观察 ReLU 对负值的截断效果
3. 观察 Sigmoid 的 S 形压缩效果
4. 结合 CIFAR-10，观察激活函数作用于真实图片特征图的效果

说明：
- ReLU(x) = max(0, x)，负值置零，正值保持不变
- Sigmoid(x) = 1 / (1 + e^(-x))，输出范围压缩到 (0, 1)
- 激活函数不改变张量的 shape，只改变数值
=============================================================================
"""
import os

import torch
import torchvision
from torch import nn
from torchvision import transforms


def run_relu_demo():
    """演示 ReLU 对小张量的截断效果。"""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    relu = nn.ReLU()
    output = relu(x)

    print("=" * 60)
    print("1. ReLU 示例")
    print("=" * 60)
    print(f"输入: {x}")
    print(f"输出: {output}")
    # ReLU(x) = max(0, x)，负值变为 0，正值不变
    print()


def run_sigmoid_demo():
    """演示 Sigmoid 的 S 形压缩效果。"""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    sigmoid = nn.Sigmoid()
    output = sigmoid(x)

    print("=" * 60)
    print("2. Sigmoid 示例")
    print("=" * 60)
    print(f"输入: {x}")
    print(f"输出: {output}")
    # 所有输出值都被压缩到 (0, 1) 之间
    # 输入为 0 时输出为 0.5，输入越大越接近 1，越小越接近 0
    print()


def run_cifar10_demo():
    """CIFAR-10 示例：观察激活函数作用于真实图片特征图的效果。"""
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

    relu = nn.ReLU()
    sigmoid = nn.Sigmoid()

    relu_out = relu(img)
    sigmoid_out = sigmoid(img)

    print("=" * 60)
    print("3. CIFAR-10 示例（激活函数对图片特征图的效果）")
    print("=" * 60)
    print(f"输入 shape: {img.shape}")
    print(f"ReLU 输出 shape: {relu_out.shape}")
    print(f"Sigmoid 输出 shape: {sigmoid_out.shape}")
    print()
    print(f"输入像素值范围:    [{img.min():.3f}, {img.max():.3f}]")
    print(f"ReLU 后像素值范围: [{relu_out.min():.3f}, {relu_out.max():.3f}]")
    print(f"Sigmoid 后范围:    [{sigmoid_out.min():.3f}, {sigmoid_out.max():.3f}]")
    print()
    # 注意：ToTensor 已将像素归一化到 [0, 1]
    # 因此 ReLU 对该图不会截断任何值（全部为非负数）
    # Sigmoid 会把 [0, 1] 的输入继续压缩，导致值更集中在 [0.5, 1) 附近


def main():
    run_relu_demo()
    run_sigmoid_demo()
    run_cifar10_demo()


if __name__ == "__main__":
    main()
