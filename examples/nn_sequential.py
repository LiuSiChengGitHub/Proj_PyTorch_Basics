"""
=============================================================================
nn.Sequential 学习脚本
=============================================================================
学习目标：
1. 理解 nn.Sequential 的作用：按顺序把多个层组合成一个模块
2. 用 Sequential 搭建一个完整的 CNN 结构（针对 CIFAR-10）
3. 打印模型结构，观察各层的参数
4. 用随机输入验证 shape 是否贯通

说明：
- nn.Sequential 接收一组层，调用时依次执行 forward
- 相比手写 forward 函数，Sequential 写法更简洁
- 适合层之间逻辑简单、顺序串行的场景
=============================================================================
"""
import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


def build_model():
    """用 nn.Sequential 搭建一个简单 CNN（适配 CIFAR-10 32x32 输入）。"""
    # 输入: (batch, 3, 32, 32)
    # Conv2d + MaxPool2d 堆叠，每次 MaxPool2d(2) 把 H/W 减半
    # 32 -> 16 -> 8 -> 4
    # 最终展平后: 64 * 4 * 4 = 1024
    return nn.Sequential(
        nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=5, padding=2),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 64),
        nn.Linear(64, 10),
    )


def print_model_structure(model):
    """打印模型结构与参数量。"""
    print("=" * 60)
    print("模型结构")
    print("=" * 60)
    print(model)
    print()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    print()


def run_shape_check(model):
    """用随机输入验证模型前向传播的 shape 是否正确。"""
    # 模拟一个 batch：4 张 CIFAR-10 图片 (3, 32, 32)
    x = torch.randn(4, 3, 32, 32)
    output = model(x)

    print("=" * 60)
    print("前向传播 shape 验证")
    print("=" * 60)
    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {output.shape}")
    # 期望输出: (4, 10)，即 4 张图片各自 10 个类的 logits
    print()


def run_cifar10_demo(model):
    """CIFAR-10 示例：用真实数据跑一遍前向传播。"""
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
        print("CIFAR-10 示例（可选）")
        print("=" * 60)
        print(f"读取 CIFAR-10 失败: {exc}")
        print(f"请确认数据集是否位于相对路径: {dataset_root}")
        return

    dataloader = DataLoader(dataset, batch_size=8)
    imgs, labels = next(iter(dataloader))

    output = model(imgs)

    print("=" * 60)
    print("CIFAR-10 示例（真实数据前向传播）")
    print("=" * 60)
    print(f"输入 shape: {imgs.shape}")
    print(f"输出 shape: {output.shape}")
    print(f"真实标签: {labels}")
    print()


def main():
    model = build_model()
    print_model_structure(model)
    run_shape_check(model)
    run_cifar10_demo(model)


if __name__ == "__main__":
    main()
