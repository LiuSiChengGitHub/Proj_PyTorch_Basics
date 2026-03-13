"""
=============================================================================
损失函数与反向传播 学习脚本
=============================================================================
学习目标：
1. 理解损失函数的作用：量化模型预测与真实标签的差距
2. 掌握 CrossEntropyLoss（分类任务）和 MSELoss（回归任务）的用法
3. 理解 loss.backward() 的作用：计算梯度，为参数更新做准备

说明：
- CrossEntropyLoss 内部已包含 Softmax，不需要提前手动 softmax
- MSELoss 计算预测值和目标值之差的均方误差
- loss.backward() 是反向传播的入口，调用后各参数的 .grad 属性会被填充
- 梯度 = loss 对该参数的偏导数，用于后续 optimizer.step() 更新参数
=============================================================================
"""
import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


def run_cross_entropy_demo():
    """演示 CrossEntropyLoss 的基本用法。"""
    # 3 个样本，10 类分类（如 CIFAR-10）
    # 模型输出的 logits，形状 (batch, num_classes)
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 预测第 2 类
            [0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 预测第 0 类
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0],  # 预测第 5 类
        ]
    )
    targets = torch.tensor([2, 0, 5])  # 真实标签

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, targets)

    print("=" * 60)
    print("1. CrossEntropyLoss 示例")
    print("=" * 60)
    print(f"logits shape: {logits.shape}")
    print(f"targets: {targets}")
    print(f"loss: {loss.item():.4f}")
    # 预测准确时 loss 很小；预测错误时 loss 大
    # 随机初始化时，10 分类的 CrossEntropyLoss 期望约为 ln(10) ≈ 2.3026
    print()


def run_mse_loss_demo():
    """演示 MSELoss 的基本用法。"""
    preds = torch.tensor([1.5, 2.0, 3.0])    # 模型预测值
    targets = torch.tensor([1.0, 2.5, 3.0])  # 真实值

    loss_fn = nn.MSELoss()
    loss = loss_fn(preds, targets)

    print("=" * 60)
    print("2. MSELoss 示例")
    print("=" * 60)
    print(f"预测值: {preds}")
    print(f"真实值: {targets}")
    print(f"MSE loss: {loss.item():.4f}")
    # 手动验证: ((1.5-1.0)^2 + (2.0-2.5)^2 + (3.0-3.0)^2) / 3
    #         = (0.25 + 0.25 + 0.0) / 3 ≈ 0.1667
    print()


def run_backward_demo():
    """演示 loss.backward() 填充梯度的过程。"""
    linear = nn.Linear(3, 1)
    x = torch.randn(4, 3)         # 4 个样本，3 个特征
    targets = torch.randn(4, 1)   # 随机目标值

    loss_fn = nn.MSELoss()
    output = linear(x)
    loss = loss_fn(output, targets)

    print("=" * 60)
    print("3. loss.backward() 示例")
    print("=" * 60)
    print(f"loss: {loss.item():.4f}")
    print(f"backward 前，weight.grad: {linear.weight.grad}")

    loss.backward()  # 反向传播，计算梯度

    print(f"backward 后，weight.grad: {linear.weight.grad}")
    print(f"backward 后，bias.grad:   {linear.bias.grad}")
    print()
    # backward 后，每个参数的 .grad 属性被填充
    # 梯度表示 loss 对该参数的偏导数
    # 下一步：optimizer.step() 会根据梯度更新参数


def run_cifar10_loss_demo():
    """CIFAR-10 示例：前向传播 + 计算损失。"""
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
        print("4. CIFAR-10 示例（可选）")
        print("=" * 60)
        print(f"读取 CIFAR-10 失败: {exc}")
        print(f"请确认数据集是否位于相对路径: {dataset_root}")
        return

    dataloader = DataLoader(dataset, batch_size=64)
    imgs, labels = next(iter(dataloader))

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 10),
    )
    loss_fn = nn.CrossEntropyLoss()

    output = model(imgs)
    loss = loss_fn(output, labels)

    print("=" * 60)
    print("4. CIFAR-10 示例（前向传播 + 损失计算）")
    print("=" * 60)
    print(f"输入 shape: {imgs.shape}")
    print(f"输出 shape: {output.shape}")
    print(f"第一个 batch 的损失: {loss.item():.4f}")
    # 随机初始化时，10 分类期望约 ln(10) ≈ 2.3026
    print()


def main():
    run_cross_entropy_demo()
    run_mse_loss_demo()
    run_backward_demo()
    run_cifar10_loss_demo()


if __name__ == "__main__":
    main()
