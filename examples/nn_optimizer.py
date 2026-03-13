"""
=============================================================================
优化器 学习脚本
=============================================================================
学习目标：
1. 理解优化器的作用：根据梯度更新模型参数
2. 掌握 SGD 和 Adam 的基本用法
3. 理解一次完整训练步骤：forward -> loss -> backward -> optimizer.step()
4. 理解 optimizer.zero_grad() 的必要性：防止梯度在多次 backward 之间累积

训练三步口诀：
  1. optimizer.zero_grad()  # 清除旧梯度
  2. loss.backward()        # 计算新梯度
  3. optimizer.step()       # 更新参数
=============================================================================
"""
import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


def build_model():
    """搭建一个简单 CNN（适配 CIFAR-10，输出 10 类）。"""
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, padding=2),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, kernel_size=5, padding=2),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=5, padding=2),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 64),
        nn.Linear(64, 10),
    )


def run_sgd_demo():
    """演示 SGD 优化器的单步参数更新。"""
    linear = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    x = torch.randn(4, 3)
    targets = torch.randn(4, 1)

    before = linear.weight.detach().clone()

    # 完整的一步训练
    optimizer.zero_grad()               # 1. 清除旧梯度
    output = linear(x)                  # 2. 前向传播
    loss = loss_fn(output, targets)     # 3. 计算损失
    loss.backward()                     # 4. 反向传播
    optimizer.step()                    # 5. 更新参数

    after = linear.weight.detach().clone()

    print("=" * 60)
    print("1. SGD 优化器单步示例")
    print("=" * 60)
    print(f"更新前 weight: {before}")
    print(f"更新后 weight: {after}")
    print(f"参数变化量: {(after - before).abs().sum().item():.6f}")
    print(f"loss: {loss.item():.4f}")
    print()


def run_adam_demo():
    """演示 Adam 优化器的单步参数更新。"""
    linear = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    x = torch.randn(4, 3)
    targets = torch.randn(4, 1)

    before = linear.weight.detach().clone()

    optimizer.zero_grad()
    output = linear(x)
    loss = loss_fn(output, targets)
    loss.backward()
    optimizer.step()

    after = linear.weight.detach().clone()

    print("=" * 60)
    print("2. Adam 优化器单步示例")
    print("=" * 60)
    print(f"更新前 weight: {before}")
    print(f"更新后 weight: {after}")
    print(f"参数变化量: {(after - before).abs().sum().item():.6f}")
    print(f"loss: {loss.item():.4f}")
    print()


def run_cifar10_training_demo():
    """CIFAR-10 示例：用 SGD 跑 5 步，观察 loss 变化趋势。"""
    dataset_root = os.path.join("..", "datasets")

    try:
        dataset = torchvision.datasets.CIFAR10(
            root=dataset_root,
            train=True,
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

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    print("=" * 60)
    print("3. CIFAR-10 示例（前 5 步训练）")
    print("=" * 60)

    for step, (imgs, labels) in enumerate(dataloader):
        if step >= 5:
            break

        optimizer.zero_grad()           # 1. 清除旧梯度
        output = model(imgs)            # 2. 前向传播
        loss = loss_fn(output, labels)  # 3. 计算损失
        loss.backward()                 # 4. 反向传播
        optimizer.step()                # 5. 更新参数

        print(f"  Step {step}: loss = {loss.item():.4f}")

    print()
    print("训练三步口诀：")
    print("  1. optimizer.zero_grad()  # 清除旧梯度（防止累积）")
    print("  2. loss.backward()        # 计算新梯度")
    print("  3. optimizer.step()       # 根据梯度更新参数")
    print()


def main():
    run_sgd_demo()
    run_adam_demo()
    run_cifar10_training_demo()


if __name__ == "__main__":
    main()
