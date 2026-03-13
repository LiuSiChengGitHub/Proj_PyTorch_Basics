"""
=============================================================================
CIFAR-10 完整训练脚本
=============================================================================
学习目标：
1. 掌握完整的训练循环：epoch → batch → forward → loss → backward → step
2. 理解 device 的设置与使用（CPU / GPU）
3. 理解训练模式 vs 评估模式的切换（model.train() / model.eval()）
4. 理解 torch.no_grad() 在测试时的作用（省显存、加速）
5. 学会保存训练好的模型

说明：
- 框架已写好，TODO 标记处是需要你填写的核心学习点
- 填完所有 TODO 后，运行 `python train.py` 即可训练
- 预期：几个 epoch 后 loss 下降、准确率上升
=============================================================================
"""
import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.simple_cnn import SimpleCNN


# =============================================================================
# 超参数配置
# =============================================================================
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
PRINT_EVERY = 100  # 每隔多少步打印一次 loss
SAVE_PATH = "saved_models"


def load_data():
    """加载 CIFAR-10 训练集和测试集。"""
    dataset_root = os.path.join("..", "datasets")

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        train=True,
        transform=transforms.ToTensor(),
        download=False,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        train=False,
        transform=transforms.ToTensor(),
        download=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    return train_loader, test_loader


def evaluate(model, test_loader, device):
    """在测试集上评估模型准确率。"""
    correct = 0
    total = 0

    # TODO: 切换到评估模式
    # 提示: model.???()

    # TODO: 关闭梯度计算（省显存、加速，测试时不需要梯度）
    # 提示: with torch.???():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        # argmax 取预测类别
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # TODO: 切换回训练模式
    # 提示: model.???()

    accuracy = correct / total
    return accuracy


def train():
    """主训练函数。"""
    # =================================================================
    # 1. 设备设置
    # =================================================================
    # TODO: 设置 device（优先用 GPU，没有则用 CPU）
    # 提示: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = None  # TODO

    print(f"使用设备: {device}")

    # =================================================================
    # 2. 数据加载
    # =================================================================
    train_loader, test_loader = load_data()

    # =================================================================
    # 3. 模型、损失函数、优化器
    # =================================================================
    model = SimpleCNN()
    # TODO: 把模型搬到 device 上
    # 提示: model.to(???)

    # TODO: 选择损失函数（这是分类任务，用哪个？）
    # 提示: nn.???
    loss_fn = None  # TODO

    # TODO: 选择优化器，传入模型参数和学习率
    # 提示: torch.optim.???(model.parameters(), lr=LEARNING_RATE)
    optimizer = None  # TODO

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"损失函数: {loss_fn}")
    print(f"优化器: {optimizer}")
    print()

    # =================================================================
    # 4. 训练循环
    # =================================================================
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0

        for step, (imgs, labels) in enumerate(train_loader):
            # 把数据搬到 device 上
            imgs = imgs.to(device)
            labels = labels.to(device)

            # TODO: 完成训练的 5 个核心步骤（写出正确顺序）
            # 提示: 回忆 nn_optimizer.py 里学的三步口诀，再加上 forward 和 loss 计算
            #
            # 第 1 步: 清除旧梯度
            # ???
            #
            # 第 2 步: 前向传播
            # outputs = ???
            #
            # 第 3 步: 计算损失
            # loss = ???
            #
            # 第 4 步: 反向传播
            # ???
            #
            # 第 5 步: 更新参数
            # ???

            running_loss += loss.item()

            # 每隔 PRINT_EVERY 步打印一次
            if (step + 1) % PRINT_EVERY == 0:
                avg_loss = running_loss / PRINT_EVERY
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], "
                      f"Step [{step+1}/{len(train_loader)}], "
                      f"Loss: {avg_loss:.4f}")
                running_loss = 0.0

        # 每个 epoch 结束后评估
        accuracy = evaluate(model, test_loader, device)
        print(f"  >>> Epoch {epoch+1} 完成, 测试集准确率: {accuracy:.2%}")
        print()

    # =================================================================
    # 5. 保存模型
    # =================================================================
    os.makedirs(SAVE_PATH, exist_ok=True)
    save_file = os.path.join(SAVE_PATH, "simple_cnn_cifar10.pth")
    torch.save(model.state_dict(), save_file)
    print(f"模型已保存到: {save_file}")


def main():
    train()


if __name__ == "__main__":
    main()
