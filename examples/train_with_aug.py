"""
=============================================================================
数据增强实验脚本
=============================================================================
学习目标：
1. 理解数据增强（Data Augmentation）对模型泛化能力的影响
2. 学会用 transforms.Compose 组装增强 pipeline
3. 对比：无增强 vs 有增强 的训练效果差异
4. 掌握适合 CIFAR-10（32x32 小图）的常用增强手段

说明：
- 本脚本会先用「基线 transform」训练，再用「增强 transform」训练
- 每个配置训练相同 epoch，最后打印准确率对比表
- TODO 处需要你组装增强 transform（不引用 src/transforms/presets.py）
=============================================================================
"""
import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

# 复用 SimpleCNN 模型
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.simple_cnn import SimpleCNN


# =============================================================================
# 超参数
# =============================================================================
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 5  # 为了对比实验，epoch 少一些
PRINT_EVERY = 200
DATASET_ROOT = os.path.join("..", "datasets")


# =============================================================================
# Transform 定义
# =============================================================================

# 基线 transform：只做 ToTensor，不做任何增强
# REVIEW: 基线没有 Normalize 而增强组有，对比时增强组的提升可能部分来自标准化而非增强本身；
#         更严谨的做法是基线也加 Normalize，只控制"增强"这一个变量
baseline_transform = transforms.Compose([
    transforms.ToTensor(),
])

# TODO: 增强 transform
# 要求：用 transforms.Compose 从零组装，适合 CIFAR-10 的 32x32 小图
# 可选的增强操作（选几个合适的组合起来）：
#   - transforms.RandomCrop(32, padding=4)    # 随机裁剪（先 pad 再 crop）
#   - transforms.RandomHorizontalFlip()       # 随机水平翻转
#   - transforms.ColorJitter(...)             # 颜色抖动
#   - transforms.RandomRotation(...)          # 随机旋转
#   - transforms.Normalize(mean, std)         # 标准化（ImageNet 均值方差）
#   - transforms.ToTensor()                   # 别忘了这个！
#
# 注意：ToTensor() 要放在 Normalize 前面，因为 Normalize 需要 Tensor 输入
#
# augmented_transform = transforms.Compose([
#     ???  # TODO: 组装你的增强 pipeline
# ])
augmented_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),  
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    ),
])

# 测试集 transform（不做增强，保持公平对比）
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    ),
])


def load_data(train_transform):
    """加载 CIFAR-10，训练集用指定 transform，测试集统一用 test_transform。"""
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATASET_ROOT,
        train=True,
        transform=train_transform,
        download=False,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATASET_ROOT,
        train=False,
        transform=test_transform,
        download=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


def evaluate(model, test_loader, device):
    """测试集评估准确率。"""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total


def train_one_config(config_name, train_transform, device):
    """用指定的 transform 训练一个模型，返回每个 epoch 的准确率列表。"""
    print("=" * 60)
    print(f"实验: {config_name}")
    print("=" * 60)

    train_loader, test_loader = load_data(train_transform)
    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    epoch_accuracies = []

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % PRINT_EVERY == 0:
                avg_loss = running_loss / PRINT_EVERY
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], "
                      f"Step [{step+1}], Loss: {avg_loss:.4f}")
                running_loss = 0.0

        accuracy = evaluate(model, test_loader, device)
        epoch_accuracies.append(accuracy)
        print(f"  >>> Epoch {epoch+1} 准确率: {accuracy:.2%}")

    print()
    return epoch_accuracies


def print_comparison(baseline_acc, augmented_acc):
    """打印两组实验的准确率对比表。"""
    print("=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"{'Epoch':<8} {'基线（无增强）':<16} {'有增强':<16} {'差值':<10}")
    print("-" * 50)
    for i in range(NUM_EPOCHS):
        base = baseline_acc[i]
        aug = augmented_acc[i] if i < len(augmented_acc) else 0.0
        diff = aug - base
        sign = "+" if diff >= 0 else ""
        print(f"{i+1:<8} {base:<16.2%} {aug:<16.2%} {sign}{diff:.2%}")

    print("-" * 50)
    print(f"{'最终':<8} {baseline_acc[-1]:<16.2%} {augmented_acc[-1]:<16.2%} "
          f"{'+'if augmented_acc[-1]-baseline_acc[-1]>=0 else ''}"
          f"{augmented_acc[-1]-baseline_acc[-1]:.2%}")
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print()

    # 实验 1: 基线（无增强）
    baseline_acc = train_one_config("基线（无增强）", baseline_transform, device)

    # 实验 2: 有增强
    if augmented_transform is None:
        print("=" * 60)
        print("增强 transform 尚未实现（TODO）")
        print("请填写 augmented_transform 后再运行")
        print("=" * 60)
        return

    augmented_acc = train_one_config("数据增强", augmented_transform, device)

    # 打印对比表
    print_comparison(baseline_acc, augmented_acc)


if __name__ == "__main__":
    main()
