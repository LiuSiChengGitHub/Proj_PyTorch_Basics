"""
=============================================================================
ResNet18 迁移学习训练脚本
=============================================================================
学习目标：
1. 学会加载 ImageNet 预训练的 ResNet18
2. 理解为什么迁移学习要替换最后的分类头
3. 亲手跑通“先冻结、再微调”的两阶段训练
4. 完成一个最小可交付的分类项目：训练 + 验证 + 保存最优模型

运行方式：
    python examples/train_transfer.py

说明：
- 数据集使用项目内的 hymenoptera_data（ants / bees）
- 训练增强与验证预处理复用 src/transforms/presets.py
- 阶段 A：只训练 fc
- 阶段 B：微调 layer4 + fc
=============================================================================
"""
import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.transforms import train_transform, val_transform


TRAIN_DIR = os.path.join("hymenoptera_data", "train")
VAL_DIR = os.path.join("hymenoptera_data", "val")
BEST_SAVE_PATH = os.path.join("saved_models", "resnet18_hymenoptera_best.pth")
LAST_SAVE_PATH = os.path.join("saved_models", "resnet18_hymenoptera_last.pth")


BATCH_SIZE = 16
NUM_WORKERS = 0

HEAD_EPOCHS = 3
HEAD_LR = 1e-3

FINETUNE_EPOCHS = 3
FINETUNE_LR = 1e-4


def build_dataloaders():
    """构建训练集和验证集 DataLoader。"""
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def build_model(num_classes):
    """加载预训练 ResNet18 并替换最后的分类头。"""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_trainable_layers(model, stage):
    """根据阶段设置可训练参数。"""
    for param in model.parameters():
        param.requires_grad = False

    if stage == "head":
        modules_to_train = [model.fc]
    elif stage == "finetune":
        modules_to_train = [model.layer4, model.fc]
    else:
        raise ValueError(f"不支持的训练阶段: {stage}")

    for module in modules_to_train:
        for param in module.parameters():
            param.requires_grad = True


def build_optimizer(model, lr):
    """只为当前可训练参数创建优化器。"""
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.Adam(trainable_params, lr=lr)


def count_trainable_params(model):
    """统计当前可训练参数数量。"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_total_params(model):
    """统计模型总参数量。"""
    return sum(param.numel() for param in model.parameters())


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, print_shapes=False):
    """训练一个 epoch，返回平均 loss。"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for step, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        if print_shapes and step == 1:
            print(f"首个 batch 输入 min: {images.min().item():.4f}")
            print(f"首个 batch 输入 max: {images.max().item():.4f}")
            print(f"首个 batch 输入 shape: {tuple(images.shape)}")
            print(f"首个 batch 输出 shape: {tuple(outputs.shape)}")


        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(model, dataloader, loss_fn, device):
    """验证集评估，返回平均 loss 和 accuracy。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predicted = outputs.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def save_checkpoint(model, class_to_idx, best_val_acc, save_path):
    """保存最优模型和类别映射，方便后续推理。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "best_val_acc": best_val_acc,
        "arch": "resnet18",
    }
    torch.save(checkpoint, save_path)


def print_learning_checklist():
    """训练结束后提醒用户做一次知识复盘。"""
    print()
    print("=" * 70)
    print("学习验收清单")
    print("=" * 70)
    print("1. 为什么 ResNet18 预训练可以迁移到蚂蚁/蜜蜂分类？")
    print("2. 为什么这里不能继续用 CIFAR-10 的 32x32 输入？")
    print("3. 为什么验证集不能用随机增强？")
    print("4. 为什么替换的是 model.fc，而不是前面的卷积层？")
    print("5. 为什么第二阶段要把学习率调小？")
    print()
    print("建议输出物：")
    print("- 一张训练日志截图或关键指标摘要")
    print("- 一份 5 行学习记录：写下今天真正搞懂了什么")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders()
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    print(f"训练集类别列表: {train_dataset.classes}")
    print(f"训练集 class_to_idx: {train_dataset.class_to_idx}")
    print(f"验证集类别列表: {val_dataset.classes}")
    print(f"验证集 class_to_idx: {val_dataset.class_to_idx}")
    print()


    try:
        model = build_model(num_classes=len(train_dataset.classes)).to(device)
        # print(model)
        print("最后分类头 model.fc:")
        print(model.fc)

    except Exception as exc:
        raise RuntimeError(
            "加载 ResNet18 预训练权重失败。"
            "如果你是第一次运行，请确认当前环境能获取 torchvision 预训练权重缓存。"
        ) from exc

    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    stages = [
        {
            "name": "阶段 A | 只训练分类头 fc",
            "stage_key": "head",
            "epochs": HEAD_EPOCHS,
            "lr": HEAD_LR,
        },
        {
            "name": "阶段 B | 微调 layer4 + fc",
            "stage_key": "finetune",
            "epochs": FINETUNE_EPOCHS,
            "lr": FINETUNE_LR,
        },
    ]

    for stage_index, stage in enumerate(stages):
        print("=" * 70)
        print(stage["name"])
        print("=" * 70)

        set_trainable_layers(model, stage["stage_key"])
        optimizer = build_optimizer(model, stage["lr"])
        total_params = count_total_params(model)
        trainable_params = count_trainable_params(model)

        print(f"当前学习率: {stage['lr']}")
        print(f"模型总参数量: {total_params:,}")
        print(f"当前可训练参数量: {trainable_params:,}")
        print(f"本阶段训练轮数: {stage['epochs']}")
        print()

        for epoch in range(1, stage["epochs"] + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                print_shapes=(stage_index == 0 and epoch == 1),
            )
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

            is_best = val_acc > best_val_acc

            if is_best:
                best_val_acc = val_acc
                save_checkpoint(model, train_dataset.class_to_idx, best_val_acc, BEST_SAVE_PATH)

            save_checkpoint(model, train_dataset.class_to_idx, best_val_acc, LAST_SAVE_PATH)

            print(
                f"阶段: {stage['name']} | "
                f"Epoch: {epoch}/{stage['epochs']} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.2%} | "
                f"best_val_acc: {best_val_acc:.2%}"
            )

            if is_best:
                print(f"  [Saved best] {BEST_SAVE_PATH}")
            print(f"  [Saved last] {LAST_SAVE_PATH}")
            print()

        print(f"阶段结束: {stage['name']}，共训练 {stage['epochs']} 个 epoch")
        print()

    print("=" * 70)
    print("训练完成")
    print("=" * 70)
    print(f"最佳验证集准确率: {best_val_acc:.2%}")
    print(f"最优模型路径: {BEST_SAVE_PATH}")
    print(f"最新模型路径: {LAST_SAVE_PATH}")
    print_learning_checklist()


if __name__ == "__main__":
    main()
