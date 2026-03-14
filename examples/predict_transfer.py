"""
=============================================================================
ResNet18 迁移学习推理脚本
=============================================================================
学习目标：
1. 学会加载迁移学习训练好的分类模型
2. 理解单张图片推理时为什么要复用验证集预处理
3. 输出类别概率，形成最小可交付推理闭环

运行方式：
    python examples/predict_transfer.py hymenoptera_data/val/ants/0013035.jpg

可选参数：
    --weights saved_models/resnet18_hymenoptera_best.pth
=============================================================================
"""
import argparse
import os
import sys

import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.transforms import val_transform


DEFAULT_WEIGHTS_PATH = os.path.join("saved_models", "resnet18_hymenoptera_best.pth")
DEFAULT_CLASS_TO_IDX = {"ants": 0, "bees": 1}


def parse_args():
    parser = argparse.ArgumentParser(description="使用迁移学习好的 ResNet18 做单图分类推理")
    parser.add_argument("image_path", help="待预测图片路径")
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS_PATH,
        help=f"模型权重路径，默认: {DEFAULT_WEIGHTS_PATH}",
    )
    return parser.parse_args()


def build_model(num_classes):
    """构建与训练时一致的 ResNet18 结构。"""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(weights_path, device):
    """兼容加载 checkpoint 字典或纯 state_dict。"""
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        class_to_idx = checkpoint.get("class_to_idx", DEFAULT_CLASS_TO_IDX)
    else:
        state_dict = checkpoint
        class_to_idx = DEFAULT_CLASS_TO_IDX

    return state_dict, class_to_idx


def predict(image_path, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到待预测图片: {image_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到模型权重: {weights_path}")

    state_dict, class_to_idx = load_checkpoint(weights_path, device)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    model = build_model(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu()
        pred_idx = probabilities.argmax().item()

    pred_class = idx_to_class[pred_idx]

    print(f"使用设备: {device}")
    print(f"图片文件: {os.path.basename(image_path)}")
    print(f"预测类别: {pred_class}")
    print("类别概率:")
    for idx in sorted(idx_to_class):
        class_name = idx_to_class[idx]
        print(f"  - {class_name}: {probabilities[idx]:.4f}")


def main():
    args = parse_args()
    predict(args.image_path, args.weights)


if __name__ == "__main__":
    main()

