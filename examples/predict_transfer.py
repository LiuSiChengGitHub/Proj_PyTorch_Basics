"""
=============================================================================
ResNet18 transfer learning inference script
=============================================================================

Examples:
    python examples/predict_transfer.py hymenoptera_data/val/ants/0013035.jpg
    python examples/predict_transfer.py hymenoptera_data/val/ants

Optional args:
    --weights saved_models/resnet18_hymenoptera_best.pth
    --confidence-threshold 0.80
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
DEFAULT_CONFIDENCE_THRESHOLD = 0.80
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Use a trained ResNet18 for single or batch inference.")
    parser.add_argument("input_path", help="Path to one image or a folder of images.")
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS_PATH,
        help=f"Model weights path. Default: {DEFAULT_WEIGHTS_PATH}",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Threshold used for the confidence hint. Default: {DEFAULT_CONFIDENCE_THRESHOLD}",
    )
    return parser.parse_args()


def build_model(num_classes):
    """Build the same ResNet18 structure used during training."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(weights_path, device):
    """Load either a full checkpoint dict or a plain state_dict."""
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        class_to_idx = checkpoint.get("class_to_idx", DEFAULT_CLASS_TO_IDX)
    else:
        state_dict = checkpoint
        class_to_idx = DEFAULT_CLASS_TO_IDX

    return state_dict, class_to_idx


def list_image_paths(input_path):
    """Return one image path or all supported image files inside a folder."""
    if os.path.isfile(input_path):
        return [input_path]

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"找不到待预测路径: {input_path}")

    image_paths = []
    for name in sorted(os.listdir(input_path)):
        full_path = os.path.join(input_path, name)
        _, ext = os.path.splitext(name)
        if os.path.isfile(full_path) and ext.lower() in IMAGE_EXTENSIONS:
            image_paths.append(full_path)

    if not image_paths:
        raise FileNotFoundError(f"文件夹中没有可用图片: {input_path}")

    return image_paths


def load_model(weights_path, device):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到模型权重: {weights_path}")

    state_dict, class_to_idx = load_checkpoint(weights_path, device)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    model = build_model(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, class_to_idx, idx_to_class


def predict_single_image(model, image_path, idx_to_class, device, confidence_threshold):
    image = Image.open(image_path).convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu()
        top_k = min(2, probabilities.numel())
        top_probs, top_indices = torch.topk(probabilities, k=top_k)

    pred_idx = top_indices[0].item()
    pred_class = idx_to_class[pred_idx]
    pred_confidence = top_probs[0].item()
    confidence_text = "高置信度" if pred_confidence >= confidence_threshold else "低置信度"

    print(f"图片文件: {os.path.basename(image_path)}")
    print(f"预测类别: {pred_class}")
    print(f"置信度提示: {confidence_text} ({pred_confidence:.2%})")
    print("Top-2 概率:")
    for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1):
        class_name = idx_to_class[idx]
        print(f"  {rank}. {class_name}: {prob:.4f}")
    print()


def predict(input_path, weights_path, confidence_threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = list_image_paths(input_path)
    model, class_to_idx, idx_to_class = load_model(weights_path, device)

    print(f"使用设备: {device}")
    print(f"待预测图片数: {len(image_paths)}")
    print(f"类别映射: {class_to_idx}")
    print()

    for image_path in image_paths:
        predict_single_image(model, image_path, idx_to_class, device, confidence_threshold)


def main():
    args = parse_args()
    predict(args.input_path, args.weights, args.confidence_threshold)


if __name__ == "__main__":
    main()
