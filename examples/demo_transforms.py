"""
Transform 变换演示 - 从原 transform.py 提取的演示代码
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from src.transforms import load_image, plot_compare, train_transform, val_transform


def main():
    # --- 准备数据 ---
    img_path = r'data\train\bees_image\16838648_415acd9e3f.jpg' 
    try:
        img_pil = load_image(img_path)
    except Exception as e:
        print(e)
        return

    # --- 基础变换原理 (Tensor & Normalize) ---
    print("\n--- 正在演示基础 Tensor 变换 ---")

    to_tensor = transforms.ToTensor()
    tensor_img = to_tensor(img_pil)

    print(f"Tensor 形状: {tensor_img.shape}")
    print(f"Tensor 范围: [{tensor_img.min():.3f}, {tensor_img.max():.3f}]")

    # Normalize 演示
    norm_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    norm_img = norm_transform(tensor_img)
    print(f"标准化后范围: [{norm_img.min():.3f}, {norm_img.max():.3f}] (出现负数是正常的)")

    # --- 常用增强操作可视化 ---
    print("\n--- 正在演示单一变换效果 ---")

    transforms_dict = {
        "Resize (缩放)": transforms.Resize((256, 256)),
        "RandomCrop (随机裁剪)": transforms.RandomCrop((200, 200)),
        "RandomRotation (随机旋转)": transforms.RandomRotation(degrees=45),
        "ColorJitter (颜色抖动)": transforms.ColorJitter(brightness=0.5, contrast=0.5),
        "RandomHorizontalFlip (水平翻转)": transforms.RandomHorizontalFlip(p=1.0)
    }

    for name, transformer in transforms_dict.items():
        demo_imgs = [transformer(img_pil) for _ in range(3)]
        plot_compare(img_pil, demo_imgs, title_prefix=name)

    # --- 完整 Compose 流水线 ---
    print("\n--- 正在演示完整 Compose 流水线 ---")
    print("✅ 训练/验证流定义完成！")
    print(f"训练 transform: {train_transform}")
    print(f"验证 transform: {val_transform}")

    # --- TensorBoard 记录 ---
    visual_compose = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3)
    ])

    log_dir = "logs/transforms_demo"
    writer = SummaryWriter(log_dir)

    print(f"\n--- 正在写入 TensorBoard (路径: {log_dir}) ---")
    writer.add_image("Original", np.array(img_pil), global_step=0, dataformats='HWC')

    for i in range(10):
        aug_img = visual_compose(img_pil)
        writer.add_image("Augmented_Showcase", np.array(aug_img), global_step=i+1, dataformats='HWC')

    writer.close()
    print(f"✅ 完成！请在终端运行: tensorboard --logdir={log_dir} 查看结果")


if __name__ == "__main__":
    main()
