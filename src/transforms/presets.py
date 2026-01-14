"""
预定义的数据变换 - 从原 transform.py 提取可复用部分
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    


def load_image(path):
    """
    读取图片并进行基础检查
    
    Args:
        path: 图片路径
    
    Returns:
        PIL.Image 对象
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 找不到图片: {path}，请检查路径！")
    
    img = Image.open(path)
    print(f"✅ 图片读取成功 | 尺寸: {img.size} (宽x高) | 模式: {img.mode}")
    return img


def plot_compare(orig_img, trans_imgs, title_prefix="Transform"):
    """
    通用绘图函数：对比原图和变换后的图片
    
    Args:
        orig_img: 原始 PIL 图片
        trans_imgs: 变换后的图片列表 (list of PIL Images)
        title_prefix: 标题前缀
    """
    count = len(trans_imgs) + 1
    cols = min(count, 4)
    rows = (count - 1) // 4 + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # 画原图
    axes[0].imshow(orig_img)
    axes[0].set_title("原始图片", fontweight='bold')
    axes[0].axis('off')

    # 画变换图
    for i, img in enumerate(trans_imgs):
        if i + 1 < len(axes):
            axes[i+1].imshow(img)
            axes[i+1].set_title(f"{title_prefix} #{i+1}")
            axes[i+1].axis('off')
    
    # 隐藏多余的空子图
    for j in range(count, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"{title_prefix} 效果展示", fontsize=14)
    plt.tight_layout()
    plt.show()


# ================= 预定义的 Compose 流水线 =================

# 【训练集】需要"折腾"图片，增加数据多样性
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),              # 1. 先放大一点
    transforms.RandomCrop((224, 224)),          # 2. 随机切出核心区域
    transforms.RandomHorizontalFlip(p=0.5),     # 3. 随机翻转
    transforms.RandomRotation(degrees=15),      # 4. 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 5. 颜色增强
    transforms.ToTensor(),                      # 6. 转 Tensor (0-1)
    transforms.Normalize(                       # 7. 标准化
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 【验证集】必须固定，不能有随机性
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),          # 验证集用 CenterCrop
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
