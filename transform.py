import torch
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# ================= 0. 全局设置 =================
# 设置中文字体（避免 matplotlib 显示中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows常用 SimHei，Mac可用 Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ================= 1. 辅助函数定义 =================
def load_image(path):
    """读取图片并进行基础检查"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 找不到图片: {path}，请检查路径！")
    
    # 始终推荐使用 PIL 读取，因为 torchvision 默认就是针对 PIL 设计的
    img = Image.open(path)
    print(f"✅ 图片读取成功 | 尺寸: {img.size} (宽x高) | 模式: {img.mode}")
    return img

def plot_compare(orig_img, trans_imgs, title_prefix="Transform"):
    """
    通用绘图函数：对比原图和变换后的图片
    :param orig_img: 原始 PIL 图片
    :param trans_imgs: 变换后的图片列表 (list of PIL Images)
    :param title_prefix: 标题前缀
    """
    count = len(trans_imgs) + 1
    # 动态计算子图布局，最多一行放 4 张
    cols = min(count, 4)
    rows = (count - 1) // 4 + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    
    # 处理 axes 为单个对象或一维数组的情况，统一转为列表处理
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # 1. 画原图
    axes[0].imshow(orig_img)
    axes[0].set_title("原始图片", fontweight='bold')
    axes[0].axis('off')

    # 2. 画变换图
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

# ================= 2. 主逻辑 =================

# --- 2.1 准备数据 ---
# 请确保此路径存在，或者修改为你自己的图片路径
img_path = r'data\train\bees_image\16838648_415acd9e3f.jpg' 
try:
    img_pil = load_image(img_path)
except Exception as e:
    print(e)
    # 如果找不到图片，这一行会让程序安全停止，方便你去修路径
    exit() 

# --- 2.2 基础变换原理 (Tensor & Normalize) ---
print("\n--- 正在演示基础 Tensor 变换 ---")

# 实例化 ToTensor
to_tensor = transforms.ToTensor()
tensor_img = to_tensor(img_pil)

print(f"Tensor 形状: {tensor_img.shape}")  # (C, H, W) -> PyTorch 格式 (通道, 高, 宽)
print(f"Tensor 范围: [{tensor_img.min():.3f}, {tensor_img.max():.3f}]") # 0.0 ~ 1.0

# 🔴 难点解析：为什么要 .permute(1, 2, 0)？
# PyTorch (机器看) 格式: (C, H, W) -> (3, 512, 768)
# Matplotlib (人眼看) 格式: (H, W, C) -> (512, 768, 3)
# permute 就是负责把维度搬运回去，否则画图会报错。
img_for_plt = tensor_img.permute(1, 2, 0)

# 定义标准化 (使用 ImageNet 统计值)
# 公式: output = (input - mean) / std
# 作用: 把数据拉回 0 附近，加速神经网络收敛
norm_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)
norm_img = norm_transform(tensor_img)
print(f"标准化后范围: [{norm_img.min():.3f}, {norm_img.max():.3f}] (出现负数是正常的)")


# --- 2.3 常用增强操作可视化 (去除了冗余代码) ---
print("\n--- 正在演示单一变换效果 ---")

# 为了减少代码冗余，我们把变换定义在一个字典里，循环展示
transforms_dict = {
    "Resize (缩放)": transforms.Resize((256, 256)),
    "RandomCrop (随机裁剪)": transforms.RandomCrop((200, 200)),
    "RandomRotation (随机旋转)": transforms.RandomRotation(degrees=45),
    "ColorJitter (颜色抖动)": transforms.ColorJitter(brightness=0.5, contrast=0.5),
    "RandomHorizontalFlip (水平翻转)": transforms.RandomHorizontalFlip(p=1.0) # p=1.0 强制翻转
}

# 循环演示每个变换
# 为了人眼看，还是在对img_pil这个原材料进行操作
for name, transformer in transforms_dict.items():
    # 生成 3 张效果图来观察随机性
    demo_imgs = [transformer(img_pil) for _ in range(3)]
    plot_compare(img_pil, demo_imgs, title_prefix=name)


# --- 2.4 核心：完整的 Compose 流水线 ---
print("\n--- 正在演示完整 Compose 流水线 ---")

# 【训练集】需要“折腾”图片，增加数据多样性
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),              # 1. 先放大一点
    transforms.RandomCrop((224, 224)),          # 2. 随机切出核心区域
    transforms.RandomHorizontalFlip(p=0.5),     # 3. 随机翻转
    transforms.RandomRotation(degrees=15),      # 4. 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # 5. 颜色增强
    transforms.ToTensor(),                      # 6. 转 Tensor (0-1)
    transforms.Normalize(                       # 7. 标准化 (变负数)
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 【验证集】必须固定，不能有随机性
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),          # ⚠️ 注意：验证集用 CenterCrop
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

print("✅ 训练/验证流定义完成！")


# --- 2.5 只有“视觉”变换的流水线 (用于 TensorBoard 展示) ---
# 专门定义一个不带 Normalize 的 compose，方便人类观察
visual_compose = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3)
    # ❌ 不加 ToTensor 和 Normalize，保持 PIL 格式方便画图
])

# 记录到 TensorBoard
log_dir = "logs/transforms_demo"
writer = SummaryWriter(log_dir)

# 记录 10 张增强后的图
print(f"\n--- 正在写入 TensorBoard (路径: {log_dir}) ---")
# 先记一张原图
writer.add_image("Original", np.array(img_pil), global_step=0, dataformats='HWC')

for i in range(10):
    aug_img = visual_compose(img_pil)
    # 注意：add_image 需要 numpy 数组或 tensor
    writer.add_image("Augmented_Showcase", np.array(aug_img), global_step=i+1, dataformats='HWC')

writer.close()
print(f"✅ 完成！请在终端运行: tensorboard --logdir={log_dir} 查看结果")