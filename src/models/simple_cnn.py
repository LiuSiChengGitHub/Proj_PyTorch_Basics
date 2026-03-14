"""
=============================================================================
SimpleCNN 模型定义
=============================================================================
学习目标：
1. 把之前学过的 Conv2d、MaxPool2d、ReLU、Linear、Flatten 组装成完整模型
2. 理解 feature（特征提取）+ classifier（分类器）的经典拆分
3. 手推每层的 tensor shape 变化，确保维度正确贯通
4. 适配 CIFAR-10：输入 (B, 3, 32, 32) → 输出 (B, 10)

说明：
- 架构沿用 examples/nn_sequential.py 的设计（3 层 Conv + 3 层 Pool + 2 层 Linear）
- 在每个 Conv2d 后加 ReLU 激活函数（之前 nn_sequential 里省略了）
- TODO 标记的地方需要你填写参数，旁边的注释标注了 shape 变化
=============================================================================
"""
import torch
from torch import nn


class SimpleCNN(nn.Module):
    """适配 CIFAR-10 的简单 CNN 分类模型。

    整体结构：
        feature（特征提取）：Conv2d → ReLU → MaxPool2d × 3 层
        classifier（分类器）：Flatten → Linear → Linear
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # =====================================================================
        # 特征提取部分：3 组 Conv → ReLU → Pool
        # =====================================================================
        self.feature = nn.Sequential(
            # --- 第 1 组 ---
            # TODO: 填写 Conv2d 参数
            # 输入: (B, 3, 32, 32) → 输出: (B, 32, 32, 32)
            # 提示: in_channels=3, out_channels=32, kernel_size=5, padding=?（要保持 H/W 不变）
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),  # TODO

            # TODO: 在这里加 ReLU
            nn.ReLU(),

            # TODO: 填写 MaxPool2d 参数
            # 输入: (B, 32, 32, 32) → 输出: (B, 32, 16, 16)
            # 提示: kernel_size=2，H/W 减半
            nn.MaxPool2d(kernel_size=2, stride=2),  # TODO

            # --- 第 2 组 ---
            # TODO: 填写 Conv2d 参数
            # 输入: (B, 32, 16, 16) → 输出: (B, 32, 16, 16)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # TODO

            # TODO: 在这里加 ReLU
            nn.ReLU(),

            # TODO: 填写 MaxPool2d 参数
            # 输入: (B, 32, 16, 16) → 输出: (B, 32, 8, 8)
            nn.MaxPool2d(kernel_size=2, stride=2),  # TODO

            # --- 第 3 组 ---
            # TODO: 填写 Conv2d 参数
            # 输入: (B, 32, 8, 8) → 输出: (B, 64, 8, 8)
            # 提示: 这一层 out_channels 变大了
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # TODO

            # TODO: 在这里加 ReLU
            nn.ReLU(),

            # TODO: 填写 MaxPool2d 参数
            # 输入: (B, 64, 8, 8) → 输出: (B, 64, 4, 4)
            nn.MaxPool2d(kernel_size=2, stride=2),  # TODO
        )

        # =====================================================================
        # 分类器部分：展平 → 全连接
        # =====================================================================
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 输入: (B, 64*4*4) = (B, 1024) → 输出: (B, 64)
            # TODO: 填写 Linear 的 in_features
            # 提示: 需要手推 feature 最后的输出 shape，然后算 C*H*W
            nn.Linear(1024, 64),  # TODO: in_features = ?

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)      # (B, 64, 4, 4)
        x = self.classifier(x)   # (B, num_classes)
        return x


# =============================================================================
# 快速验证（直接运行本文件）
# =============================================================================
if __name__ == "__main__":
    model = SimpleCNN()
    print(model)
    print()

    # 模拟 CIFAR-10 输入
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {output.shape}")
    print(f"期望输出: torch.Size([1, 10])")
