"""
=============================================================================
nn.MaxPool2d 学习脚本骨架
=============================================================================
学习目标：
1. 用一个很小的输入张量观察 MaxPool2d 的取最大值过程
2. 打印输入/输出张量以及它们的 shape
3. 结合 CIFAR-10 图片，观察池化前后的尺寸变化

说明：
- 本脚本故意保留了几个 TODO，方便你自己完成关键思考过程
- CIFAR-10 数据位置使用相对路径，默认指向项目外层的 ../datasets
=============================================================================
"""
import os

import torch
import torchvision
from torch import nn
from torchvision import transforms


# ============================================================================
# TODO 区：这些参数先留给你自己填写
# ============================================================================
POOL_KERNEL_SIZE = 3  # TODO: 填写 MaxPool2d 的 kernel_size
POOL_STRIDE = 1       # TODO: 填写 MaxPool2d 的 stride
POOL_PADDING = 0      # TODO: 填写 MaxPool2d 的 padding


def build_pool_layer():
    """创建池化层。"""
    if POOL_KERNEL_SIZE is None or POOL_STRIDE is None or POOL_PADDING is None:
        raise ValueError(
            "请先完成脚本顶部的 TODO：填写 kernel_size、stride、padding。"
        )

    return nn.MaxPool2d(
        kernel_size=POOL_KERNEL_SIZE,
        stride=POOL_STRIDE,
        padding=POOL_PADDING,
    )


def run_small_tensor_demo(pool_layer):
    """小张量示例：方便手算每个池化窗口的最大值。"""
    small_input = torch.tensor(
        [
            [
                [
                    [1.0, 3.0, 2.0, 4.0],
                    [5.0, 6.0, 1.0, 0.0],
                    [2.0, 8.0, 7.0, 3.0],
                    [4.0, 1.0, 9.0, 2.0],
                ]
            ]
        ]
    )

    small_output = pool_layer(small_input)

    print("=" * 60)
    print("1. 小张量示例（方便手算）")
    print("=" * 60)
    print("输入张量 small_input =")
    print(small_input)
    print(f"输入 shape: {small_input.shape}")
    print()
    print("输出张量 small_output =")
    print(small_output)
    print(f"输出 shape: {small_output.shape}")
    print()

    # TODO: 在这里手动推导 small_output 的 shape，并写下你的计算过程
    # shape: (1, 1, 2, 2)
    # TODO: 解释为什么池化前后通道数没有变化
    # 因为池化每个通道独立计算


def load_cifar10_sample():
    """读取一张 CIFAR-10 图片，观察池化后的尺寸变化。"""
    dataset_root = os.path.join("..", "datasets")

    try:
        dataset = torchvision.datasets.CIFAR10(
            root=dataset_root,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )
    except RuntimeError as exc:
        print("=" * 60)
        print("2. CIFAR-10 示例")
        print("=" * 60)
        print(f"读取 CIFAR-10 失败: {exc}")
        print(f"请确认数据集是否位于相对路径: {dataset_root}")
        return None, None, None

    image_tensor, label = dataset[0]
    class_name = dataset.classes[label]
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor, label, class_name


def run_cifar10_demo(pool_layer):
    """CIFAR-10 示例：重点观察 32x32 图片池化后的尺寸。"""
    image_tensor, label, class_name = load_cifar10_sample()
    if image_tensor is None:
        return

    pooled_tensor = pool_layer(image_tensor)

    print("=" * 60)
    print("2. CIFAR-10 示例（观察尺寸变化）")
    print("=" * 60)
    print(f"样本标签: {label} ({class_name})")
    print(f"输入 shape: {image_tensor.shape}")
    print(f"输出 shape: {pooled_tensor.shape}")
    print()
    print("输入张量的第 1 个通道:")
    print(image_tensor[0, 0])
    print()
    print("池化后的第 1 个通道:")
    print(pooled_tensor[0, 0])
    print()

    # TODO: 手动推导 CIFAR-10 图片池化后的输出 shape
    # shape:(1,3,30,30)
    # TODO: 用你自己的话说明“为什么通道数不变”
    # 因为池化每个通道独立计算
    # TODO: 补充注释：池化和卷积有什么区别
    # 池化是对输入特征图进行降采样，提取局部区域的最大值（MaxPool）或平均值（AvgPool），而卷积是通过卷积核对输入特征图进行加权求和，提取特征信息。池化不改变通道数，而卷积可以改变通道数。


def main():
    pool_layer = build_pool_layer()
    print(f"当前池化层: {pool_layer}")

    run_small_tensor_demo(pool_layer)
    run_cifar10_demo(pool_layer)


if __name__ == "__main__":
    main()
