"""
DataLoader 使用演示 - 从原 dataloader.py 迁移
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader


def main():
    # 1. 准备数据集
    test_data = torchvision.datasets.CIFAR10(
        root="../datasets",         
        train=False,                 
        transform=torchvision.transforms.ToTensor(), 
        download=True               
    )

    # 2. 创建DataLoader
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    # 3. 简单测试
    img, target = test_data[0]
    print(f"Image Shape: {img.shape}")
    print(f"Target Label: {target}")

    # 4. 初始化TensorBoard
    writer = SummaryWriter("dataloader")

    # 5. 模拟训练循环
    for epoch in range(2):
        step = 0
        
        for data in test_loader:
            imgs, targets = data
            writer.add_images("Epoch:{}".format(epoch), imgs, step)
            step = step + 1

    writer.close()
    print("✅ 完成！请运行: tensorboard --logdir=dataloader 查看结果")


if __name__ == "__main__":
    main()
