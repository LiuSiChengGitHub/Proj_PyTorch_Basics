"""
分类数据集演示 - 从原 dataset_classification.py 提取的测试代码
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from src.data_modules import ClassificationDataset

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # 1. 设置路径
    root_dir = r"hymenoptera_data\train"          
    ants_label_dir = "ants"
    bees_label_dir = "bees"             

    # 2. 实例化对象
    ants_dataset = ClassificationDataset(root_dir, ants_label_dir) 
    bees_dataset = ClassificationDataset(root_dir, bees_label_dir)

    print(f"蚂蚁数据集长度: {len(ants_dataset)}")
    print(f"蜜蜂数据集长度: {len(bees_dataset)}")

    # 3. 读取并显示
    print("正在读取蜜蜂数据集的第 2 张图片...")
    img_bees, label_bees = bees_dataset[1] 
    plt.imshow(img_bees)
    plt.title(f"Label: {label_bees}")
    plt.show()

    print("正在读取蚂蚁数据集的第 1 张图片...")
    img_ants, label_ants = ants_dataset[0] 
    plt.imshow(img_ants)           
    plt.title(f"Label: {label_ants}")
    plt.show()


if __name__ == "__main__":
    main()
