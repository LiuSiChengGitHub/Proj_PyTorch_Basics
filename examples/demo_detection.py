"""
检测数据集演示 - 从原 dataset_detection.py 提取的测试代码
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from src.data_modules import DetectionDataset

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # 1. 设置路径
    root_dir = r"data\train"          

    ants_img_dir = "ants_image"      
    bees_img_dir = "bees_image"       

    ants_label_dir = "ants_label"
    bees_label_dir = "bees_label"

    # 2. 实例化
    ants_dataset = DetectionDataset(root_dir, ants_img_dir, ants_label_dir) 
    bees_dataset = DetectionDataset(root_dir, bees_img_dir, bees_label_dir)

    print(f"蚂蚁数据集长度: {len(ants_dataset)}")
    print(f"蜜蜂数据集长度: {len(bees_dataset)}")

    # 3. 测试读取
    img_ants, label_ants = ants_dataset[0] 
    print(f"标签内容: {label_ants}")
    plt.imshow(img_ants)        
    plt.title("Ants Example")
    plt.show()

    img_bees, label_bees = bees_dataset[1] 
    print(f"标签内容: {label_bees}")
    plt.imshow(img_bees)          
    plt.title("Bees Example")
    plt.show()


if __name__ == "__main__":
    main()
