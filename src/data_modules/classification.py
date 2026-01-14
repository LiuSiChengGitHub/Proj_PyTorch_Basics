"""
简单分类数据集 - 使用 OpenCV 读取
从原 dataset_classification.py 提取 (移除测试代码)
"""
from torch.utils.data import Dataset
import cv2
import os


class ClassificationDataset(Dataset):
    """
    简单图像分类数据集 (使用 OpenCV)
    
    Args:
        root_dir: 数据集根目录
        label_dir: 标签/类别文件夹名
    
    Example:
        >>> dataset = ClassificationDataset("hymenoptera_data/train", "ants")
        >>> img, label = dataset[0]  # img 是 numpy array (H, W, C) RGB格式
    """
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir   
        self.label_dir = label_dir
        
        # 拼接完整路径
        self.path = os.path.join(self.root_dir, self.label_dir) 
        
        # 获取文件名列表
        self.img_path = os.listdir(self.path) 

    def __getitem__(self, idx):
        # 1. 根据索引获取文件名
        img_name = self.img_path[idx] 
        
        # 2. 拼接完整路径
        img_item_path = os.path.join(self.path, img_name)
        
        # 3. 读取数据 (cv2 默认 BGR)
        img = cv2.imread(img_item_path)
        
        # 4. 颜色空间转换 (BGR -> RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 5. 标签是文件夹名
        label = self.label_dir
        
        return img, label

    def __len__(self):
        return len(self.img_path)
