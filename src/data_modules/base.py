"""
基础分类数据集 - 支持 Transform
从原 dataset.py 迁移
"""
from torch.utils.data import Dataset
from PIL import Image
import os
import torch


class MyData(Dataset):
    """
    基础图像分类数据集
    
    Args:
        root_dir: 数据集根目录
        label_dir: 标签/类别文件夹名 (如 "ants", "bees")
        transform: 可选的数据变换
    
    Example:
        >>> dataset = MyData("hymenoptera_data/train", "ants", transform=my_transform)
        >>> img, label = dataset[0]
    """
    def __init__(self, root_dir, label_dir, transform=None):
        self.root_dir = root_dir   
        self.label_dir = label_dir
        self.transform = transform 
        
        # 拼接路径：hymenoptera_data/train/ants
        self.path = os.path.join(self.root_dir, self.label_dir) 
        
        # 拿到该文件夹下所有图片的文件名列表
        self.img_name_list = os.listdir(self.path) 

        # 为了配合神经网络的 Loss 计算把字符串标签映射为数字
        self.class_to_idx = {
            "ants": 0,
            "bees": 1
        }

    def __getitem__(self, idx):
        # 1. 拿到文件名
        img_name = self.img_name_list[idx]
        
        # 2. 拼接绝对路径
        img_item_path = os.path.join(self.path, img_name)
        
        # 3. 读取图片
        img = Image.open(img_item_path).convert('RGB') 

        # 4. 应用变换
        if self.transform:
            img = self.transform(img) 

        # 5. 处理标签
        label_str = self.label_dir
        label_idx = self.class_to_idx[label_str]
        label = torch.tensor(label_idx)

        return img, label

    def __len__(self):
        return len(self.img_name_list)
