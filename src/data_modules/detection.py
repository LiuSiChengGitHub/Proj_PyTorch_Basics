"""
目标检测数据集 - 图片 + 标签文件
从原 dataset_detection.py 提取 (移除测试代码)
"""
from torch.utils.data import Dataset
import cv2
import os


class DetectionDataset(Dataset):
    """
    目标检测数据集 (图片目录 + 标签目录分离)
    
    Args:
        root_dir: 数据集根目录
        img_dir: 图片文件夹名
        label_dir: 标签文件夹名 (txt文件)
    
    Example:
        >>> dataset = DetectionDataset("data/train", "ants_image", "ants_label")
        >>> img, label_content = dataset[0]
    """
    def __init__(self, root_dir, img_dir, label_dir):
        self.root_dir = root_dir   
        self.img_dir = img_dir
        self.label_dir = label_dir
        
        # 拼接出两个独立的路径
        self.img_path = os.path.join(self.root_dir, self.img_dir) 
        self.label_path = os.path.join(self.root_dir, self.label_dir) 
        
        # 获取文件名列表
        self.img_list = os.listdir(self.img_path) 
        self.label_list = os.listdir(self.label_path) 

    def __getitem__(self, idx):
        # 1. 获取图片文件名
        img_name = self.img_list[idx] 
        
        # 2. 读取图片数据
        img_item_path = os.path.join(self.img_path, img_name)
        img = cv2.imread(img_item_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. 推断标签文件名 (同名 .txt)
        label_name = img_name.split('.')[0] + '.txt'
        
        # 4. 拼接标签文件路径
        label_item_path = os.path.join(self.label_path, label_name)
        
        # 5. 读取标签内容
        label_content = ""
        try:
            with open(label_item_path, 'r', encoding='utf-8') as f:
                label_content = f.read()
        except FileNotFoundError:
            label_content = "No Label Found"
            
        return img, label_content

    def __len__(self):
        return len(self.img_list)
