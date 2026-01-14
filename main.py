# 导入库
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms # 将其重命名为 transforms，方便调用

# 导入类 (从重构后的 src 模块)
from src.data_modules import MyData

# 传入参数
root = r"hymenoptera_data\train"
pics = "ants"

# 定义transform
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                      
    transforms.Normalize(                      
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 实例化dataset
data_dataset = MyData(
    root_dir= root, 
    label_dir= pics, 
    transform=data_transform
)


# 实例化dataloader
data_dataloader = DataLoader(
    dataset = data_dataset,         
    batch_size=4,      
    shuffle=True,      
    num_workers=0,
    drop_last=False     
)


# 运行
for i,data in enumerate(data_dataloader):
    imgs,labels = data
    print(f"图片 Batch 形状: {imgs.shape}") 
    print(f"标签列表: {labels}")
    break