import torch
import torchvision
import os
from torch.utils.data import DataLoader
from torch import nn  # 修复：必须导入 nn 才能继承 nn.Module
from torch.utils.tensorboard import SummaryWriter  # 修复：必须导入并初始化 SummaryWriter

# 1. 准备数据集 (CIFAR10 测试集)
dataset = torchvision.datasets.CIFAR10(
    root="../datasets", 
    train=False, 
    transform=torchvision.transforms.ToTensor(), 
    download=True
)

dataloader = DataLoader(dataset, batch_size=64) 

# 2. 定义神经网络结构
class Tudui(nn.Module): 
    def __init__(self): 
        super(Tudui, self).__init__() 
        # 输入3通道(RGB)，输出6通道(特征图)，卷积核3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x): 
        x = self.conv1(x) 
        return x 

tudui = Tudui()

# 3. 初始化 TensorBoard (修复：指定日志存放路径)
writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    
    # 打印形状用于 Debug
    # 输入: [64, 3, 32, 32] -> 输出: [64, 6, 30, 30]
    print(f"Step {step}: Input {imgs.shape}, Output {output.shape}") 
    
    # 可视化原始输入
    writer.add_images("input", imgs, step)
    
    # 4. 维度重塑 (Reshape)
    # 重点：TensorBoard 的 add_images 期望 3 通道图片。
    # 我们将 6 通道输出 [64, 6, 30, 30] 拆解为 [128, 3, 30, 30] 才能正常显示。
    output = torch.reshape(output, (-1, 3, 30, 30))
    
    # 可视化卷积后的特征图
    writer.add_images("output", output, step)
    
    step += 1

writer.close() # 养成好习惯：运行结束关闭 writer