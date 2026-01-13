from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import DataLoader

# 1. 准备数据集
test_data = torchvision.datasets.CIFAR10(
    root="../datasets",         
    train=False,                 
    # transform的作用：
    # 1. 将PIL Image或numpy.ndarray转换为tensor
    # 2. 将数据范围从[0, 255]归一化到[0.0, 1.0]
    # 3. 调整维度顺序为 (Channel, Height, Width)
    transform=torchvision.transforms.ToTensor(), 
    download=True               
)

# 2. 创建DataLoader
# DataLoader的作用是将Dataset里的数据打包，便于循环读取
test_loader = DataLoader(
    dataset=test_data,
    batch_size=64,      # 每次从数据集中取64张图片打包成一个Batch
    shuffle=True,       # 打乱数据顺序（每次取出的64张都不一样，增加随机性）
    num_workers=0,      # 加载数据的子进程数，0表示在主进程中加载（Windows系统下通常设为0以免报错）
    drop_last=False     # 如果数据集总数除以64有余数，False表示"不丢弃"最后剩下的那点数据
)

# 3. 简单测试：查看第一张图片的信息
img, target = test_data[0]       # 从数据集中解包拿到第一组数据（img是图片张量，target是标签索引）
print(f"Image Shape: {img.shape}") # 输出应该是 torch.Size([3, 32, 32]) -> (通道数, 高, 宽)
print(f"Target Label: {target}")   # 输出一个整数，代表类别索引

# 4. 初始化TensorBoard写入器
# 运行后会生成一个名为 "dataloader" 的文件夹存放日志文件
writer = SummaryWriter("dataloader")

# 5. 开始模拟训练/测试循环
for epoch in range(2):           # 外层循环：模拟训练2轮 (Epoch 0, Epoch 1)
    step = 0                     # 初始化步数，每个Epoch开始时重置为0
    
    # 内层循环：遍历DataLoader
    # 每次循环，DataLoader会"吐出"一批数据（64张图 + 64个标签）
    for data in test_loader:
        imgs, targets = data     # 解包：imgs形状为 [64, 3, 32, 32]，targets形状为 [64]
        
        # print(imgs.shape)
        # print(targets)
        
        # 将这一批图片记录到TensorBoard中
        # 这里的tag使用了 "Epoch:{}".format(epoch)，意味着每个Epoch会分开显示
        # step 是横坐标，表示这是第几批数据
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        
        step = step + 1          # 记录完一次，步数加1

# 关闭writer，确保数据写入磁盘
writer.close()