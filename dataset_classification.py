# 1. 导入必要的库
from torch.utils.data import Dataset #这是 PyTorch 提供的标准“模版”，所有自定义数据集都必须继承它
import cv2                           # OpenCV库，用来读取图片（读取速度快，但默认是 BGR 颜色通道）
import torch
import os                            # Python 标准库，用来处理文件路径（比如拼接路径、列出文件名）
import matplotlib.pyplot as plt      # 画图库，用来把读取到的数字矩阵画成我们可以看的图片

# 2. 定义自定义数据集类
class MyData(Dataset): # 继承 Dataset 类，相当于告诉 PyTorch：“我写的这个类是用来处理数据的”
    
    # ------------------------------------------------------------------
    # 【第一步：初始化】
    # 就像去图书馆办卡，只要告诉我书在哪里（路径），以及有哪些书（文件名列表）
    # 注意：这里一般【不读取】图片数据，只准备路径，防止内存爆炸
    # ------------------------------------------------------------------
    def __init__(self, root_dir, label_dir):
        """
        :param root_dir: 数据集的根目录（例如："../test_jpg"）
        :param label_dir: 子文件夹名（例如："ants"），在这里它既是文件夹名，也被我们当作标签名
        """
        # 将传入的参数保存到类内部，方便其他函数（如 __getitem__）使用
        self.root_dir = root_dir   
        self.label_dir = label_dir
        
        # 拼接完整路径。
        # 使用 os.path.join 的好处是它会自动适配 Windows(\) 和 Linux(/) 的路径分隔符
        # 结果类似于： "D:/data/test_jpg/ants"
        self.path = os.path.join(self.root_dir, self.label_dir) 
        
        # os.listdir 会去这个路径下看一眼，把所有文件的名字变成一个列表
        # 结果类似于：['001.jpg', '002.jpg', '003.jpg']
        # 这一步非常重要，因为后续我们要靠索引（0, 1, 2...）来在这个列表里找文件
        self.img_path = os.listdir(self.path) 

    # ------------------------------------------------------------------
    # 【第二步：获取单样本】
    # 这是最核心的函数！PyTorch 训练时会不断调用这个函数。
    # 当你写 dataset[0] 时，其实就是在调用 __getitem__(0)
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        # 1. 根据索引 (idx) 从刚才的文件名列表中拿到具体的文件名
        # 例如 idx=0，拿到 '001.jpg'
        img_name = self.img_path[idx] 
        
        # 2. 再次拼接路径，获得这张图片的“绝对地址”
        # 结果类似于： "D:/data/test_jpg/ants/001.jpg"
        img_item_path = os.path.join(self.path, img_name)
        
        # 3. 【真正读取数据】
        # 使用 OpenCV 读取图片。注意：此时 img 是一个 NumPy 数组（数字矩阵）
        # OpenCV 默认读取格式是 BGR (蓝绿红)，跟我们需要的不一样
        img = cv2.imread(img_item_path)
        
        # 4. 【关键修正】颜色空间转换
        # 因为后面的 Matplotlib (plt) 和 PyTorch 默认都是 RGB 顺序
        # 如果不转，显示的图片颜色会很奇怪（人脸发蓝）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 5. 准备标签
        # 在这个简单的例子里，我们直接把文件夹的名字（"ants"）当作标签
        label = self.label_dir
        
        # 6. 返回结果
        # 必须返回图片和标签，这样训练的时候才能告诉模型：“这张图(img) 是 机器人(label)”
        return img, label

    # ------------------------------------------------------------------
    # 【第三步：获取长度】
    # 告诉 PyTorch 数据集一共有多少张图
    # 这样 DataLoader 才知道一个 epoch 需要循环多少次
    # ------------------------------------------------------------------
    def __len__(self):
        # 列表的长度就是图片的数量
        return len(self.img_path)


# ================== 以下是测试代码 ==================

# 1. 设置路径
# ".." 表示上一级目录。请确保你的硬盘上真的有这个文件夹结构，否则会报错
root_dir = r"hymenoptera_data\train"          
ants_label_dir = "ants"
bees_label_dir = "bees"             

# 2. 实例化对象（并没有开始读取图片，只是建立了索引）
# 创建了两个独立的数据集对象：一个只包含机器人，一个只包含车
ants_dataset = MyData(root_dir, ants_label_dir) 
bees_dataset = MyData(root_dir, bees_label_dir)

# 打印一下机器人数据集有多少张图
print(f"蚂蚁数据集长度: {len(ants_dataset)}")
print(f"蜜蜂数据集长度: {len(bees_dataset)}")


# 3. 真正读取并显示（调用 __getitem__）
# bees_dataset[1] 自动触发 __getitem__(1)
print("正在读取蜜蜂数据集的第 2 张图片...")
img_bees, label_bees = bees_dataset[1] 
plt.imshow(img_bees)           # 把数字矩阵画成图
plt.title(f"Label: {label_bees}") # 给图片加个标题
plt.show()                    # 弹窗显示

# ants_dataset[0] 自动触发 __getitem__(0)
print("正在读取蚂蚁数据集的第 1 张图片...")
img_ants, label_ants = ants_dataset[0] 
plt.imshow(img_ants)           
plt.title(f"Label: {label_ants}")
plt.show()