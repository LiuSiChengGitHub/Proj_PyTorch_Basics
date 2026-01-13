# 1. 导入必要的库
from torch.utils.data import Dataset # PyTorch 数据集基类
import cv2                           # OpenCV，用来读图
import torch
import os                            # 用来处理文件路径
import matplotlib.pyplot as plt      # 用来画图

# ================= 0. 全局设置 =================
# 设置中文字体（避免 matplotlib 显示中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows常用 SimHei
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 2. 定义自定义数据集类
class MyData(Dataset): 
    
    # ------------------------------------------------------------------
    # 【第一步：初始化】
    # ------------------------------------------------------------------
    def __init__(self, root_dir, img_dir, label_dir):
        self.root_dir = root_dir   
        self.img_dir = img_dir
        self.label_dir = label_dir
        
        # 拼接出两个独立的路径：一个放图，一个放标签
        self.img_path = os.path.join(self.root_dir, self.img_dir) 
        self.label_path = os.path.join(self.root_dir, self.label_dir) 
        
        # 获取文件名列表
        self.img_list = os.listdir(self.img_path) 
        self.label_list = os.listdir(self.label_path) 

    # ------------------------------------------------------------------
    # 【第二步：获取单样本】
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        
        # 1. 获取图片文件名
        img_name = self.img_list[idx] 
        
        # 2. 读取图片数据
        img_item_path = os.path.join(self.img_path, img_name)
        img = cv2.imread(img_item_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转 RGB

        # 3. 【关键逻辑】推断标签文件名
        # 假设图片叫 "123.jpg"，标签叫 "123.txt"
        label_name = img_name.split('.')[0] + '.txt'
        
        # 4. 拼接标签文件的完整路径
        label_item_path = os.path.join(self.label_path, label_name)
        
        # 5. 读取 txt 文件里的具体内容
        label_content = ""
        try:
            with open(label_item_path, 'r', encoding='utf-8') as f: # 加上 encoding 防止中文乱码
                label_content = f.read()
        except FileNotFoundError:
            label_content = "No Label Found"
            
        # 6. 返回图片数据和读取到的文字标签
        return img, label_content

    # ------------------------------------------------------------------
    # 【第三步：获取长度】
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.img_list)


# ================== 以下是修改后的测试代码 ==================

# 1. 设置路径
# 根据你的截图 image_0ee16c.png，结构如下：
# data/train/
#    ├── ants_image/  <-- 这里改了
#    ├── ants_label/
#    ├── bees_image/  <-- 这里改了
#    ├── bees_label/

root_dir = r"data\train"          

# 🔴 核心修改：这里必须跟你的文件夹名字完全一致
ants_img_dir = "ants_image"       # 原来是 "ants"
bees_img_dir = "bees_image"       # 原来是 "bees"

ants_label_dir = "ants_label"
bees_label_dir = "bees_label"

# 2. 实例化
ants_dataset = MyData(root_dir, ants_img_dir, ants_label_dir) 
bees_dataset = MyData(root_dir, bees_img_dir, bees_label_dir)

print(f"蚂蚁数据集长度: {len(ants_dataset)}")
print(f"蜜蜂数据集长度: {len(bees_dataset)}")

# 3. 测试读取
# 读取第 0 张蚂蚁图
img_ants, label_ants = ants_dataset[0] 
print(f"标签内容: {label_ants}")
plt.imshow(img_ants)        
plt.title("Ants Example")
plt.show()

# 读取第 1 张蜜蜂图
img_bees, label_bees = bees_dataset[1] 
print(f"标签内容: {label_bees}")
plt.imshow(img_bees)          
plt.title("Bees Example")
plt.show()