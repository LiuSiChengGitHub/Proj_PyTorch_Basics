
# PyTorch 基础学习项目 — 完整学习笔记

> 从 Dataset 到迁移学习的完整学习链路 项目：Proj_Pytorch_Basics

---

## 一、项目概览

### 项目简介

这是一个围绕图像任务逐步学习 PyTorch 的入门项目，从零开始掌握数据加载、神经网络层、CNN 结构、训练循环，直到迁移学习。

项目采用"边学边拆分"的工程化组织方式：

- `src/` 放可复用核心模块（工具箱）
- `examples/` 放阶段性实验脚本（练兵场）
- `train.py` 完成 CIFAR-10 上的完整训练循环
- `docs/` 存放学习笔记和计划文档

项目同时服务两条学习线：

- 蚂蚁/蜜蜂分类与 YOLO 标注数据：理解真实项目的数据组织方式
- CIFAR-10：理解通用层操作和训练流程

### 环境与依赖

| 项目          | 版本                                      |
| ----------- | --------------------------------------- |
| GPU         | NVIDIA GeForce RTX 3060 / 4060 LAPTOP   |
| CUDA        | 11.8                                    |
| PyTorch     | 2.0.0                                   |
| torchvision | 0.15.0                                  |
| Python      | 3.9                                     |
| 其他          | OpenCV 4.8、NumPy、Matplotlib、TensorBoard |

```bash
conda env create -f environment.yml
conda activate pytorch_basics
```

### 学习进度总览

|阶段|内容|状态|
|---|---|---|
|数据加载|Dataset → Transform → DataLoader|✅|
|神经网络层|nn.Module, Conv2d, MaxPool2d, ReLU, Linear, Sequential|✅|
|损失与优化|CrossEntropyLoss, SGD/Adam, backward|✅|
|SimpleCNN 训练|CIFAR-10 完整训练循环 + 数据增强对比|✅|
|迁移学习|ResNet18 预训练 + 渐进解冻|✅|
|→ YOLO 缺陷检测|Week 3 核心项目|待启动|

---

## 二、工程化项目结构

### 为什么不把代码写在一个文件里

初学者常见做法是一个 500 行的 `.py` 文件塞下所有逻辑。问题：可维护性差、无法复用、团队协作冲突、无法独立测试。

工程化做法：按功能分模块。

```
Proj_Pytorch_Basics/
├── src/                          # 核心可复用模块
│   ├── __init__.py               # 懒加载入口
│   ├── data_modules/             # Dataset 定义
│   │   ├── __init__.py
│   │   ├── base.py               # MyData：PIL 分类数据集
│   │   ├── classification.py     # ClassificationDataset：OpenCV 版
│   │   └── detection.py          # DetectionDataset：YOLO 格式
│   ├── transforms/
│   │   ├── __init__.py
│   │   └── presets.py            # train/val transform + 可视化工具
│   └── models/
│       ├── __init__.py
│       └── simple_cnn.py         # SimpleCNN 模型定义
├── examples/                     # 知识点演示脚本（练兵场）
│   ├── test.py                   # Tensor / NumPy 前向传播
│   ├── demo_classification.py    # 分类 Dataset 读取
│   ├── demo_detection.py         # YOLO 检测数据读取
│   ├── demo_dataloader.py        # DataLoader 批量加载
│   ├── demo_transforms.py        # 图像变换与可视化
│   ├── study_nn_layers.py        # nn.Module 最小模板
│   ├── nn_conv2d.py              # Conv2d 输出观察
│   ├── nn_maxpool2d.py           # MaxPool2d 输出观察
│   ├── nn_relu.py                # ReLU 激活函数
│   ├── nn_linear.py              # Linear + Flatten
│   ├── nn_sequential.py          # Sequential 搭 CNN
│   ├── nn_loss.py                # 损失函数 + backward
│   ├── nn_optimizer.py           # 优化器 + 完整训练步骤
│   ├── train_with_aug.py         # 数据增强对比实验
│   ├── train_transfer.py         # ResNet18 迁移学习
│   └── predict_transfer.py       # 单图/批量推理
├── main.py                       # Dataset + Transform + DataLoader 串联
├── train.py                      # CIFAR-10 完整训练脚本
├── saved_models/ # 训练保存的模型（.gitignore）
├── docs/                         # 学习笔记和计划文档
├── data/                         # 蚂蚁/蜜蜂检测数据（含 YOLO 标注）
├── hymenoptera_data/             # ImageFolder 格式分类数据
├── logs/                         # TensorBoard 日志
├── environment.yml               # Conda 环境配置
├── CLAUDE.md                     # Claude Code 项目上下文
└── README.md
```

### 关键设计原则

|文件夹|职责|设计原则|
|---|---|---|
|`src/`|可复用核心代码|只放类定义，不放执行逻辑|
|`examples/`|演示和实验|可以随意写测试逻辑|
|`data/`|数据存放|在 .gitignore 中，不提交|
|`main.py`|组装入口|导入 src 模块并运行|

**核心原则：类定义与执行逻辑分离。** `src/` 的代码应该可以被 `import` 而不触发任何执行。

### `__init__.py` 的作用

```python
# src/data_modules/__init__.py
from .base import MyData
from .classification import ClassificationDataset

# 有了这个文件，才能：
from src.data_modules import MyData, ClassificationDataset
# 否则报 ModuleNotFoundError
```

`__init__.py` 让文件夹变成 Python 包（Package）。项目中使用懒加载模式（`__getattr__`），避免无关依赖被提前导入。

### 重要路径约定

|数据/目录|说明|路径|
|---|---|---|
|`hymenoptera_data/`|蚂蚁/蜜蜂分类数据集|项目内|
|`data/`|蚂蚁/蜜蜂检测数据 + YOLO 标注|项目内|
|CIFAR-10|训练用公共数据集|项目外 `../datasets`|
|`logs/`|TensorBoard 日志|项目内|

公开数据集不提交到仓库，通过相对路径 `../datasets` 引用。

---

## 三、数据加载：Dataset → Transform → DataLoader

> 一句话总结：Dataset 负责"怎么读一条数据"，Transform 负责"怎么处理这条数据"，DataLoader 负责"怎么批量送给模型"。

### 3.1 Dataset：数据的"仓库"

继承 `torch.utils.data.Dataset`，实现三个方法：

|方法|作用|调用时机|面试考点|
|---|---|---|---|
|`__init__`|初始化，准备文件路径列表|创建对象时|只建立索引，不读图（惰性加载）|
|`__getitem__(idx)`|根据索引读取单个样本|`dataset[0]` 时|返回 (img, label) 元组|
|`__len__`|返回数据集总长度|`len(dataset)` 时|DataLoader 用它决定迭代次数|

**惰性加载（Lazy Loading）**：`__init__` 只建立文件名列表索引，不读取图片内容。真正读图发生在 `__getitem__` 被调用时。好处：可处理 TB 级数据集而不爆内存。

#### 代码逐行解析

```python
def __init__(self, root_dir, label_dir, transform=None):
    self.root_dir = root_dir      # "hymenoptera_data/train"
    self.label_dir = label_dir    # "ants"
    self.transform = transform
    self.path = os.path.join(self.root_dir, self.label_dir)
    self.img_name_list = os.listdir(self.path)  # ["ant_001.jpg", ...]

    # 标签映射：神经网络只认数字，不认字符串
    self.class_to_idx = {"ants": 0, "bees": 1}

def __getitem__(self, idx):
    img_name = self.img_name_list[idx]
    img_item_path = os.path.join(self.path, img_name)

    # .convert('RGB')：防止 PNG(4通道) 或灰度图(1通道) 导致维度错误
    img = Image.open(img_item_path).convert('RGB')

    if self.transform:
        img = self.transform(img)

    # 标签必须转 Tensor，PyTorch 的 loss 函数期望 Tensor 类型
    label = torch.tensor(self.class_to_idx[self.label_dir])
    return img, label

def __len__(self):
    return len(self.img_name_list)
```

核心流程：`索引 idx → 文件名 → 完整路径 → 读取图片 → Transform → 返回 (img, label)`

#### 项目中的三个 Dataset 类（学习迭代）

|类名|文件|读图方式|返回值|作用|
|---|---|---|---|---|
|`MyData`|`base.py`|PIL|`(image_tensor, label_tensor)`|正式训练用|
|`ClassificationDataset`|`classification.py`|OpenCV|`(numpy_image, label_str)`|理解 Dataset 原理|
|`DetectionDataset`|`detection.py`|OpenCV|`(numpy_image, yolo_label_text)`|理解检测数据格式|

体现了学习迭代：简化版理解原理 → 规范版用于训练 → 检测版理解标注格式。

### 3.2 Transform：数据的"加工厂"

原始数据的问题：尺寸不一致、数据类型不对（PIL 不是 Tensor）、数值范围不标准（0-255）。

```
原始图片 (PIL) → Resize → ToTensor → Normalize → 可训练的 Tensor
```

#### 常用变换

```python
# 基础变换
transforms.Resize((256, 256))       # 统一尺寸
transforms.ToTensor()                # PIL → Tensor, [0,255] → [0,1], (H,W,C) → (C,H,W)
transforms.Normalize(mean, std)      # 标准化: output = (input - mean) / std

# 训练集数据增强
transforms.RandomCrop((224, 224))           # 随机裁剪
transforms.RandomHorizontalFlip(p=0.5)      # 随机水平翻转
transforms.RandomRotation(degrees=15)       # 随机旋转
transforms.ColorJitter(brightness=0.2)      # 颜色抖动
```

#### 训练集 vs 验证集的 Transform

```python
# 训练集：带随机增强，增加多样性
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集：固定处理，保证可复现
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),   # 中心裁剪，不随机
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

|变换|训练集|验证集|原因|
|---|---|---|---|
|RandomCrop|✅|❌|增加多样性|
|CenterCrop|❌|✅|保持一致性|
|RandomFlip|✅|❌|数据增强|
|ToTensor|✅|✅|必需|
|Normalize|✅|✅|必需|

**顺序不能错**：`Normalize` 必须在 `ToTensor` 之后（Normalize 需要 Tensor 输入）。

**不同数据集的 Normalize 值**：

|数据集|mean|std|
|---|---|---|
|CIFAR-10|[0.4914, 0.4822, 0.4465]|[0.2023, 0.1994, 0.2010]|
|ImageNet|[0.485, 0.456, 0.406]|[0.229, 0.224, 0.225]|

### 3.3 DataLoader：数据的"搬运工"

直接用 Dataset 遍历：一次只能拿 1 个样本，无法批量、无法打乱、无法多进程。DataLoader 解决这些问题。

```python
data_dataloader = DataLoader(
    dataset=data_dataset,
    batch_size=4,       # 每批 4 个样本
    shuffle=True,       # 打乱顺序（训练集用）
    num_workers=0,      # Windows 设 0
    drop_last=False     # 是否丢弃最后不足的批次
)
```

|参数|训练集推荐|验证集推荐|说明|
|---|---|---|---|
|`batch_size`|16/32/64|32/64|显存不够就减半|
|`shuffle`|True|False|训练需要随机性，验证需要确定性|
|`num_workers`|0（Windows）|0|Linux 可设 CPU 核心数|
|`drop_last`|True|False|训练丢弃防止 BN 层异常，验证保留全部|

**单样本 vs 批次的 shape 变化**：

```python
# Dataset 返回单个样本
img, label = dataset[0]
print(img.shape)    # torch.Size([3, 224, 224])      ← 3 维
print(label.shape)  # torch.Size([])                  ← 标量

# DataLoader 返回批次
imgs, labels = next(iter(dataloader))
print(imgs.shape)   # torch.Size([4, 3, 224, 224])   ← 4 维（多了 batch）
print(labels.shape) # torch.Size([4])                 ← 1 维
```

### 3.4 完整数据流转

```
📁 磁盘图片 (hymenoptera_data/train/ants/*.jpg)
  ↓
Dataset.__init__()  →  建立文件名索引（惰性，不读图）
  ↓  DataLoader 请求数据
Dataset.__getitem__(idx)
  ├─ 读取图片 (PIL.Image.open)
  ├─ Transform 处理 (Resize → RandomCrop → ToTensor → Normalize)
  └─ 返回 (img_tensor, label)    shape: (3, 224, 224), scalar
  ↓
DataLoader 收集 batch_size 个样本
  ↓
打包成批次: (batch_imgs, batch_labels)   shape: (B, 3, 224, 224), (B,)
  ↓
送入模型训练
```

### 3.5 完整代码模板

```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data_modules import MyData
from src.transforms.presets import train_transform, val_transform

# 创建 Dataset（使用预设 Transform）
train_dataset = MyData(root_dir="hymenoptera_data/train", label_dir="ants",
                       transform=train_transform)
val_dataset = MyData(root_dir="hymenoptera_data/val", label_dir="ants",
                     transform=val_transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        num_workers=0, drop_last=False)

# 迭代数据
for imgs, labels in train_loader:
    print(f"图片形状: {imgs.shape}")  # torch.Size([32, 3, 224, 224])
    print(f"标签: {labels}")
    break
```

---

## 四、CNN 结构与训练循环

### 4.1 Conv2d + ReLU + MaxPool2d 经典组合

为什么是这个组合：

- **Conv2d**：提取局部特征（边缘、纹理），kernel 滑动实现参数共享
- **ReLU**：引入非线性，没有它多层 Conv 等价于一层线性变换
- **MaxPool2d**：下采样，缩小空间尺寸，减少计算量，增大感受野

**shape 速推口诀**：

- `kernel_size=3, padding=1` → H/W 不变
- `kernel_size=5, padding=2` → H/W 不变
- `MaxPool2d(kernel_size=2)` → H/W 减半
- 通用公式：`H_out = (H_in + 2*padding - kernel_size) / stride + 1`

### 4.2 SimpleCNN 完整 shape 速查表

|层|操作|输出 shape|备注|
|---|---|---|---|
|输入|—|(B, 3, 32, 32)|CIFAR-10 彩色图|
|Conv1|Conv2d(3→32, k=5, p=2)|(B, 32, 32, 32)|padding=2 保持尺寸|
|ReLU1|ReLU|(B, 32, 32, 32)|shape 不变|
|Pool1|MaxPool2d(k=2)|(B, 32, 16, 16)|尺寸减半|
|Conv2|Conv2d(32→32, k=3, p=1)|(B, 32, 16, 16)|padding=1 保持尺寸|
|ReLU2|ReLU|(B, 32, 16, 16)|—|
|Pool2|MaxPool2d(k=2)|(B, 32, 8, 8)|尺寸减半|
|Conv3|Conv2d(32→64, k=3, p=1)|(B, 64, 8, 8)|通道 32→64|
|ReLU3|ReLU|(B, 64, 8, 8)|—|
|Pool3|MaxPool2d(k=2)|(B, 64, 4, 4)|尺寸减半|
|Flatten|—|(B, 1024)|64 × 4 × 4 = 1024|
|Linear1|Linear(1024→64)|(B, 64)|降维|
|Linear2|Linear(64→10)|(B, 10)|10 类输出|

### 4.3 训练循环 5 步

```python
for imgs, labels in train_loader:
    optimizer.zero_grad()              # 1. 清除旧梯度（不清除会累积）
    outputs = model(imgs)              # 2. 前向传播
    loss = loss_fn(outputs, labels)    # 3. 计算损失
    loss.backward()                    # 4. 反向传播（算梯度）
    optimizer.step()                   # 5. 更新参数
```

|步骤|函数|不做会怎样|
|---|---|---|
|zero_grad|清零 `.grad`|梯度在 batch 间累积，训练不稳定|
|forward|数据过网络|—|
|loss|算预测与标签差距|—|
|backward|反向传播填充 `.grad`|参数没梯度，step 无法更新|
|step|用梯度更新参数|算了梯度但不更新，白算|

**关键理解**：`backward()` 只算梯度，`step()` 才改参数，这是两个独立步骤。

### 4.4 eval 与 train 模式

```python
model.eval()                    # 切到评估模式
with torch.no_grad():           # 关闭梯度计算
    # ... 计算准确率
model.train()                   # 切回训练模式（容易忘！）
```

- `model.eval()` 影响 Dropout（停止丢弃）和 BatchNorm（用全局统计量）
- `torch.no_grad()` 不影响模型行为，只省显存、加速
- 两者作用不同，评估时通常都要用
- **eval 后忘记 `train()` 是常见 bug**：下一个 epoch 的 Dropout/BN 行为不正确

### 4.5 数据增强对比实验

```python
# CIFAR-10 数据增强
augmented_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])
```

为什么训练集增强、测试集不增强：训练集增强 = 扩大数据多样性提升泛化；测试集不增强 = 保持评估一致可复现。

---

## 五、迁移学习：ResNet18

### 5.1 核心概念

一句话：把在大数据集（ImageNet，120 万张图）上训练好的模型权重，搬到小数据集（蚂蚁/蜜蜂，244 张图）上用，只微调最后几层。

```python
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # 1000 类 → 2 类
```

为什么有效：预训练模型已学会"边缘、纹理、形状"等通用视觉特征。244 张图不够从零学这些特征（11M 参数），但可以复用预训练特征，只训练分类决策。

### 5.2 Freeze vs Fine-tune

|策略|含义|适用场景|
|---|---|---|
|冻结训练（Freeze）|冻结所有卷积层，只训练 fc|数据量极小、域相似|
|微调（Fine-tune）|解冻部分/全部层，用小学习率训练|数据量适中、需要适配|

渐进解冻策略（本项目采用）：

- 阶段 A：冻结全部卷积层，只训练 fc（1,026 个参数）
- 阶段 B：解冻 layer4 + fc（8,394,754 个参数），用更小的学习率

为什么不一上来全量训练：大学习率 + 全部参数 = 破坏预训练特征（灾难性遗忘）。

### 5.3 关键代码片段

#### 冻结参数

```python
def set_trainable_layers(model, stage):
    for param in model.parameters():
        param.requires_grad = False     # 先冻结所有

    if stage == "head":
        modules = [model.fc]
    elif stage == "finetune":
        modules = [model.layer4, model.fc]

    for module in modules:
        for param in module.parameters():
            param.requires_grad = True  # 再解冻需要的部分
```

#### 只为可训练参数创建 optimizer

```python
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=lr)
```

为什么不直接传 `model.parameters()`：Adam 会为冻结参数也维护动量状态，浪费内存。

#### 推理时的关键步骤

```python
image = Image.open(path).convert("RGB")
tensor = val_transform(image).unsqueeze(0).to(device)  # (3,224,224) → (1,3,224,224)

model.eval()
with torch.no_grad():
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    pred_class = idx_to_class[pred_idx]
```

- `unsqueeze(0)`：单张图没有 batch 维度，必须插入
- `val_transform`：推理必须用验证集的固定预处理
- `softmax`：把 logits 转换为概率分布

### 5.4 实验结果

#### 预训练 vs 非预训练（同一数据集 hymenoptera）

|对比项|预训练|非预训练|
|---|---|---|
|阶段 A 第 1 epoch val_acc|84.97%|45.75%（接近随机猜）|
|阶段 A 最终 best|92.16%|69.28%|
|阶段 B 最终 best|94.77%|69.93%|
|总训练 epoch|6|20|
|训练稳定性|单调上升|剧烈震荡|

结论：244 张图从零训练完全不够，20 个 epoch 还不如预训练版第 1 个 epoch。

#### 学习率对比

|组合|HEAD_LR|FINETUNE_LR|最终 best val_acc|
|---|---|---|---|
|组合 1|1e-3|1e-4|**94.77%**|
|组合 2|1e-3|1e-5|93.46%|
|组合 3|5e-4|1e-4|94.12%|

- 组合 2 的 FINETUNE_LR=1e-5 太小，阶段 B 学不动
- 组合 3 的 HEAD_LR=5e-4 起步慢
- 最优区间在 HEAD_LR=1e-3 + FINETUNE_LR=1e-4 附近

#### 阶段策略对比

|版本|训练范围|可训练参数|最终 best val_acc|
|---|---|---|---|
|A|只训 fc|1,026|92.16%|
|B|layer4 + fc|8,394,754|**94.77%**|
|C|全模型|11,177,538|未超越 B（过拟合）|

版本 C 的 train_loss 降到 0.05 但 val_acc 反降到 86%，典型过拟合。小数据集上渐进解冻比全量微调更稳定。

### 5.5 认知升级

Phase 1 的 SimpleCNN（~62K 参数）在 CIFAR-10 上跑到 ~65%。Phase 2 发现：244 张图从零训练 11M 参数的 ResNet18 连 50% 都达不到，但加载预训练权重后 6 个 epoch 就能到 94.77%。核心是预训练权重里沉淀了 120 万张图学到的通用特征。这就是工业界 90% 的 CV 项目选择迁移学习而不是从零训练的原因。

---

## 六、踩坑记录（全阶段汇总）

### 数据加载阶段

**路径找不到（FileNotFoundError）**：优先检查相对路径、反斜杠转义和 `os.path.exists()`。用正斜杠 `"data/train/ants"` 或原始字符串 `r"data\train\ants"`。

**形状不匹配（Expected 4-dimensional input）**：检查是否缺少 batch 维度，用 `unsqueeze(0)` 补。直接用 Dataset 取样本是 3 维 `(C,H,W)`，模型期望 4 维 `(B,C,H,W)`。

**标签类型错误**：`label` 必须转 `torch.tensor()`，PyTorch 的 loss 函数期望 Tensor 类型。

**Transform 顺序错**：`Normalize` 必须在 `ToTensor` 之后（Normalize 需要 Tensor 输入）。

**Windows 下 num_workers > 0 报错**：Windows 不支持 fork，解决方案：`num_workers=0`。

**导入路径报错**：永远从项目根目录启动脚本，不要直接运行 `src/data_modules/base.py`。

### CNN 训练阶段

**padding 和 kernel_size 的配合**：Conv2d 不设 padding 输出会缩小。口诀：`padding = (kernel_size - 1) / 2` 保持尺寸不变（stride=1 时）。

**Flatten 后 in_features 算错**：报 `RuntimeError: mat1 and mat2 shapes cannot be multiplied`。最靠谱的验证：

```python
x = torch.randn(1, 3, 32, 32)
out = model.feature(x)
print(out.shape)  # 直接看输出，不靠手算
```

**eval 后忘记切回 train**：`evaluate()` 函数最后一定要 `model.train()`。

**CrossEntropyLoss 不需要手动 Softmax**：内部已含 `log_softmax`，重复会导致 loss 错误。

**实验对比要控制变量**：baseline 没有 Normalize 而增强组有，提升可能部分来自标准化。

### 迁移学习阶段

**输入预处理必须和预训练一致**：去掉 ImageNet Normalize 后准确率明显下降。ResNet18 的卷积核是在标准化输入下优化的。

**非预训练模型在冻结阶段几乎学不动**：随机初始化的特征层没有区分能力，fc 层在对"噪声"做分类。

**全量微调反而不如渐进解冻**：244 张图训练 11M 参数严重过拟合。小数据集上参数越多，过拟合风险越大。

**每个阶段的 optimizer 是重新创建的**：新 optimizer 没有继承之前的动量状态，切换阶段时 val_acc 可能暴跌。

**学习率太小等于没在学**：FINETUNE_LR=1e-5 时 train_loss 几乎不动。学习率不是越小越好。

**checkpoint 要保存足够信息**：不仅保存 model_state_dict，还要保存 class_to_idx、epoch、stage、optimizer_state_dict。

**类别映射反转**：训练时 `class_to_idx = {'ants': 0}`，推理时需要反转为 `idx_to_class = {0: 'ants'}`。

**unsqueeze 不是取元素**：`unsqueeze(0)` 是增加维度，不是索引。推理单张图必须加。

---

## 七、面试高频考点（全阶段汇总）

### 数据加载相关

|#|问题|一句话答案|
|---|---|---|
|1|`__init__` 里为什么不直接读图片？|惰性加载：只建索引不读图，处理 TB 级数据不爆内存|
|2|为什么要 `.convert('RGB')`？|防止 PNG(4通道) 或灰度图(1通道) 导致维度错误|
|3|`ToTensor()` 做了什么？|[0,255]→[0,1] + (H,W,C)→(C,H,W)|
|4|为什么测试集不做随机增强？|评估需要可复现性，随机性导致指标不稳定|
|5|`drop_last` 什么时候用 True？|训练集用 True 防止最后一个小 batch 影响 BN 层|
|6|`batch_size` 怎么选？|从 32 开始，显存不够减半，有余翻倍|

### CNN 训练相关

|#|问题|一句话答案|
|---|---|---|
|7|为什么要 `zero_grad()`？|PyTorch 默认梯度累加，不清零会导致叠加|
|8|`eval()` 和 `no_grad()` 区别？|eval 改变模型行为（关 Dropout/BN），no_grad 省显存，两者独立|
|9|`CrossEntropyLoss` 要先 Softmax？|不要，内部已含 log_softmax，重复导致 loss 错误|
|10|`backward()` 和 `step()` 区别？|backward 算梯度填充 .grad，step 用梯度更新参数|
|11|过拟合怎么解决？|数据增强、Dropout、BatchNorm、迁移学习、正则化|

### 迁移学习相关

|#|问题|一句话答案|
|---|---|---|
|12|为什么 pretrained 在小数据集有效？|已学会通用视觉特征，小数据集只需微调分类决策|
|13|为什么冻结前面的层？|前层提取通用特征已经很好，大学习率会破坏它们（灾难性遗忘）|
|14|backbone 和 head 学习率怎么设？|head 用 1e-3（从头学），backbone 用 1e-4（微调）|
|15|迁移学习 vs 从头训练怎么选？|数据量大 + 域差异大 → 从头训；数据量小或域相似 → 迁移|
|16|部署到边缘设备怎么做？|torch.onnx.export → ONNX Runtime / TensorRT 推理|
|17|输入必须 224x224 + ImageNet Normalize？|权重在该分布下训练，改变输入分布导致特征提取失效|
|18|val_acc 不升但 train_loss 在降？|过拟合，应减少可训练参数、增加增强或提前停止|

---

## 八、调试技巧速查

### 反标准化（可视化调试 Transform）

```python
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

from torchvision.utils import save_image
img_denorm = denormalize(img)
save_image(img_denorm, 'debug_transform.png')
```

### 验证模型 shape

```python
x = torch.randn(1, 3, 32, 32)
out = model.feature(x)
print(out.shape)  # 直接看，不靠手算
```

### 常见报错速查

|报错|原因|修复|
|---|---|---|
|`Expected 4-dimensional input`|缺少 batch 维度|`x.unsqueeze(0)`|
|`mat1 and mat2 shapes cannot be multiplied`|Linear 的 in_features 算错|打印前一层输出 shape|
|`Expected Long but got Int`|label 没转 Tensor|`torch.tensor(label)`|
|`CUDA out of memory`|batch_size 太大|减半 batch_size|
|`FileNotFoundError`|路径错误|`os.path.exists()` 检查|
|`ModuleNotFoundError`|缺少 `__init__.py` 或启动目录不对|从项目根目录运行|

---

## 九、对应文件索引

### 数据加载阶段

|文件|作用|
|---|---|
|`src/data_modules/base.py`|MyData：PIL 分类数据集|
|`src/data_modules/classification.py`|ClassificationDataset：OpenCV 版|
|`src/data_modules/detection.py`|DetectionDataset：YOLO 格式|
|`src/transforms/presets.py`|train/val Transform 预设|
|`examples/demo_classification.py`|分类 Dataset 演示|
|`examples/demo_detection.py`|YOLO 检测数据演示|
|`examples/demo_transforms.py`|图像变换演示|
|`examples/demo_dataloader.py`|DataLoader 演示|
|`main.py`|Dataset + Transform + DataLoader 串联|

### CNN 训练阶段

|文件|作用|
|---|---|
|`examples/study_nn_layers.py`|nn.Module 最小模板|
|`examples/nn_conv2d.py`|Conv2d 输出观察|
|`examples/nn_maxpool2d.py`|MaxPool2d 输出观察|
|`examples/nn_relu.py`|ReLU 激活函数|
|`examples/nn_linear.py`|Linear + Flatten|
|`examples/nn_sequential.py`|Sequential 搭 CNN|
|`examples/nn_loss.py`|损失函数 + backward|
|`examples/nn_optimizer.py`|优化器 + 训练步骤|
|`src/models/simple_cnn.py`|SimpleCNN 模型定义|
|`train.py`|CIFAR-10 完整训练脚本|
|`examples/train_with_aug.py`|数据增强对比实验|

### 迁移学习阶段

|文件|作用|
|---|---|
|`examples/train_transfer.py`|ResNet18 两阶段迁移学习训练|
|`examples/predict_transfer.py`|单图/文件夹批量推理|
|`docs/resnet18_transfer_learning_plan.md`|迁移学习完整学习方案|

---

## 十、下一步计划

PyTorch 基础阶段全部完成，直接进入 **Week 3：YOLO 缺陷检测项目**。

这是简历上的核心项目，目标：NEU-DET 钢材缺陷检测 → mAP@0.5 > 0.70 → ONNX 导出 → GitHub 精品项目。