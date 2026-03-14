
# PyTorch 基础学习项目 — 完整笔记

> Phase 1 收尾文档：项目描述 + 学习笔记 + 进度总结 最后更新：2026年3月14日

---

## 一、项目简介

这是一个围绕图像任务逐步学习 PyTorch 的入门项目，从零开始掌握 Tensor、Dataset、Transform、DataLoader、常见神经网络层、CNN 结构，直到完整的训练评估流程。

项目采用"边学边拆分"的方式组织：

- `src/` 放可复用核心模块，面向后续正式训练
- `examples/` 放阶段性实验脚本，面向单个知识点验证
- `main.py` 把 Dataset、Transform、DataLoader 串起来
- `train.py` 完成 CIFAR-10 上的完整训练循环
- `src/models/simple_cnn.py` 定义 SimpleCNN 分类模型

项目同时服务两条学习线：

- 蚂蚁/蜜蜂分类与 YOLO 标注数据：理解真实项目的数据组织方式
- CIFAR-10：理解 DataLoader、Conv2d、MaxPool2d、训练循环等通用操作

---

## 二、环境与依赖

|项目|版本|
|---|---|
|GPU|NVIDIA GeForce RTX 3060|
|CUDA|11.8|
|PyTorch|2.0.0|
|torchvision|0.15.0|
|Python|3.9|
|其他依赖|OpenCV 4.8、NumPy、Matplotlib、TensorBoard、Jupyter|

```bash
conda env create -f environment.yml
conda activate pytorch_basics
```

---

## 三、学习进度

### 已完成

|阶段|状态|对应文件|
|---|---|---|
|PyTorch 环境验证|✅|环境准备与基础运行|
|Tensor 与前向传播|✅|`examples/test.py`|
|自定义 Dataset|✅|`src/data_modules/`、`examples/demo_classification.py`、`examples/demo_detection.py`|
|Transform 数据处理|✅|`src/transforms/`、`examples/demo_transforms.py`|
|DataLoader 使用|✅|`examples/demo_dataloader.py`、`main.py`|
|`nn.Module` 基础|✅|`examples/study_nn_layers.py`|
|`nn.Conv2d`|✅|`examples/nn_conv2d.py`|
|`nn.MaxPool2d`|✅|`examples/nn_maxpool2d.py`|
|ReLU 非线性激活|✅|`examples/nn_relu.py`|
|Linear + Flatten|✅|`examples/nn_linear.py`|
|Sequential|✅|`examples/nn_sequential.py`|
|损失函数 + 反向传播|✅|`examples/nn_loss.py`|
|优化器|✅|`examples/nn_optimizer.py`|
|SimpleCNN 模型定义|✅|`src/models/simple_cnn.py`|
|完整训练循环|✅|`train.py`|
|数据增强对比实验|✅|`examples/train_with_aug.py`|

### 当前所处阶段

PyTorch 基础训练链路已全部走通。完整链路：Dataset → Transform → DataLoader → Model → Loss → Backward → Optimizer → 训练循环 → 评估 → 数据增强。

### 待完成

|阶段|状态|文件|
|---|---|---|
|迁移学习（ResNet18）|待完成|`examples/train_transfer.py`|
|模型评估与推理|待完成|尚未创建独立脚本|
|→ 进入 Week 3 YOLO 项目|待启动|—|

---

## 四、项目结构

```text
Proj_Pytorch_Basics/
├── src/                          # 核心可复用模块
│   ├── __init__.py               # 懒加载入口
│   ├── data_modules/             # Dataset 定义
│   │   ├── __init__.py
│   │   ├── base.py               # MyData：PIL 分类数据集，支持 transform
│   │   ├── classification.py     # ClassificationDataset：OpenCV 分类数据集
│   │   └── detection.py          # DetectionDataset：YOLO 检测数据集
│   ├── transforms/
│   │   ├── __init__.py
│   │   └── presets.py            # train/val transform 与可视化工具
│   └── models/
│       ├── __init__.py           # 懒加载导出 SimpleCNN
│       └── simple_cnn.py         # SimpleCNN 模型定义
├── examples/                     # 知识点演示脚本
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
│   └── train_with_aug.py         # 数据增强对比实验
├── main.py                       # Dataset + Transform + DataLoader 串联
├── train.py                      # CIFAR-10 完整训练脚本
├── docs/                         # 文档
│   ├── phase1_notes.md           # Phase 1 核心学习笔记（精简版）
│   └── pytorch_basics_I.md       # Phase 1 完整笔记（进度 + 结构 + 学习总结）
├── data/                         # 蚂蚁/蜜蜂检测数据（含 YOLO 标注）
├── hymenoptera_data/             # ImageFolder 格式分类数据
├── logs/                         # TensorBoard 日志
├── environment.yml               # Conda 环境配置
├── CLAUDE.md                     # Claude Code 项目上下文
└── README.md
```

### 重要路径约定

|数据/目录|说明|路径|
|---|---|---|
|`hymenoptera_data/`|蚂蚁/蜜蜂分类数据集|项目内|
|`data/`|蚂蚁/蜜蜂检测数据与 YOLO 标注|项目内|
|CIFAR-10|训练用公共数据集|项目外 `../datasets`|
|`logs/`|TensorBoard 日志|项目内|

公开数据集不提交到仓库，放在仓库外部通过相对路径 `../datasets` 引用。

---

## 五、`src/` 与 `examples/` 的分工

**`src/`** 是"工具箱"——后续训练脚本会真正复用的模块：

|模块|文件|作用|
|---|---|---|
|`MyData`|`src/data_modules/base.py`|PIL 分类数据集，支持 transform，正式训练用|
|`ClassificationDataset`|`src/data_modules/classification.py`|OpenCV 简化版，学习 Dataset 原理用|
|`DetectionDataset`|`src/data_modules/detection.py`|YOLO 检测数据读取|
|`train/val_transform`|`src/transforms/presets.py`|训练/验证 Transform 管线 + load_image + plot_compare|
|`SimpleCNN`|`src/models/simple_cnn.py`|3 层 Conv + 2 层 Linear，适配 CIFAR-10|

**`examples/`** 是"练兵场"——每个脚本只解决一个知识点，可独立运行。

两者的关系：`examples/` 调用 `src/` 的模块来做实验，`src/` 沉淀的是可复用的积木。

---

## 六、学习笔记

### 1. 数据加载三件套：Dataset → Transform → DataLoader

一句话总结：Dataset 负责"怎么读一条数据"，Transform 负责"怎么处理这条数据"，DataLoader 负责"怎么批量送给模型"。

```python
dataset = CIFAR10(root, transform=transform)  # 读单条: (image, label)
transform = transforms.Compose([...])          # 处理: resize、flip、toTensor
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # 打包: (B, C, H, W)
```

关键点：

- `__getitem__` 返回单个样本，DataLoader 自动 stack 成 batch
- `ToTensor()` 做两件事：`[0,255]` 整数 → `[0,1]` 浮点，`(H,W,C)` → `(C,H,W)`
- 标签映射很重要：模型训练用数字标签，不直接用字符串
- `convert('RGB')` 必须有，避免灰度图或透明通道导致维度错误
- Windows 下 `num_workers=0`，`shuffle=True` 只用在训练集

### 2. CNN 结构：Conv2d + ReLU + MaxPool2d 经典组合

为什么是这个组合：

- `Conv2d`：提取局部特征（边缘、纹理），kernel 滑动实现参数共享
- `ReLU`：引入非线性，没有它多层 Conv 等价于一层线性变换
- `MaxPool2d`：下采样，缩小空间尺寸，减少计算量，增大感受野

shape 速推口诀：

- `kernel_size=3, padding=1` → H/W 不变
- `kernel_size=5, padding=2` → H/W 不变
- `MaxPool2d(kernel_size=2)` → H/W 减半
- 通用公式：`H_out = (H_in + 2*padding - kernel_size) / stride + 1`

### 3. 训练循环 5 步

```python
for imgs, labels in train_loader:
    optimizer.zero_grad()       # 1. 清除旧梯度（不清除会累积）
    outputs = model(imgs)       # 2. 前向传播
    loss = loss_fn(outputs, labels)  # 3. 计算损失
    loss.backward()             # 4. 反向传播（算梯度）
    optimizer.step()            # 5. 更新参数
```

|步骤|函数|不做会怎样|
|---|---|---|
|zero_grad|清零 `.grad`|梯度在 batch 间累积，训练不稳定|
|forward|数据过网络|—|
|loss|算预测与标签差距|—|
|backward|反向传播填充 `.grad`|参数没梯度，step 无法更新|
|step|用梯度更新参数|算了梯度但不更新，白算|

关键理解：`backward()` 只算梯度，`step()` 才改参数，这是两个独立步骤。

### 4. eval 与 train 模式

```python
model.eval()                    # 切到评估模式
with torch.no_grad():           # 关闭梯度计算
    # ... 计算准确率
model.train()                   # 切回训练模式（容易忘！）
```

- `model.eval()` 影响 Dropout（停止丢弃）和 BatchNorm（用全局统计量）
- `torch.no_grad()` 不影响模型行为，只省显存、加速
- 两者作用不同，评估时通常都要用
- eval 后忘记 `train()`：下一个 epoch 的 Dropout/BN 行为不正确

### 5. 数据增强

为什么训练集增强、测试集不增强：训练集增强 = 扩大数据多样性提升泛化；测试集不增强 = 保持评估一致可复现。

```python
augmented_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])
```

不同数据集的 Normalize 值不同：

|数据集|mean|std|
|---|---|---|
|CIFAR-10|[0.4914, 0.4822, 0.4465]|[0.2023, 0.1994, 0.2010]|
|ImageNet|[0.485, 0.456, 0.406]|[0.229, 0.224, 0.225]|

### 6. SimpleCNN 完整 shape 速查表

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

---

## 七、踩坑记录

**padding 和 kernel_size 的配合**：Conv2d 不设 padding 输出会缩小。`kernel_size=5` 不设 padding，32×32 直接变 28×28。口诀：`padding = (kernel_size - 1) / 2` 保持尺寸不变（stride=1 时）。

**Flatten 后 in_features 算错**：Linear 的 in_features 必须等于 feature extractor 最后输出的 C×H×W。算错会报 `RuntimeError: mat1 and mat2 shapes cannot be multiplied`。最靠谱的验证：

```python
x = torch.randn(1, 3, 32, 32)
out = model.feature(x)
print(out.shape)  # 直接看输出
```

**eval 后忘记切回 train**：`evaluate()` 函数最后一定要 `model.train()`。

**CrossEntropyLoss 不需要手动 Softmax**：`nn.CrossEntropyLoss` 内部已包含 `log_softmax`。模型最后再加 Softmax 等于做两次，loss 会不对。

**实验对比要控制变量**：baseline 没有 Normalize 而增强组有，提升可能部分来自标准化。更严谨的做法是只改"增强"一个变量。

**Windows 下 num_workers > 0 报错**：Windows 不支持 fork，解决方案：`num_workers=0`。

**FileNotFoundError**：优先检查相对路径、反斜杠转义和 `os.path.exists`。

**RuntimeError: Expected 4-dimensional input**：检查是否缺少 batch 维度，用 `unsqueeze(0)` 补。

**导入路径报错**：优先从项目根目录启动脚本。

---

## 八、面试高频考点

|#|问题|一句话答案|
|---|---|---|
|1|训练时为什么要 `zero_grad()`？|PyTorch 默认梯度累加，不清零会导致梯度是多个 batch 的叠加，训练不稳定|
|2|`model.eval()` 和 `torch.no_grad()` 有什么区别？|`eval()` 改变模型行为（关 Dropout/BN 随机性），`no_grad()` 关闭梯度计算图（省显存），两者独立且通常同时用|
|3|`CrossEntropyLoss` 输入需要先过 Softmax 吗？|不需要，内部已含 `log_softmax + nll_loss`，重复会导致 loss 错误|
|4|`backward()` 和 `step()` 分别做了什么？|`backward()` 反向传播算梯度填充 `.grad`，`step()` 用梯度和学习率更新参数值|
|5|为什么测试集不做数据增强？|测试集衡量真实性能，必须保持一致可复现；增强只用于扩大训练数据多样性|

---

## 九、工程结构上的收获

- 理解了 `src/`、`examples/`、`main.py` 的分工：积木 vs 实验 vs 组装
- 掌握了 `__init__.py` 的包作用和懒加载机制
- 数据集、日志、公开下载数据不应提交进仓库
- 项目里 3 个 Dataset 类体现了学习迭代：简化版理解原理 → 规范版用于训练 → 检测版理解标注格式

---

## 十、下一步计划

1. 迁移学习（ResNet18 pretrained → CIFAR-10），创建 `examples/train_transfer.py`
2. 完成后直接进入 **Week 3：YOLO 缺陷检测项目**（学习路线核心项目）
3. Week 2 的代码重构和 Git 规范在做 YOLO 的过程中同步完成