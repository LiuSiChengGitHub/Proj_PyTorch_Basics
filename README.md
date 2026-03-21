# PyTorch 基础学习项目

## 项目简介

这是一个围绕图像任务逐步学习 PyTorch 的入门项目，目标是从零开始掌握 Tensor、Dataset、Transform、DataLoader、常见神经网络层、CNN 结构以及后续训练评估流程。

项目采用“边学边拆分”的方式组织：
- `src/` 放可复用核心模块，面向后续正式训练
- `examples/` 放阶段性实验脚本，面向单个知识点验证
- `main.py` 负责把 Dataset、Transform、DataLoader 串起来
- `train.py` 完整训练循环（CIFAR-10），`src/models/simple_cnn.py` 定义 CNN 模型

当前项目同时服务两条学习线：
- 蚂蚁 / 蜜蜂分类与 YOLO 标注数据：帮助理解真实项目的数据组织方式
- CIFAR-10：帮助理解 DataLoader、Conv2d、MaxPool2d 这类通用层操作

## 环境与依赖

| 项目 | 版本 |
|------|------|
| GPU | NVIDIA GeForce RTX 3060 |
| CUDA | 11.8 |
| PyTorch | 2.0.0 |
| torchvision | 0.15.0 |
| Python | 3.9 |
| 其他依赖 | OpenCV 4.8、NumPy、Matplotlib、TensorBoard、Jupyter |

```bash
conda env create -f environment.yml
conda activate pytorch_basics
```

## 项目目标

项目的总目标是完成一条完整的 PyTorch 学习闭环：

1. 理解 Tensor 和前向传播
2. 理解 Dataset / DataLoader 的职责分工
3. 学会图像 Transform 与训练集、验证集的不同策略
4. 理解 `nn.Module`、`Conv2d`、`MaxPool2d`、激活函数等常见层
5. 搭建一个简单 CNN 分类模型
6. 完成训练、验证、保存模型
7. 进行评估与推理

## 当前学习进度

### 已完成

| 阶段 | 状态 | 对应内容 |
|------|------|----------|
| PyTorch 环境验证 | 已完成 | 环境准备与基础运行 |
| Tensor 与前向传播 | 已完成 | `examples/test.py` |
| 自定义 Dataset | 已完成 | `src/data_modules/`、`examples/demo_classification.py`、`examples/demo_detection.py` |
| Transform 数据处理 | 已完成 | `src/transforms/`、`examples/demo_transforms.py` |
| DataLoader 使用 | 已完成 | `examples/demo_dataloader.py`、`main.py` |
| `nn.Module` 基础 | 已完成 | `examples/study_nn_layers.py` |
| `nn.Conv2d` | 已完成 | `examples/nn_conv2d.py` |
| `nn.MaxPool2d` | 已完成 | `examples/nn_maxpool2d.py` |
| 非线性激活（ReLU / Sigmoid） | 已完成 | `examples/nn_relu.py` |
| `nn.Linear` 全连接层 | 已完成 | `examples/nn_linear.py` |
| `nn.Sequential` | 已完成 | `examples/nn_sequential.py` |
| 损失函数与反向传播 | 已完成 | `examples/nn_loss.py` |
| 优化器（SGD / Adam） | 已完成 | `examples/nn_optimizer.py` |
| SimpleCNN 模型定义 | 已完成 | `src/models/simple_cnn.py` |
| 完整训练循环 | 已完成 | `train.py` |
| 数据增强对比实验 | 已完成 | `examples/train_with_aug.py` |

| 迁移学习（ResNet18） | 已完成 | `examples/train_transfer.py` |
| 单图/批量推理 | 已完成 | `examples/predict_transfer.py` |

### 当前所处阶段

PyTorch 基础阶段全部完成（Dataset → DataLoader → CNN → 训练循环 → 数据增强 → 迁移学习）。

完整链路：
- 数据加载三件套（Dataset → Transform → DataLoader）
- 神经网络核心组件（Conv2d、MaxPool2d、ReLU、Linear、Sequential）
- 训练基础三件套（损失函数、反向传播、优化器）
- SimpleCNN 模型定义 + 完整训练循环 + 数据增强实验
- ResNet18 迁移学习（渐进解冻两阶段训练 + 单图推理）

### 待完成

| 阶段 | 状态 | 说明 |
|------|------|------|
| YOLO 缺陷检测项目 | 待启动 | NEU-DET 钢材缺陷检测，独立项目 |

## 项目结构

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
│       ├── __init__.py
│       └── simple_cnn.py         # SimpleCNN：3层Conv+Pool + 2层Linear，适配 CIFAR-10
├── examples/                     # 知识点演示脚本
│   ├── test.py
│   ├── demo_classification.py
│   ├── demo_detection.py
│   ├── demo_dataloader.py
│   ├── demo_transforms.py
│   ├── study_nn_layers.py
│   ├── nn_conv2d.py
│   ├── nn_maxpool2d.py
│   ├── nn_relu.py
│   ├── nn_linear.py
│   ├── nn_sequential.py
│   ├── nn_loss.py
│   ├── nn_optimizer.py
│   ├── train_with_aug.py         # 数据增强对比实验（CIFAR-10）
│   ├── train_transfer.py         # ResNet18 迁移学习训练
│   └── predict_transfer.py       # 单图/文件夹推理
├── main.py                       # Dataset + Transform + DataLoader 串联示例
├── train.py                      # CIFAR-10 完整训练脚本（SimpleCNN）
├── saved_models/                 # 训练保存的模型（.gitignore）
├── docs/                         # 学习笔记
│   ├── pytorch_basics_I-0314.md  # Phase 1 完整笔记（进度 + 结构 + 学习总结）
│   ├── pytorch_basics_II_transfer_learning.md  # Phase 2 迁移学习笔记
│   ├── resnet18_transfer_learning_plan.md      # 迁移学习学习方案
│   └── yolo_defect_detection_plan.md           # YOLO 项目启动方案
├── data/                         # 自定义蚂蚁/蜜蜂数据（含 YOLO 标注）
├── hymenoptera_data/             # ImageFolder 格式分类数据
├── logs/                         # TensorBoard 日志
├── environment.yml               # Conda 环境配置
└── README.md
```

## 重要数据与路径

| 数据 / 目录 | 说明 | 当前路径约定 |
|------------|------|-------------|
| `hymenoptera_data/` | 蚂蚁 / 蜜蜂分类数据集 | 项目内 |
| `data/` | 蚂蚁 / 蜜蜂检测数据与 YOLO 标注 | 项目内 |
| CIFAR-10 | 用于 DataLoader、Conv2d、MaxPool2d 学习 | 项目外同级目录 `../datasets` |
| `logs/` | TensorBoard 日志输出目录 | 项目内 |

说明：
- 当前项目里很多 CIFAR-10 示例使用相对路径 `../datasets`
- 对应实际位置是项目同级目录 `D:\Base\CodingSpace\datasets`
- 公开数据集不建议直接提交到仓库中，适合放在仓库外部并通过相对路径引用

## `src` 和 `examples` 的联系与区别

### `src` 的角色：可复用核心逻辑

`src/` 更像“工具箱”或“正式工程代码”。

这里放的是后续训练脚本会真正复用的模块：
- `src/data_modules/base.py` 中的 `MyData`：正式训练更适合使用的分类 Dataset
- `src/data_modules/classification.py`：用于理解 Dataset 基本结构的简化版本
- `src/data_modules/detection.py`：用于理解检测任务的数据读取方式
- `src/transforms/presets.py`：封装训练集 / 验证集的 Transform 管线
- `src/__init__.py` 与 `src/data_modules/__init__.py`：使用懒加载，避免无关依赖被提前导入

### `examples` 的角色：阶段性练习与验证

`examples/` 更像“练兵场”或“单知识点实验脚本”。

它们的特点是：
- 每个脚本只解决一个学习问题
- 可以单独运行，便于观察现象
- 不追求完整训练流程，而是强调概念验证

### 二者之间的关系

两者不是重复，而是分工不同：
- `examples/demo_classification.py` 调用 `src.data_modules.ClassificationDataset`
- `examples/demo_detection.py` 调用 `src.data_modules.DetectionDataset`
- `examples/demo_transforms.py` 调用 `src.transforms` 里的工具函数和预设管线
- `main.py` 调用 `src.data_modules.MyData`，把核心模块串起来
- `examples/nn_conv2d.py` 与 `examples/nn_maxpool2d.py` 当前更偏“层级实验”，主要围绕 CIFAR-10 观察输入输出变化，还没有沉淀到 `src/models/` 中

可以把它理解为：
- `src/` 是以后要复用的积木
- `examples/` 是你学习这些积木时做的实验记录

## 核心模块说明

### `src/data_modules/`

项目里一共有 3 个数据集类，对应不同学习阶段和任务：

| 类名 | 文件 | 读图方式 | 返回值 | 作用 |
|------|------|---------|--------|------|
| `MyData` | `src/data_modules/base.py` | PIL | `(image_tensor, label_tensor)` | 正式分类训练版本 |
| `ClassificationDataset` | `src/data_modules/classification.py` | OpenCV | `(numpy_image, label_str)` | 学习 Dataset 基本原理 |
| `DetectionDataset` | `src/data_modules/detection.py` | OpenCV | `(numpy_image, yolo_label_text)` | 学习检测数据格式 |

这里体现了明显的学习迭代：
- 先写一个简单的分类数据集理解 `__init__`、`__getitem__`、`__len__`
- 再写一个更规范的 `MyData` 版本，用于后续 transform 和训练
- 再拓展到检测任务，理解“图片 + 标注文件分离”的数据组织方式

### `src/transforms/`

`src/transforms/presets.py` 提供了：
- `train_transform`：训练用增强流程
- `val_transform`：验证用固定流程
- `load_image()`：安全读取图片
- `plot_compare()`：原图与变换结果对比

这里沉淀出的核心认识是：
- 训练集需要随机增强，提升泛化能力
- 验证集需要固定处理，保证评估公平
- `ToTensor()` 会把维度从 `(H, W, C)` 转成 `(C, H, W)`

### `src/models/`

`src/models/simple_cnn.py` 实现了 `SimpleCNN` 分类模型：
- 特征提取：3 组 Conv2d → ReLU → MaxPool2d，通道 3→32→32→64，空间 32→16→8→4
- 分类器：Flatten → Linear(1024, 64) → Linear(64, 10)
- 适配 CIFAR-10：输入 (B, 3, 32, 32) → 输出 (B, 10)
- 使用 `__init__.py` 懒加载，`from src.models import SimpleCNN` 即可使用

## `examples` 学习脚本说明

| 脚本 | 作用 | 当前定位 |
|------|------|----------|
| `examples/test.py` | 用 NumPy 手写前向传播 | Tensor 与网络计算起点 |
| `examples/demo_classification.py` | 演示分类 Dataset 的读取 | 自定义 Dataset 入门 |
| `examples/demo_detection.py` | 演示 YOLO 检测数据读取 | 检测数据格式入门 |
| `examples/demo_transforms.py` | 演示图像变换与可视化 | Transform 入门 |
| `examples/demo_dataloader.py` | 演示批量加载、shuffle、drop_last | DataLoader 入门 |
| `examples/study_nn_layers.py` | 最小 `nn.Module` 模板 | 神经网络层入门 |
| `examples/nn_conv2d.py` | 观察卷积层输出尺寸与特征图 | Conv2d 入门 |
| `examples/nn_maxpool2d.py` | 观察池化层输出尺寸与局部最大值 | MaxPool2d 入门 |
| `examples/nn_relu.py` | 演示 ReLU / Sigmoid 对张量的变换效果 | 非线性激活入门 |
| `examples/nn_linear.py` | 演示全连接层输入输出与 Flatten 配合 | Linear 全连接层入门 |
| `examples/nn_sequential.py` | 用 Sequential 搭建完整 CNN，验证 shape | Sequential 容器入门 |
| `examples/nn_loss.py` | 演示 CrossEntropyLoss、MSELoss 和 backward | 损失函数与反向传播入门 |
| `examples/nn_optimizer.py` | 演示 SGD / Adam 完整训练三步 | 优化器入门 |
| `examples/train_with_aug.py` | 基线 vs 增强 transform 的训练对比实验 | 数据增强实战 |
| `examples/train_transfer.py` | ResNet18 两阶段迁移学习训练 | 迁移学习训练 |
| `examples/predict_transfer.py` | 单图/文件夹批量推理（top-k + 置信度） | 迁移学习推理 |

建议学习顺序：

```text
test.py
-> demo_classification.py / demo_detection.py
-> demo_transforms.py
-> demo_dataloader.py
-> study_nn_layers.py
-> nn_conv2d.py
-> nn_maxpool2d.py
-> nn_relu.py
-> nn_linear.py
-> nn_sequential.py
-> nn_loss.py
-> nn_optimizer.py
-> simple_cnn.py
-> train.py
-> train_with_aug.py
-> train_transfer.py
-> predict_transfer.py
```

## 当前阶段的学习笔记提炼

### 1. 工程结构上的收获

- 已经理解为什么要把项目拆成 `src`、`examples`、`main.py`
- 已经掌握 `__init__.py` 的包作用，以及懒加载的基本思路
- 已经意识到数据集、日志、公开下载数据不应随意提交进仓库

### 2. Dataset 上的收获

- 掌握了 `__init__`、`__getitem__`、`__len__` 的职责划分
- 理解了标签映射的重要性：模型训练使用数字标签，不直接使用字符串
- 理解了 `convert('RGB')` 的必要性，避免灰度图或透明通道带来维度错误

### 3. Transform 上的收获

- 理解 `Compose` 按顺序执行
- 理解训练集与验证集的 transform 不能完全相同
- 理解 `ToTensor()` 会改变数值范围和维度顺序

### 4. DataLoader 上的收获

- 理解 Dataset 输出单样本，DataLoader 输出 batch
- 理解批处理后形状会变成 `(B, C, H, W)`
- 理解 Windows 环境下常用 `num_workers=0`
- 理解 `shuffle=True` 主要用于训练集
- 理解 `drop_last` 的典型使用场景

### 5. 神经网络层上的收获

- 已经建立 `nn.Module` 的最小模板概念：继承、定义层、实现 `forward`
- 已经观察过 `Conv2d` 会改变空间尺寸，也可以改变通道数
- 已经观察过 `MaxPool2d` 主要作用在空间维度，通常不改变通道数
- 理解 `ReLU(x) = max(0, x)`，负值截断，正值保持；Sigmoid 把输入压缩到 (0, 1)
- 理解 `nn.Linear(in, out)` 等价于 `y = xW^T + b`，权重形状为 (out, in)
- 理解 `nn.Flatten` 把 (B, C, H, W) 展平为 (B, C*H*W)，方便接全连接层
- 理解 `nn.Sequential` 可以把多个层按顺序组合，适合串行结构

### 6. 损失函数与训练流程上的收获

- 理解 `CrossEntropyLoss` 内部已包含 Softmax，不需要手动再加
- 理解 `MSELoss` 计算预测值与目标值之差的均方误差
- 理解 `loss.backward()` 后，每个参数的 `.grad` 属性被填充为梯度
- 掌握完整训练三步：`optimizer.zero_grad()` -> `loss.backward()` -> `optimizer.step()`
- 理解 `zero_grad()` 的必要性：防止梯度在多次 backward 之间累积

### 7. 完整训练循环上的收获

- 掌握训练 5 步口诀：zero_grad → forward → loss → backward → step
- 理解 device 管理：model、images、labels 都需要 `.to(device)`
- 理解 `model.eval()` 与 `torch.no_grad()` 的区别和各自作用
- 理解 eval 后必须切回 `model.train()`
- 掌握模型保存：`torch.save(model.state_dict(), path)`

### 8. 数据增强上的收获

- 理解训练集增强的目的是扩大数据多样性，提升泛化能力
- 理解测试集不能做随机增强，否则评估结果不可复现
- 掌握 CIFAR-10 经典增强组合：RandomCrop(32, padding=4) + RandomHorizontalFlip
- 理解不同数据集有不同的 Normalize 统计值（CIFAR-10 vs ImageNet）
- 理解实验对比需要控制变量，只改变待测试的因素

## 已掌握的避坑与调试经验

你目前已经积累了比较实用的一批调试经验：

- `FileNotFoundError`：优先检查相对路径、反斜杠转义和 `os.path.exists`
- `RuntimeError: Expected 4-dimensional input`：先检查是否缺少 batch 维度，必要时用 `unsqueeze(0)`
- `AttributeError: 'NoneType' object is not callable`：检查 Dataset 初始化时是否漏传 `transform`
- 导入路径报错：优先从项目根目录启动脚本
- Windows 多进程数据加载异常：先把 `num_workers` 设为 0

## 下一步建议

### 当前阶段小结

PyTorch 基础阶段已全部完成：
- 数据加载三件套（Dataset → Transform → DataLoader）
- 神经网络核心组件（Conv2d、MaxPool2d、ReLU、Linear、Sequential）
- 训练基础三件套（损失函数、反向传播、优化器）
- SimpleCNN 模型定义 + 完整训练循环 + 数据增强实验
- 详细学习笔记见 `docs/pytorch_basics_I.md`（完整版）和 `docs/phase1_notes.md`（精简版）

### 进入下一阶段前，最好确认自己能做到

1. 能独立搭出一个 CNN 模型并手推 shape 变化链
2. 能解释训练 5 步（zero_grad / forward / loss / backward / step）每一步的作用
3. 能解释 `model.eval()` 和 `torch.no_grad()` 的区别
4. 能解释为什么 CrossEntropyLoss 不需要提前手动 softmax
5. 能设计合理的数据增强策略并解释为什么测试集不增强

### 下一步

直接进入 **YOLO 缺陷检测项目**（学习路线核心实战项目）：
- 数据集：NEU-DET 钢材表面缺陷（6 类缺陷，约 1800 张）
- 模型：YOLOv8n
- 目标：mAP@0.5 > 0.70，ONNX 导出 + 推理验证
- 详细方案：`docs/yolo_defect_detection_plan.md`

## 总结

这个项目已经完成了从”零散学习”到”完整训练闭环”的跨越：
- Phase 1 覆盖了从 Tensor 基础到完整训练循环的全部核心知识
- Phase 2 完成了 ResNet18 迁移学习（渐进解冻 + 推理 + 对比实验）
- 代码组织清晰：`src/` 存放可复用模块，`examples/` 存放学习实验
- 学习笔记：`docs/pytorch_basics_I-0314.md`（Phase 1）、`docs/pytorch_basics_II_transfer_learning.md`（Phase 2）

