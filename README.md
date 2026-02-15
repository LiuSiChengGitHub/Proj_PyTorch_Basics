
# PyTorch 基础学习项目

## 简介

这是一个系统性的 PyTorch 深度学习入门项目，目标是从零开始掌握 PyTorch 核心概念，最终完成一个完整的图像分类模型训练流程。项目采用模块化设计，将可复用代码（数据集、变换、模型）封装在 `src/` 中，学习过程中的演示脚本放在 `examples/` 中。

## 硬件与环境

| 项目 | 版本 |
|------|------|
| GPU | NVIDIA GeForce RTX 3060 |
| CUDA | 11.8 |
| PyTorch | 2.0.0 + torchvision 0.15.0 |
| Python | 3.9 |
| 其他依赖 | OpenCV 4.8, NumPy, Matplotlib, TensorBoard 2.13, Jupyter |

```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate pytorch_basics
```

## 学习目标与完成进度

| # | 目标 | 状态 | 对应文件 |
|---|------|------|----------|
| 1 | 掌握 PyTorch Tensor 操作与神经网络前向传播 | 已完成 | `examples/test.py` |
| 2 | 理解 Dataset 和 DataLoader 机制 | 已完成 | `src/data_modules/`, `examples/demo_classification.py`, `examples/demo_dataloader.py` |
| 3 | 学会数据增强（transforms） | 已完成 | `src/transforms/`, `examples/demo_transforms.py` |
| 4 | 理解 nn.Module 与卷积层 | 已完成 | `examples/nn_conv2d.py`, `examples/study_nn_layers.py` |
| 5 | 理解目标检测数据格式（YOLO） | 已完成 | `src/data_modules/detection.py`, `examples/demo_detection.py` |
| 6 | 定义 CNN 分类模型 | **待完成** | `src/models/simple_cnn.py`（空文件） |
| 7 | 完成完整训练循环（训练+验证+保存模型） | **待完成** | `train.py`（空文件） |
| 8 | 模型评估与推理 | **待完成** | — |

## 项目结构

```
Proj_Pytorch_Basics/
│
├── src/                            # 核心可复用模块
│   ├── __init__.py                 # 懒加载入口
│   │
│   ├── data_modules/               # 数据集类定义
│   │   ├── __init__.py
│   │   ├── base.py                 # MyData — 基于 PIL 的分类数据集，支持 transform
│   │   ├── classification.py       # ClassificationDataset — 基于 OpenCV 的分类数据集
│   │   └── detection.py            # DetectionDataset — YOLO 格式目标检测数据集
│   │
│   ├── transforms/                 # 数据变换
│   │   ├── __init__.py
│   │   └── presets.py              # 预定义 train/val transform 管线及可视化工具
│   │
│   └── models/                     # 模型定义（待实现）
│       ├── __init__.py
│       └── simple_cnn.py           # CNN 分类网络（待实现）
│
├── examples/                       # 学习演示脚本
│   ├── test.py                     # 基础神经网络前向传播（NumPy 手写 + PyTorch）
│   ├── demo_classification.py      # ClassificationDataset 用法演示
│   ├── demo_detection.py           # DetectionDataset 用法演示
│   ├── demo_dataloader.py          # DataLoader + TensorBoard 完整流程演示
│   ├── demo_transforms.py          # 各类 transform 效果可视化
│   ├── nn_conv2d.py                # Conv2d 卷积层 + CIFAR-10 特征图可视化
│   └── study_nn_layers.py          # 最简 nn.Module 子类测试
│
├── main.py                         # 主入口：加载 hymenoptera 数据集 + DataLoader 示例
├── train.py                        # 训练脚本入口（待实现）
│
├── data/                           # 自定义蚂蚁/蜜蜂数据集
│   ├── train/
│   │   ├── ants_image/             # 蚂蚁训练图片
│   │   ├── ants_label/             # 蚂蚁 YOLO 格式标注
│   │   ├── bees_image/             # 蜜蜂训练图片
│   │   └── bees_label/             # 蜜蜂 YOLO 格式标注
│   └── val/
│       ├── ants/                   # 蚂蚁验证图片
│       └── bees/                   # 蜜蜂验证图片
│
├── hymenoptera_data/               # 另一份蚂蚁/蜜蜂分类数据集（ImageFolder 格式）
│   ├── train/
│   └── val/
│
├── datasets/                       # 下载的公开数据集
│   └── cifar-10-python.tar.gz      # CIFAR-10（~90MB）
│
├── environment.yml                 # Conda 环境配置
└── .gitignore
```

## 各模块功能说明

### `src/data_modules/` — 数据集

- **`base.py` (MyData)**：基于 PIL 的图像分类 Dataset。通过目录名自动映射标签（`"ants"→0, "bees"→1`），支持传入 transform。供 `main.py` 使用。
- **`classification.py` (ClassificationDataset)**：基于 OpenCV 读取图片的分类 Dataset，返回 `(numpy array, label_str)`。
- **`detection.py` (DetectionDataset)**：目标检测 Dataset，图片目录和标注目录分离，读取 YOLO 格式 `.txt` 标注文件，能处理缺失标注。

### `src/transforms/` — 数据变换

- **`presets.py`**：
  - `train_transform`：训练用增强管线 — Resize(256) → RandomCrop(224) → RandomHorizontalFlip → RandomRotation(15) → ColorJitter → ToTensor → Normalize（ImageNet 均值/标准差）
  - `val_transform`：验证用管线 — Resize(256) → CenterCrop(224) → ToTensor → Normalize
  - `load_image()`：安全图片加载（带校验）
  - `plot_compare()`：原图 vs 变换后对比可视化

### `src/models/` — 模型（待实现）

- **`simple_cnn.py`**：计划实现一个简单的 CNN 分类网络。

### `examples/` — 学习演示

| 脚本 | 学习内容 | 用到的工具 |
|------|---------|-----------|
| `test.py` | 基本前向传播、sigmoid、3 层网络 | NumPy, PyTorch |
| `demo_classification.py` | 分类数据集加载与可视化 | ClassificationDataset, Matplotlib |
| `demo_detection.py` | 检测数据集加载与可视化 | DetectionDataset, Matplotlib |
| `demo_dataloader.py` | DataLoader 参数（batch_size, shuffle, drop_last）、2 epoch 模拟训练 | CIFAR-10, DataLoader, TensorBoard |
| `demo_transforms.py` | 各种 transform 效果对比 | transforms, TensorBoard |
| `nn_conv2d.py` | Conv2d 卷积层原理、特征图可视化 | CIFAR-10, nn.Conv2d, TensorBoard |
| `study_nn_layers.py` | 最简 nn.Module 子类写法 | nn.Module |

## 工作流

项目按以下学习路径递进推进：

```
1. 基础概念          test.py
   NumPy 手写前向传播、理解 Tensor 运算
         │
         ▼
2. 数据加载          src/data_modules/ + demo_classification.py + demo_detection.py
   自定义 Dataset、__getitem__/__len__、分类与检测两种数据格式
         │
         ▼
3. 数据增强          src/transforms/ + demo_transforms.py
   transform 管线、训练/验证不同策略、TensorBoard 可视化
         │
         ▼
4. DataLoader        demo_dataloader.py + main.py
   批量加载、shuffle、多进程、与 TensorBoard 结合
         │
         ▼
5. 神经网络层        nn_conv2d.py + study_nn_layers.py
   nn.Module 子类化、Conv2d 参数与特征图
         │
         ▼
6. 模型定义          src/models/simple_cnn.py  ← 【下一步】
   搭建完整 CNN 分类网络
         │
         ▼
7. 训练循环          train.py  ← 【待实现】
   loss 函数、优化器、训练/验证循环、模型保存与加载
         │
         ▼
8. 评估与推理        ← 【待实现】
   准确率、混淆矩阵、单张图片推理
```

## 数据集

| 数据集 | 格式 | 用途 | 位置 |
|--------|------|------|------|
| Hymenoptera（蚂蚁/蜜蜂） | ImageFolder | 分类训练主数据集 | `hymenoptera_data/`, `data/` |
| CIFAR-10 | torchvision 内置 | DataLoader 和 Conv2d 演示 | `datasets/` |

## 接下来要做的事

1. **定义 CNN 模型** — 在 `src/models/simple_cnn.py` 中实现一个简单的卷积神经网络（如 Conv→ReLU→Pool→FC 结构），用于蚂蚁/蜜蜂二分类或 CIFAR-10 十分类。
2. **实现训练脚本** — 在 `train.py` 中编写完整的训练循环：损失函数（CrossEntropyLoss）、优化器（SGD/Adam）、逐 epoch 训练与验证、TensorBoard 记录 loss/accuracy 曲线、模型权重保存。
3. **模型评估与推理** — 加载保存的模型权重，计算测试集准确率，可视化预测结果。
