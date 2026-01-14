
# PyTorch 基础学习项目

## 简介
这是我的第一个深度学习项目，用于熟悉 PyTorch 的 Tensor 操作和训练流程。

## 环境配置

### 1. 根据配置文件创建环境
```bash
conda env create -f environment.yml
```

### 2. 激活环境
```bash
conda activate pytorch_basics
```

## 项目结构

```

Proj_Pytorch_Basics/

├── src/                      # 核心可复用代码

│   ├── data_modules/         # 数据集类定义

│   │   ├── __init__.py

│   │   ├── base.py           # MyData

│   │   ├── classification.py # ClassificationDataset

│   │   └── detection.py      # DetectionDataset

│   └── transforms/           # 数据变换

│       ├── __init__.py

│       └── presets.py        # 预定义 transforms

│

├── examples/                 # 演示/测试代码

│   ├── demo_classification.py

│   ├── demo_detection.py

│   ├── demo_transforms.py

│   └── demo_dataloader.py

│

├── main.py                   # 主入口

├── data/                     # 数据文件夹

└── logs/                     # 日志文件夹


```

## 学习目标

- 掌握 PyTorch Tensor 操作
    
- 理解训练循环
    
- 学会数据增强
    
- 完成第一个分类模型
    

## 硬件环境

- GPU: NVIDIA GeForce RTX 3060
    
- CUDA: 11.8
    
- PyTorch: 2.0.0
    

## 数据集

CIFAR-10 数据集位于: `../datasets/cifar-10-batches-py/`