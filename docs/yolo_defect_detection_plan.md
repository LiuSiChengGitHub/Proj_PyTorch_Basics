# YOLO 钢材缺陷检测项目 — 启动方案

> 创建日期：2026年3月21日

---

## 一、背景说明

这是简历上的核心项目，目标岗位是外企 CV 方向（博世、西门子、Cognex 等）。面试时这个项目承载两个任务：

1. 证明你有完整的"数据 → 训练 → 评估 → 部署"项目经验
2. 证明你能用业界主流工具（YOLO + ONNX）解决工业场景的实际问题

时间紧迫，追求快速出成果，不追求理论完美。先跑通 baseline，再逐步打磨。

---

## 二、项目目标

| 项目 | 内容 |
|------|------|
| 数据集 | NEU-DET 钢材表面缺陷数据集 |
| 缺陷类别 | 6 类：Crazing、Inclusion、Patches、Pitted Surface、Rolled-in Scale、Scratches |
| 样本量 | 约 1800 张（每类约 300 张） |
| 模型 | YOLOv8n（nano 版本，轻量、边缘部署友好） |
| 训练指标 | mAP@0.5 > 0.70 |
| 部署验证 | ONNX 导出 + Python ONNX Runtime 推理，精度对齐 |
| 最终交付 | GitHub 精品项目（README + Demo + 清晰项目结构） |

---

## 三、分步计划

### Step 1：数据准备（预计 2-3 小时）

任务：

1. 下载 NEU-DET 数据集
2. 分析数据分布：每类样本数、图片尺寸、标注格式
3. 转换为 YOLO 格式（如果原始格式不是 YOLO）
4. 划分训练集/验证集/测试集（推荐 7:2:1）
5. 创建 `data.yaml` 配置文件

关键产出：

- `data/` 目录结构符合 YOLO 标准
- `data.yaml` 指明 train/val/test 路径和类别名
- 数据分布分析笔记

### Step 2：基线训练（预计 1-2 小时）

任务：

1. 安装 ultralytics：`pip install ultralytics`
2. 用 YOLOv8n 默认参数跑通训练

```bash
yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

3. 记录 baseline mAP@0.5
4. 查看训练曲线和预测样例

关键产出：

- 训练能跑通，loss 下降
- baseline mAP@0.5 数值（预期 0.50-0.65）

### Step 3：调参优化（预计 3-4 小时）

对比实验：

| 实验 | 变量 | 说明 |
|------|------|------|
| 实验 A | imgsz | 320 vs 640 vs 800，观察精度和速度 |
| 实验 B | 学习率 | lr0=0.01 vs 0.001，warmup 设置 |
| 实验 C | 增强策略 | 默认增强 vs 调整 mosaic/mixup |
| 实验 D | 模型大小 | YOLOv8n vs YOLOv8s（如果 n 达不到目标） |

每组实验只改一个变量，记录 mAP@0.5 对比。

关键产出：

- 找到使 mAP@0.5 > 0.70 的配置
- 实验对比记录

### Step 4：结果分析（预计 2-3 小时）

任务：

1. 绘制 PR 曲线（ultralytics 自动生成）
2. 分析混淆矩阵：哪些缺陷容易互相误判
3. 找出典型误检案例，分析原因
4. 按类别分析 AP：哪类最难检测，为什么

关键产出：

- PR 曲线图
- 混淆矩阵截图 + 分析
- 误检案例分析（3-5 个典型 case）

### Step 5：ONNX 导出 + 推理验证（预计 2-3 小时）

任务：

1. 导出 ONNX 模型

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

2. 编写 Python ONNX Runtime 推理脚本
3. 对比 PyTorch 推理和 ONNX 推理的精度是否对齐
4. 记录推理速度对比

关键产出：

- `best.onnx` 模型文件
- `inference_onnx.py` 推理脚本
- 精度对齐报告

### Step 6：GitHub 美化（预计 2-3 小时）

任务：

1. 编写 README.md（中英文双版本或英文版本）
2. 录制 Demo GIF（推理效果展示）
3. 整理项目结构，确保干净专业
4. 添加 requirements.txt
5. 添加 LICENSE

关键产出：

- 一个面试官打开就知道你做了什么的 GitHub 页面

---

## 四、推荐项目结构

这是一个独立新项目，不在当前 Proj_Pytorch_Basics 里。

```text
YOLO_Defect_Detection/
├── README.md                     # 项目说明（英文为主）
├── requirements.txt              # 依赖列表
├── LICENSE
├── data/
│   ├── data.yaml                 # YOLO 数据集配置
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── scripts/
│   ├── prepare_data.py           # 数据格式转换 + 划分
│   ├── train.py                  # 训练入口（封装 YOLO CLI）
│   ├── evaluate.py               # 评估 + 可视化
│   └── inference_onnx.py         # ONNX Runtime 推理
├── configs/
│   └── train_config.yaml         # 超参数配置
├── runs/                         # 训练输出（.gitignore）
├── models/                       # 导出的 ONNX 模型
│   └── best.onnx
├── docs/
│   ├── experiment_log.md         # 实验记录
│   └── assets/                   # README 用图（Demo GIF、PR 曲线等）
└── .gitignore
```

---

## 五、技术栈

| 工具 | 用途 |
|------|------|
| ultralytics (YOLOv8) | 训练 + 评估 + 导出 |
| ONNX | 模型格式转换 |
| onnxruntime | 推理部署 |
| OpenCV | 图像读取和可视化 |
| matplotlib | PR 曲线、混淆矩阵绘制 |

安装命令：

```bash
pip install ultralytics onnx onnxruntime opencv-python matplotlib
```

---

## 六、简历话术模板

### 中文版

**YOLO 钢材表面缺陷检测系统**

- 基于 YOLOv8n 构建钢材表面缺陷检测系统，覆盖 6 类常见缺陷（裂纹、夹杂、斑块等），在 NEU-DET 数据集上实现 mAP@0.5 > 0.70
- 完成数据预处理、模型训练调优、PR 曲线分析、混淆矩阵误检分析等全流程
- 将训练模型导出为 ONNX 格式，使用 ONNX Runtime 完成推理部署验证，确保精度对齐
- 技术栈：Python / PyTorch / YOLOv8 / ONNX / OpenCV

### English Version

**Steel Surface Defect Detection with YOLOv8**

- Built a YOLOv8n-based defect detection system for 6 steel surface defect types on the NEU-DET dataset, achieving mAP@0.5 > 0.70
- Conducted end-to-end pipeline: data preparation, training optimization, PR curve analysis, and confusion matrix-based error analysis
- Exported the model to ONNX format and validated inference with ONNX Runtime, ensuring accuracy alignment with PyTorch
- Tech stack: Python / PyTorch / YOLOv8 / ONNX / OpenCV

---

## 七、第一步具体操作

项目启动后，第一件事：

```bash
# 1. 创建新项目目录
mkdir D:\01_Base\CodingSpace\YOLO_Defect_Detection
cd D:\01_Base\CodingSpace\YOLO_Defect_Detection
git init

# 2. 创建 conda 环境
conda create -n yolo_defect python=3.9 -y
conda activate yolo_defect

# 3. 安装依赖
pip install ultralytics onnx onnxruntime opencv-python matplotlib

# 4. 验证安装
yolo version

# 5. 下载 NEU-DET 数据集
# 来源：https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
# 或者直接搜索 "NEU-DET dataset download"
```

下载完数据集后的第一步是分析数据分布（每类多少张、图片尺寸、标注格式），然后再开始转格式和训练。
