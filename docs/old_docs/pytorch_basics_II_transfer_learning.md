
# PyTorch 基础学习项目 — Phase 2 迁移学习笔记

> Phase 2 收尾文档：迁移学习核心概念 + 实验总结 + 面试考点 最后更新：2026年3月21日

---

## 一、迁移学习核心概念

### 什么是迁移学习

一句话：把在大数据集（ImageNet，120 万张图）上训练好的模型权重，搬到小数据集（蚂蚁/蜜蜂，244 张图）上用，只微调最后几层。

```python
# 加载 ImageNet 预训练的 ResNet18，替换最后的分类头
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # 1000 类 → 2 类
```

为什么有效：预训练模型已经学会了"边缘、纹理、形状"等通用视觉特征，这些特征对大多数图像分类任务都适用。小数据集根本不够从零学这些特征（244 张图 vs 1100 万参数），但可以直接复用预训练特征，只训练分类决策。

### Freeze vs Fine-tune

| 策略 | 含义 | 适用场景 |
|------|------|---------|
| 冻结训练（Freeze） | 冻结所有卷积层，只训练 fc | 数据量极小、目标域和源域相似 |
| 微调（Fine-tune） | 解冻部分/全部层，用小学习率训练 | 数据量适中、需要适配目标域特征 |

渐进解冻策略（本项目采用）：

- 阶段 A：冻结全部卷积层，只训练 fc（1,026 个参数）
- 阶段 B：解冻 layer4 + fc（8,394,754 个参数），用更小的学习率

为什么不一上来就全量训练：大学习率 + 全部参数 = 破坏预训练特征（灾难性遗忘）。渐进解冻让 fc 先适应新任务，再微调高层特征。

### 替换 fc 层

```python
model = resnet18(weights=ResNet18_Weights.DEFAULT)
in_features = model.fc.in_features  # 512（ResNet18 最后一层输出维度）
model.fc = nn.Linear(in_features, num_classes)  # 替换为新的分类头
```

为什么替换 fc 而不是前面的卷积层：

- 前面的卷积层提取的是通用特征（边缘、纹理），对所有图像任务都有用
- fc 层是针对 ImageNet 1000 类的分类决策，和你的任务（2 类）完全不同，必须换掉
- `model.fc.in_features` 动态读取维度，换成 ResNet50（2048）代码也不用改

---

## 二、关键代码片段

### 1. 加载预训练模型 + 替换 fc

```python
from torchvision.models import resnet18, ResNet18_Weights

def build_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
```

### 2. 冻结参数

```python
def set_trainable_layers(model, stage):
    # 先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 再解冻需要训练的部分
    if stage == "head":
        modules = [model.fc]
    elif stage == "finetune":
        modules = [model.layer4, model.fc]

    for module in modules:
        for param in module.parameters():
            param.requires_grad = True
```

### 3. 只为可训练参数创建 optimizer

```python
def build_optimizer(model, lr):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable_params, lr=lr)
```

为什么不直接传 `model.parameters()`：如果传全部参数，Adam 会为冻结参数也维护动量状态，浪费内存。只传可训练参数更干净。

### 4. 推理时的关键步骤

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

- `unsqueeze(0)`：单张图没有 batch 维度，插入一个维度变成 `(1, 3, 224, 224)` 让模型能处理
- `val_transform`：推理必须用验证集的固定预处理，不能用训练增强
- `softmax`：把 logits 转换为概率分布，方便人阅读

---

## 三、实验结果对比

### SimpleCNN vs ResNet18 迁移学习

| 对比项 | SimpleCNN（Phase 1） | ResNet18 迁移学习（Phase 2） |
|--------|---------------------|---------------------------|
| 模型来源 | 从零搭建 | ImageNet 预训练 |
| 参数量 | ~62K | 11,177,538（实际只训练 1K~8M） |
| 数据集 | CIFAR-10（50,000 张，10 类） | hymenoptera（244 张，2 类） |
| 输入尺寸 | 32x32 | 224x224 |
| 训练 epoch | 10 | 3+3（两阶段各 3） |
| 最佳准确率 | ~65% | 94.77% |
| 训练策略 | 全量训练 | 渐进解冻（先 fc，再 layer4+fc） |

两个实验用的不是同一个数据集，不能直接比准确率绝对值。关键对比点是：244 张图就能训到 94.77%，说明迁移学习在小数据集上远强于从零训练。

### 预训练 vs 非预训练（同一数据集 hymenoptera）

| 对比项 | 预训练（weights=DEFAULT） | 非预训练（weights=None） |
|--------|-------------------------|------------------------|
| 阶段 A 第 1 个 epoch val_acc | 84.97% | 45.75%（接近随机猜测） |
| 阶段 A 最终 best | 92.16% | 69.28% |
| 阶段 B 最终 best | 94.77% | 69.93% |
| 总训练 epoch | 6 | 20 |
| 训练稳定性 | 单调上升 | 剧烈震荡 |

结论：244 张图从零训练 ResNet18 完全不够，20 个 epoch 还不如预训练版第 1 个 epoch 的成绩。

### 学习率对比实验

| 组合 | HEAD_LR | FINETUNE_LR | 最终 best val_acc |
|------|---------|-------------|------------------|
| 组合 1 | 1e-3 | 1e-4 | **94.77%** |
| 组合 2 | 1e-3 | 1e-5 | 93.46% |
| 组合 3 | 5e-4 | 1e-4 | 94.12% |

结论：

- 组合 2 的 FINETUNE_LR=1e-5 太小，阶段 B 的 train_loss 几乎不动，学不动
- 组合 3 的 HEAD_LR=5e-4 起步慢，阶段 A 第 1 个 epoch 只有 67.97%
- 最优区间在 HEAD_LR=1e-3 + FINETUNE_LR=1e-4 附近

### 阶段策略对比

| 版本 | 训练范围 | 可训练参数 | 最终 best val_acc |
|------|---------|-----------|------------------|
| 版本 A | 只训 fc | 1,026 | 92.16% |
| 版本 B | layer4 + fc | 8,394,754 | **94.77%** |
| 版本 C | 全模型 | 11,177,538 | 未超越 B（过拟合） |

结论：小数据集上，渐进解冻比全量微调更稳定。版本 C 的 train_loss 降到 0.05 但 val_acc 反降到 86%，典型过拟合。

---

## 四、面试高频考点

| # | 问题 | 一句话答案 |
|---|------|-----------|
| 1 | 为什么 pretrained 模型在小数据集上也有效？ | 预训练模型已在大数据集上学会通用视觉特征（边缘、纹理、形状），小数据集只需微调分类决策，不需要重新学底层特征 |
| 2 | 为什么要冻结前面的层？ | 前面的层提取的是通用特征且已经很好了；大学习率会破坏这些特征（灾难性遗忘），小数据集也不够重新学 |
| 3 | backbone 和 head 的学习率怎么设？ | head（fc）用较大学习率（1e-3），因为随机初始化要从头学；backbone 用小学习率（1e-4），是在已有好权重上微调 |
| 4 | 迁移学习 vs 从头训练怎么选？ | 数据量大（>10 万）+ 目标域和 ImageNet 差异大 → 可以从头训；数据量小或域相似 → 迁移学习 |
| 5 | 如果要把模型部署到边缘设备怎么做？ | `torch.onnx.export()` 导出 ONNX → ONNX Runtime 或 TensorRT 推理，减少 PyTorch 依赖，提升推理速度 |
| 6 | 为什么输入必须是 224x224 + ImageNet Normalize？ | 模型权重在该分布下训练，改变输入分布导致每层激活值偏离预训练时的期望，特征提取失效 |
| 7 | val_acc 不升但 train_loss 在降说明什么？ | 过拟合。模型在记忆训练集而非学习泛化规律。应减少可训练参数、增加数据增强或提前停止 |

---

## 五、踩坑记录

**输入预处理必须和预训练一致**：去掉 ImageNet Normalize 后准确率明显下降。ResNet18 的卷积核是在 [-2, 2] 范围的标准化输入下优化的，输入 [0, 1] 范围的 tensor 导致激活值偏移。

**非预训练模型在阶段 A 几乎学不动**：`weights=None` 时阶段 A 的 train_loss 10 个 epoch 只从 0.714 降到 0.713。随机初始化的特征层没有任何区分能力，fc 层在对"噪声"做分类。

**全量微调反而不如渐进解冻**：版本 C（训练全部 11M 参数）在 244 张图上严重过拟合，train_loss 降到 0.05 但 val_acc 反降到 86%。小数据集上参数越多，过拟合风险越大。

**每个阶段的 optimizer 是重新创建的**：进入版本 C 时 val_acc 第 1 个 epoch 暴跌到 86%，原因是新 optimizer 没有继承之前的动量状态，等于突然对所有参数做全新的梯度更新。

**学习率太小等于没在学**：FINETUNE_LR=1e-5 时阶段 B 的 train_loss 3 个 epoch 只从 0.303 降到 0.241，val_acc 只到 93.46%。学习率不是越小越好。

**checkpoint 要保存足够信息**：只保存 model_state_dict 不够，还需保存 class_to_idx（类别映射）、epoch、stage、optimizer_state_dict，否则无法恢复训练或正确推理。

**类别映射反转**：训练时 `class_to_idx = {'ants': 0, 'bees': 1}`，推理时需要反转为 `idx_to_class = {0: 'ants', 1: 'bees'}`，否则预测类名对不上。

**unsqueeze 不是取元素**：`unsqueeze(0)` 是增加维度（(3,224,224) → (1,3,224,224)），不是从列表取第一个元素。推理单张图必须加这一步。

---

## 六、从 SimpleCNN 到 ResNet18 的认知升级

Phase 1 你学会了从零搭建 CNN（3 层 Conv + 2 层 Linear，~62K 参数）在 CIFAR-10 上跑通训练循环。Phase 2 你发现：当数据量只有 244 张时，从零训练一个 11M 参数的 ResNet18 连 50% 都达不到，但加载预训练权重后 6 个 epoch 就能到 94.77%。这中间的差距不是"模型更深"就能解释的，核心是预训练权重里沉淀了 120 万张图学到的通用特征。你从 Phase 1 的"自己造轮子"进化到 Phase 2 的"站在巨人肩膀上"——这就是工业界 90% 的 CV 项目选择迁移学习而不是从零训练的原因。

---

## 七、对应文件

| 文件 | 作用 |
|------|------|
| `examples/train_transfer.py` | ResNet18 两阶段迁移学习训练脚本 |
| `examples/predict_transfer.py` | 单图/文件夹批量推理脚本 |
| `src/transforms/presets.py` | 训练/验证 transform（ImageNet 标准化） |
| `docs/resnet18_transfer_learning_plan.md` | 迁移学习完整学习方案（任务清单 + 实验设计） |
