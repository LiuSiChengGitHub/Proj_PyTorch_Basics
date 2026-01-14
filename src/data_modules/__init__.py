# 数据集模块
"""
提供三种 Dataset 实现:
- MyData: 基础分类数据集 (支持 transform)
- ClassificationDataset: 简单分类数据集 (使用 cv2)
- DetectionDataset: 目标检测数据集 (图片+标签文件)
"""

# 基础模块直接导入 (只依赖 PIL)
from .base import MyData

# cv2 依赖的模块使用延迟导入
def __getattr__(name):
    if name == "ClassificationDataset":
        from .classification import ClassificationDataset
        return ClassificationDataset
    elif name == "DetectionDataset":
        from .detection import DetectionDataset
        return DetectionDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['MyData', 'ClassificationDataset', 'DetectionDataset']
