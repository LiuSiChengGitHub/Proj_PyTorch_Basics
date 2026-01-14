# PyTorch 学习项目 - 核心模块
"""
src/
├── data_modules/   # 数据集类定义
└── transforms/     # 数据变换预设
"""

# 使用延迟导入，避免强制依赖
def __getattr__(name):
    if name == "data_modules":
        from . import data_modules
        return data_modules
    elif name == "transforms":
        from . import transforms
        return transforms
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['data_modules', 'transforms']
