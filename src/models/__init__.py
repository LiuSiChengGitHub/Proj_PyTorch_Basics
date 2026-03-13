# 模型定义模块
"""
models/
└── simple_cnn.py  # SimpleCNN: 适配 CIFAR-10 的简单 CNN
"""


# 使用延迟导入，避免强制依赖 torch
def __getattr__(name):
    if name == "SimpleCNN":
        from .simple_cnn import SimpleCNN
        return SimpleCNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SimpleCNN"]
