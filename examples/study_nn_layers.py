# 用于测试卷积、池化等组件

import torch

from torch import nn

class newNN(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,input):
        output = input +1
        return output

newNN1 = newNN()
x = torch.tensor(1.0)
output = newNN1(x)
print(output)