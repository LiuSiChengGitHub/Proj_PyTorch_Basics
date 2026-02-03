import torch
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(
    root="../datasets",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

dataloader = DataLoader(datasaet, batch_size=64) 

class Tudui(nn.Module):
    super(Tudui, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)