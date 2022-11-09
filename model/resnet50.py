import torch
import torch.nn as nn
from data.dataset import FLDDataset
from torch.utils.data import DataLoader
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
class Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(weights= ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = torch.nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.fc2 = torch.nn.Linear(1024, 1)
        print(self.resnet)
    def forward(self, x):
        x = self.resnet(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.head(x)
        return x.sigmoid()