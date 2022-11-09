import torch
import torch.nn as nn
import sys
import torchvision
import timm
from torchvision.models import resnet50, EfficientNet_B7_Weights
class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b7(EfficientNet_B7_Weights.IMAGENET1K_V1)
        print(self.backbone)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )
    def forward(self, x):
        x = self.backbone(x)
        # x = self.head(x)
        return x.sigmoid()
if __name__ == '__main__':
    device = torch.device('cuda')
    model = EfficientNet()
    model.to(device)
    x = torch.zeros((1, 3,672, 672)).to(device)
    print(model(x).shape)