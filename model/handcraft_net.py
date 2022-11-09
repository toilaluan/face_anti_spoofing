import torch
import torch.nn as nn
import sys
import torchvision
import timm
from torchvision.models import resnet50, EfficientNet_B5_Weights

class HandcraftNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_h = timm.create_model('tinynet_a')
        print(self.backbone_h)
        self.backbone_h.classifier = nn.Identity()
        self.backbone_rgb = torchvision.models.efficientnet_b5(weights = EfficientNet_B5_Weights.IMAGENET1K_V1)
        self.backbone_h.conv_stem = nn.Conv2d(1,32,(3,3),(2,2),(1,1),bias=False)
        self.backbone_rgb.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, 1280),
            nn.ReLU(),
        )
        self.fuser = nn.Sequential(
            nn.Linear(1280, 1)
        )
    def forward(self, x1, x2):
        x2 = self.backbone_h(x2)
        x1 = self.backbone_rgb(x1)
        x = x1+x2
        x = self.fuser(x)        
        # x = self.head(x)
        return x.sigmoid()
    
if __name__ == '__main__':
    model = EfficientNet()
    x1 = torch.zeros((4, 3,288,288))
    x2 = torch.zeros((4, 1,288,288))
    print(model(x1, x2).shape)