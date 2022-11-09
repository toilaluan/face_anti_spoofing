import torch
import torch.nn as nn
import sys
import torchvision
import timm
from torchvision.models import resnet50, EfficientNet_B7_Weights
class Eff_v2_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_xl_in21k', pretrained=True)
        print(self.backbone)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Dropout(0.5, inplace=True),
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
    x = torch.zeros((4, 3,384, 384)).to(device)
    print(model(x).shape)