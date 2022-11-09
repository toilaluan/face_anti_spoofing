import torch
import torch.nn as nn
import sys
import torchvision
from torchvision.models import resnet50, EfficientNet_B5_Weights
class EffHandNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b5(weights = EfficientNet_B5_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(2048, 1)
        )
        self.backbone.features[0][0] = nn.Conv2d(4,48,(3,3), stride=(2,2), padding = (1,1), bias = False)
        # self.backbone.features.
        origin_stdout = sys.stdout
        with open("backbone.txt", 'w') as f:
            sys.stdout = f
            print(self.backbone.features[0][0])
            sys.stdout = origin_stdout
        # print(self.swin)
        # self.swin.head = nn.Linear(768, 1)
    def forward(self, x):
        x = self.backbone(x)
        # x = self.head(x)
        return x.sigmoid()
    
if __name__ == '__main__':
    model = EfficientNet()
    lbp = torch.zeros((4,1,224,224))
    x = torch.zeros((4, 3,224,224))
    print(model(x, lbp).shape)