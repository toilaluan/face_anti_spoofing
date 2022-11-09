import torch
import torch.nn as nn
import sys
import torchvision
import timm
class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv2_120d', pretrained=True)
        self.backbone.classifier = nn.Linear(1280, 1, bias=False)
        print(self.backbone)
    def forward(self, x):
        x = self.backbone(x)
        # x = self.head(x)
        return x.sigmoid()
    
if __name__ == '__main__':
    device = torch.device('cuda')
    model = MobileNet()
    model.to(device)
    x = torch.zeros((64, 3,224,224)).to(device)
    print(model(x).shape)