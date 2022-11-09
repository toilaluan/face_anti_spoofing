import torch
import torch.nn as nn
import torchvision
import timm
from torchvision.models import resnet50, ResNet50_Weights, Swin_S_Weights, ViT_L_16_Weights
class VIT_H_14(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = timm.create_model('vit_base_r50_s16_384', pretrained=True)
        self.swin.head = nn.Linear(768, 1)
        # self.swin.head = nn.Linear(768, 1)
    def forward(self, x):
        x = self.swin(x)
        # x = self.head(x)
        return x.sigmoid()
    
if __name__ == '__main__':
    device = torch.device('cuda')
    model = VIT_H_14()
    model.to(device)
    x = torch.zeros((12, 3,384,384)).to(device)
    print(model(x).shape)