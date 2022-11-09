import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights, Swin_S_Weights
class Swin_S(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = torchvision.models.swin_s(weights = Swin_S_Weights.IMAGENET1K_V1)
        self.swin.head = nn.Linear(768, 1)
    def forward(self, x):
        x = self.swin(x)
        # x = self.head(x)
        return x.sigmoid()
    
if __name__ == '__main__':
    model = Swin_S()
    x = torch.zeros((4, 3, 600,600))
    print(model(x).shape)