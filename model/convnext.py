import torch
import torch.nn as nn
import sys
import torchvision
import timm
class ConvNext(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnext_base_384_in22ft1k', True)
        self.backbone.head.fc = nn.Identity()
        self.backbone.head.drop = nn.Identity()
        print(self.backbone)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1, bias=False)
        )
        # self.small_branch = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(1536, 256)
        # )
        # print(self.backbone.stages[-1])
    def forward(self, x):
        features = self.backbone(x)
        # print(features.shape)
        logits = self.classifier(features)
        # features = self.small_branch(features)
        # x = self.head(x)
        return logits.sigmoid()
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
if __name__ == '__main__':
    device = torch.device('cuda')
    model = ConvNext()
    model.to(device)
    x = torch.zeros((4, 3,600, 600)).to(device)
    prob = model(x)
    print(prob.shape)