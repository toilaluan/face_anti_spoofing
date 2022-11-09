import torch
import torch.nn as nn
from data.dataset import FLDDataset
from torch.utils.data import DataLoader
import torchvision
from model.efficient_net import EfficientNet
from model.handcraft_net import HandcraftNet
from model.mobilenet import MobileNet
from torch import optim as optim
from model.convnext import ConvNext
import numpy as np
import glob
import torchvision.transforms as T
from tqdm import tqdm
trans = T.Compose([
    # T.ToPILImage(),
    # T.RandomRotation(10),
    # T.RandomVerticalFlip(p=0.5),
    # T.RandomAdjustSharpness(1.5),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
def main():
    model = EfficientNet()
    model.load_state_dict(torch.load(pretrained_path))
    model.eval()
    model.to(device)
    result = {}
    dataset = FLDDataset(test_path, 600, trans, 1, 'test', False)
    for video_name, image, _ in tqdm(dataset):
        image = image.to(device)
        # high = 0.
        # low = 1.
        res = 0.
        for t in range(image.shape[1]):
            img = image[:, t,:,:]
            print(img.shape)
            img = img.unsqueeze(0)
            out = model(img)
            out = out.detach().cpu()
            # high = max(out,high)
            # low = min(out,low)
            res += out[0][0].item()
        res /= image.shape[1]
        # print(res[0][0].item())
        result[video_name] = res
    with open("result/Predict.csv", 'w') as f:
        f.write("fname,liveness_score\n")
        for key, value in result.items():
            f.write("{},{}\n".format(key+'.mp4', value))

if __name__ == '__main__':
    pretrained_path = "/home/aimenext/luantt/face_liveness_detecion/weights/eff_b7_fulldata40.pth"
    test_path = "dataset/new_data/public"
    device = torch.device("cuda")
    main()