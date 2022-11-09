import torch
import torch.nn as nn
from ignite.handlers import create_lr_scheduler_with_warmup
from data.dataset import FLDDataset
from torch.utils.data import DataLoader
import torchvision
from model.resnet50 import Resnet50
from model.swin_s import Swin_S
from model.efficient_net import EfficientNet
from model.effv2 import Eff_v2_net
from model.eff_net_handcraft import EffHandNet
from model.convnext import ConvNext
from model.handcraft_net import HandcraftNet
from model.vit import VIT_H_14
from loss.focal_loss import sigmoid_focal_loss
from torch import optim as optim
from sklearn.metrics import accuracy_score
from metrics.eer import eer
import torchvision.transforms as T
import random
import numpy as np
from tqdm import tqdm
import timm
from model.mobilenet import MobileNet
import albumentations as A
import cv2
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(2301)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/eff_b7_no_aug")
def validate(model, val_loader, device):
    cpu_device = torch.device('cuda')
    # model.to(cpu_device)
    model.eval()
    preds = []
    labels = []
    prob_preds = []
    for label, image, _ in tqdm(val_loader):
        image = image[:,:,0,:,:]
        # print(label)
        label = label.to(cpu_device)
        image = image.to(cpu_device)
        
        # image_lbp = image_lbp.to(device)
        outputs = model(image)
        prediction = outputs > 0.5
        # print(prediction)
        preds.extend(prediction.tolist())
        prob_preds.extend(outputs.tolist())
        # print(label)
        labels.append(label.item())
        # print(preds, labels)
    # print(prob_preds, preds)
    preds = np.array(preds).astype(np.int32)
    labels = np.array(labels)
    # model.to(device)
    return accuracy_score(preds, labels), eer(labels, prob_preds)


img_size = 600



trans = T.Compose([
    T.ToPILImage(),
    T.RandomCrop(500, pad_if_needed = True),
    T.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 3)),
    T.ColorJitter(0.2, 0.2, 0.2, 0.2),
    T.RandomRotation(10),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomPosterize(2, p=0.1),
    T.RandomEqualize(p=0.1),
    T.RandomAdjustSharpness(1.5),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

val_trans = trans = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = FLDDataset("dataset/new_data/train", img_size=img_size, transform=trans, n_frame=16, only_mixed=False)
val_dataset = FLDDataset("dataset/new_data/train", img_size=img_size, transform=val_trans, n_frame=1, only_mixed=False)
train_size = int(1 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, test_size])
_, test_dataset = torch.utils.data.random_split(val_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size = 2, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size = 1)
epochs = 100
device = torch.device("cuda")
model = EfficientNet()
model.load_state_dict(torch.load("/home/aimenext/luantt/face_liveness_detecion/weights/eff_b7_no_aug_5.pth")['model_state_dict'])
model.to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr = 5e-4)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20,60] , gamma=0.2)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
a_loss = nn.MSELoss()
min_eer = 0.03
accum_iter = 4
t = 0
for epoch in range(epochs):
    
    model.train()
    epoch_loss = 0.
    # acc, eer = validate(model, val_loader, device)
    # model.zero_grad()
    for i, (label, images, ft_images) in enumerate(train_loader):
        t = t % images.shape[2]
        image = images[:,:,t,:,:]
        # ft_image = ft_images[:,t,:]
        # ft_image = ft_image.to(device)
        label = label.to(device)
        image = image.to(device)
        outputs = model(image)
        label = label.unsqueeze(1)
        # loss = criterion(outputs, label)
        loss, ce_loss = sigmoid_focal_loss(outputs, label, reduction='mean', alpha=-1)
        # # print(ft_image.shape)
        # loss_ft = a_loss(features, ft_image)
        # loss = 0.99*loss_cls + 0.05*loss_ft
        loss = loss / accum_iter
        loss.backward()
        if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss += loss.item()
        # print(ce_loss.mean().item())
        print("Epoch {}, Lr [{}], Step [{}/{}], Loss [{:.2f}], CE_loss [{:2f}]".format(epoch, scheduler.get_last_lr(), i, len(train_loader), loss.item()*accum_iter, ce_loss.mean().item()))
    t += 1
    scheduler.step()
    writer.add_scalar("Epoch_loss", epoch_loss*accum_iter / len(train_loader), epoch)        
    # acc, eer_score = validate(model, val_loader, device)
    # print("Epoch {}, acc [{:.2f}], eer [{:.2f}]".format(epoch, acc, eer_score))
    # writer.add_scalar("eer", eer_score, epoch)
    # writer.add_scalar("acc", acc, epoch)
    path_save_ckpt = 'weights/eff_b7_no_aug_{}.pth'.format(epoch)
    # if eer_score < min_eer:
    #     min_eer = eer_score
    #     torch.save({
    #         # 'epoch' : epoch,
    #         'model_state_dict' : model.state_dict(),
    #         # 'optimizer_state_dict': optimizer.state_dict(),
    #         # 'loss' : loss.item()
    #     }, path_save_ckpt)
    if epoch % 10 == 0:
        torch.save({
            # 'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss' : loss.item()
        }, path_save_ckpt)