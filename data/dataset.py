from torch.utils.data import DataLoader, Dataset
import glob
import pandas as pd
import cv2
import torch
import imgaug.augmenters as iaa
import random
import numpy as np
import math
from .utils.generateFT import *
from torch.utils.data import DataLoader
# from .utils.lbp import LocalBinaryPatterns
import torchvision.transforms as T
def resizeKeepRatio(im, new_h, new_w):
    im = cv2.resize(im, (new_w, new_h))
    return im
def resizeKeepRatioPadding(im, desired_size):
    # im = cv2.imread(im)
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im
def getAllFrames(cap):
    frames = []
    success,image = cap.read()
    frames.append(image)
    count = 0
    success = True

    # so long as vidcap can read the current frame... 
    while success:

    # ...read the next frame (this is now your current frame)
        success,image = cap.read()
        if image is None:
            continue
        frames.append(image)
        count += 1 # moved this to be accurate with my 'second frame' statement
    return frames
def get_random_crop(image, crop_height, crop_width):
    
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = image[y: y + crop_height, x: x + crop_width]

        return crop
def mixupImageInVideo(images, desired_size):
        h, w, _ = images[0].shape
        out_size = desired_size
        desired_size = min(h, w, desired_size)
        dw = desired_size // 2
        dw_last = desired_size - dw
        dh = desired_size // 2
        dh_last = desired_size - dh
        result = np.zeros((desired_size, desired_size, 3))
        frames = random.choices(images, k=4)
        result[0:dh, 0:dw] = get_random_crop(frames[0], dh, dw)
        result[0:dh, dw:] = get_random_crop(frames[1], dh, dw_last)
        result[dh:, 0:dw] = get_random_crop(frames[2], dh_last, dw)
        result[dh:, dw:] = get_random_crop(frames[3], dh_last, dw_last)
        # result = cv2.resize(result, (out_size, out_size))
        return result
class FLDDataset(Dataset):
    def __init__(self, path, img_size = 288, transform = None, n_frame = 8, mode = 'train', only_mixed = False, fframe = False):
        # self.LBP = LocalBinaryPatterns(8,1)
        self.path = path
        self.img_size = img_size
        self.trained_video = {}
        self.n_frame = n_frame
        self.only_mixed = only_mixed
        self.fframe = fframe
        self.transform = transform
        # print(self.video_paths)
        if mode == 'train':
            self.label_path = path + "/label.csv"
            self.labels = pd.read_csv(self.label_path)
            self.labels.set_index("fname", inplace=True)
        self.frame_paths = glob.glob(self.path + "/frames/*/")
        self.mode = mode
        # print(self.frame_folder)
    def _get_frame_idx(self, idx, cap):
        success, image = cap.read()
        if idx == 0:
            return image
        count = 0
        while success:
            success, image = cap.read()
            count += 1
            if count == idx:
                return image
        return image
    def __len__(self):
        return len(self.frame_paths)
        
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        frame_paths = glob.glob(frame_path +'*')
        video_name = frame_path.split('/')[-2]
        
        sub_frames = frame_paths[::(len(frame_paths) // self.n_frame)]
        # print(sub_frames)
        sub_frames = sub_frames[:self.n_frame]
        frames = []
        
        for p in sub_frames:
            f = cv2.imread(p)
            frames.append(f)
        if self.img_size is None:
            return video_name, frames
        if self.only_mixed == True:
            mixed_frames = []
            for i in range(self.n_frame):
                cutMixedFrame = mixupImageInVideo(frames, self.img_size * 4)
                cutMixedFrame = cv2.resize(cutMixedFrame, (self.img_size, self.img_size))
                mixed_frames.append(cutMixedFrame.astype(np.uint8))
            frames = mixed_frames
        # cv2.imwrite("mixup.png", cutMixedFrame)
        transformed_frames = []
        FT_frames = []
        for frame in frames:
            # cv2.imwrite("debug.png", frame)
            if self.only_mixed:
                frame = cv2.resize(frame, (self.img_size, self.img_size))
            else:
                frame = resizeKeepRatioPadding(frame, self.img_size)
            if self.transform is not None:
                if self.fframe:
                    fframe = generate_FT(frame)
                    fframe = cv2.resize(fframe, (16,16))
                    fframe = torch.from_numpy(fframe).float()
                    fframe = fframe.flatten()
                # print(fframe.shape)
                # fframe = fframe.unsqueeze(0)
                # cv2.imwrite("debug1.png", frame)
                frame = self.transform(frame)
                # cv2.imwrite('debug.png', tmp)
                # frame = np.array(frame.permute(1,2,0))
                # frame = self.t(frame)
            else:
                frame = frame / 255.0
                frame = torch.tensor(frame, dtype = torch.float32).permute(2,0,1)
            transformed_frames.append(frame)
            if self.fframe:
                FT_frames.append(fframe)
                FT_frames = torch.stack(FT_frames, dim=0)
        transformed_frames = torch.stack(transformed_frames, dim=1)
        # print(transformed_frames.shape)
        # print(FT_frames.shape)
            # lbp = torch.tensor(lbp, dtype = torch.float32).permute(2,0,1)
        # print(lbp.shape)
        if self.mode == 'train':
            label = int(self.labels.loc[video_name])
            return torch.tensor(label, dtype = torch.float32), transformed_frames, FT_frames
        else:
            return video_name, transformed_frames, FT_frames
if __name__ == '__main__':
    trans = T.Compose([
    T.ToPILImage(),
    T.RandomRotation(10),
    T.RandomVerticalFlip(p=0.5),
    T.RandomAdjustSharpness(1.5),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
    dataset = FLDDataset("/home/aimenext/luantt/face_liveness_detecion/dataset/new_data/train", 672, trans, only_mixed=True)
    label, img, fimg = dataset[5]
    cv2.imwrite("debug.jpg", np.array(img[:,-1,:,:].permute(1,2,0)*255).astype(np.uint8))
    print(img[-1].shape)
    # dataloader = DataLoader(dataset, 4)
    # label, video = next(iter(dataloader))
    # print(video.shape)
        