from torch.utils.data import DataLoader, Dataset
import glob
import pandas as pd
import cv2
import torch
import imgaug.augmenters as iaa
import random
import math
import numpy as np
from .utils.lbp import LocalBinaryPatterns
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
def resizeKeepRatio(im, new_h):
    h, w, _ = im.shape
    new_w = math.floor(new_h/h * w)
    new_w = 288
    im = cv2.resize(im, (new_w, new_h))
    return im
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
        frames.append(image)
        count += 1 # moved this to be accurate with my 'second frame' statement
    return frames
    # this is where you put your functionality (this just saves the frame)
    # cv2.imwrite("frame%d.jpg" % count, image)
class FLDTestDataset(Dataset):
    def __init__(self, path, img_size = 288, transform=None):
        self.path = path
        self.transform = transform
        self.img_size = img_size
        self.video_paths = glob.glob(path + "/videos/*.mp4")
        self.LBP = LocalBinaryPatterns(8,1)

    def __len__(self):
        return len(self.video_paths)
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split("/")[-1]
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        _, frame = cap.read()
        frame = resizeKeepRatioPadding(frame, self.img_size)
        if self.transform is not None:
            frame = self.transform(frame)
        else:
            frame = frame / 255.0
            frame = torch.tensor(frame, dtype = torch.float32).permute(2,0,1)
        # frame2 = torch.tensor(frame2, dtype = torch.float32).permute(2,0,1)
        return video_name, frame
if __name__ == '__main__':
    dataset = FLDTestDataset("/home/aimenext/luantt/zaloai/face_liveness_detection/dataset/train")
    label, image = dataset[1]
    print(image, label)
        