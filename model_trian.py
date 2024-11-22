import os
import numpy as np
import  torch
from torch.utils.data import DataLoader, Dataset
import cv2
import torch.nn as nn
import torch.nn.functional as F

#获取图像数据，转为tensor张量格式
class  CircleDataset(Dataset):

    def __init__(self, data_dir, img_size=128,transform=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.annotations = []
        with open(os.path.join(data_dir, "annotations.cvs"), "r") as f:
            self.annotations = [line.strip().split(",") for line in f.readlines()]
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path, x, y = self.annotations[idx]
        img = cv2.imread(os.path.join(self.data_dir, img_path))
        img = cv2.resize(img, (self.img_size, self.img_size))
        x, y = float(x), float(y)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        label = torch.tensor([x, y], dtype=torch.float32)
        return img, label


class CircleCenterNet(nn.Module):

    def __init__(self):
        super(CircleCenterNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2,stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2,stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2,stride=2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x





