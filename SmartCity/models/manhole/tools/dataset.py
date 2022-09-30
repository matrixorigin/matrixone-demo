import cv2
import os
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import sys


class MyDataset(Dataset):
    def __init__(self, dir):
        super(MyDataset, self).__init__()

        self.input_path = []

        if os.path.isdir(dir):
            for root, dirs, files in os.walk(dir):
                for file in files:
                    path = os.path.join(root, file)
                    self.input_path.append(path)

    def __getitem__(self, index):
        path = self.input_path[index]
        img_org = cv2.imread(path, cv2.IMREAD_COLOR)
        img_org = cv2.resize(img_org, (1920, 1080))
        # img_640, r = preproc(img_org)
        return img_org, path

    def __len__(self):
        return len(self.input_path)
