import cv2
import os
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import sys
from .preproc import preproc

class MyDataset(Dataset):
    def __init__(self, dir):
        super(MyDataset, self).__init__()

        self.input_path = []

        if os.path.isdir(dir):
            for root, dirs, files in os.walk(dir):
                for file in files:
                    path = os.path.join(root, file)
                    if not 'mask' in file:
                        self.input_path.append(path)

    def __getitem__(self, index):
        path = self.input_path[index]
        img_org = cv2.imread(path, cv2.IMREAD_COLOR)
        img_640, r = preproc(img_org)
        return img_org, img_640, path, r


    def __len__(self):
        return len(self.input_path)


if __name__ == '__main__':
    dataset = MyDataset('test_imgs/')
    BATCH_SIZE = 8
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)
    begin = time.time()
    count = 0
    for step, data in enumerate(dataloader):
        img_org, img_640, path, r = data
        print(img_org.shape)
        print(img_640.shape)
        print(r)
        print()
