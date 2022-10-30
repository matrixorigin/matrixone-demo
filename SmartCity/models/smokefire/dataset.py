import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from PIL import Image
import numpy as np
#import torch.optim as optim
import os
import cv2
# import skimage
# from skimage import util
import random
from scipy.io import loadmat
#import icecream as ic


def img_loader(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')  # 128,128,3
    img = img.transpose(2, 0, 1)
    img = img - np.mean(img)
    img = img / np.max(np.abs(img))

    return img


class MyDataset(Dataset):
    def __init__(self, txt_dir, img_loader=img_loader):
        super(MyDataset, self).__init__()
        f_input = open(txt_dir, 'r')
        self.loader = img_loader
        self.input = []

        for line in f_input:
            line = line.strip('\n')
            self.input.append(line)

    def __getitem__(self, index):
        line = self.input[index]
        img = self.loader(line)
        if line.split('/')[-2]!='peace':
            label = torch.tensor(1., dtype=torch.int64)
        else:
            label = torch.tensor(0., dtype=torch.int64)
        return img,line,label

    def __len__(self):
        return len(self.input)