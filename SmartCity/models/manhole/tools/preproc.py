import math
import random
import cv2
import numpy as np
import os
import torch
import time
import sys
import torch.nn as nn
from torch.nn import functional as F
from yolox.utils import xyxy2cxcywh

def preproc_torch(image, input_size=(640,640), mean=torch.Tensor([0.406, 0.485, 0.456]), std=torch.Tensor([0.225, 0.229, 0.224])):

    padded_img = torch.ones(image.shape[0], 3, input_size[0], input_size[1]) * 114.0  # 4, 3, 416(h), 416(w)
    r = min(input_size[0] / image.shape[1], input_size[1] / image.shape[2])  # min(416/1080,416/1920)=416/1920

    # 将 4,h,w,3 转成 4,3,h,w
    image = image.permute(0, 3, 1, 2)  # 4, 3, h, w
    resized_img = F.interpolate(image, size=(int(image.shape[2] * r), int(image.shape[3] * r)))  # 4, 3, h*r, w*r

    padded_img[:, :, :resized_img.shape[2], :resized_img.shape[3]] = resized_img  # 顶着左上角填充

    #BGR转RGB
    index = [2, 1, 0]
    padded_img = padded_img[:, index, :, :]

    padded_img /= 255.0
    if mean is not None:
        mean=mean.unsqueeze(-1).unsqueeze(-1)
        padded_img -= mean

    if std is not None:
        std=std.unsqueeze(-1).unsqueeze(-1)
        padded_img /= std

    return padded_img, r

def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r