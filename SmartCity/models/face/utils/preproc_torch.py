import cv2
import os
import numpy as np
import torch
import time
import sys
import torch.nn as nn
from torch.nn import functional as F


def preproc_torch(image, input_size=(416, 416), mean=torch.Tensor([0.406, 0.485, 0.456]), std=torch.Tensor([0.225, 0.229, 0.224])):

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


if __name__ == '__main__':
    print('************ numpy ************')
    img = cv2.imread('../test_imgs/ccy.jpg')
    # print(img.shape)# 1080, 1920, 3
    img_640, r = preproc(img)

    print('************ Tensor ************')
    img_tensor = torch.from_numpy(np.expand_dims(img, axis=0))
    # print(img_tensor.shape) # 1, 1080, 1920, 3
    img_640_tensor, r1 = preproc_torch(img_tensor)
