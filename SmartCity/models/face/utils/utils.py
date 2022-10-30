import numpy as np
from torch.nn import functional as F
import torch
import cv2
import os
import torch.nn as nn
import sys
import json
import pickle


def calculate_cos_distance(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    cose = torch.mm(a, b.t())
    return 1 - cose


def compute_cos_similarity(x, y):
    x = F.normalize(x, dim=1)
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()

    dist = torch.ones_like(dist) - 0.5 * torch.pow(dist, 2)

    return dist


def save_obj(dict_name, obj):
    with open(dict_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(dict_name):
    with open(dict_name, 'rb') as f:
        return pickle.load(f)


def locate(count, i):
    total = 0
    for j in range(len(count)):
        total += count[j]
        if total > i:
            return j


def if_start(count, i):
    sum = 0
    for j in range(len(count)):
        sum += count[j]
        if i == 0 or i == sum:
            return True
    return False


def if_end(count, i):
    sum = 0
    for j in range(len(count)):
        sum += count[j]
        if i == sum - 1:
            return True
    return False


def crop(img_org, crop_top, crop_bottom, crop_left, crop_right, size):
    crop_img = img_org[crop_top:crop_bottom, crop_left:crop_right, :]  # h,w,3
    crop_img = crop_img.permute(2, 0, 1).unsqueeze(0)  # 1,3,h,w
    crop_img = F.interpolate(crop_img, size=size)# cv2.resize 的gpu版
    crop_img = crop_img - torch.mean(crop_img.float())
    crop_img = crop_img / torch.max(torch.abs(crop_img))
    return crop_img.squeeze(0)  # 3,h,w


def get_coordinate(det, shape):
    left, top, right, bottom = det
    left = max(0, round(left))
    right = min(shape[2], round(right))
    top = max(0, round(top))
    bottom = min(shape[1], round(bottom))

    w = right - left
    h = bottom - top
    #print('w=%d,h=%d'%(w,h))
    left = max(0, int(left - (h - w) / 2))
    right = min(shape[2], int(right + (h - w) / 2))

    return left, right, top, bottom


def get_coordinate2(det, frame_width, frame_height):
    left, top, right, bottom = det
    left = max(0, round(left))
    right = min(frame_width, round(right))
    top = max(0, round(top))
    bottom = min(frame_height, round(bottom))

    w = right - left
    h = bottom - top
    left = max(0, int(left - (h - w) / 2))
    right = min(frame_width, int(right + (h - w) / 2))

    return int(left), int(right), int(top), int(bottom)


def get_crop_coordinate(left, right, top, bottom, shape):
    w = right - left
    h = bottom - top
    crop_top = max(0, int(top - h / 2.5))
    crop_bottom = min(shape[1], int(bottom + h / 10))
    h = crop_bottom - crop_top
    crop_left = max(0, int(left - (h - w) / 2))
    crop_right = min(shape[2], int(right + (h - w) / 2))
    return crop_left, crop_right, crop_top, crop_bottom


def get_crop_coordinate2(left, right, top, bottom, frame_width, frame_height):
    w = right - left
    h = bottom - top
    crop_top = max(0, int(top - h / 5))
    crop_bottom = min(frame_height, int(bottom + h / 10))
    h = crop_bottom - crop_top
    crop_left = max(0, int(left - (h - w) / 2))
    crop_right = min(frame_width, int(right + (h - w) / 2))
    return int(crop_left), int(crop_right), int(crop_top), int(crop_bottom)


def write_json(img_save, json_people, output_json):
    base = "Base64"
    if img_save is not None:
        base64_str = cv2.imencode('.jpg', img_save)[1].tobytes()
        base64_str = base64.b64encode(base64_str)
        base = "data:image/jpg;base64," + str(base64_str)[2:-1]  # 写json的时候应该加前面那段
    json_dict1 = {"camera_id": 1, "frame_id": 2, "time": 3, "encoding_picture": base,
                  "person": json_people}
    json_dict2 = {"frame_info": json_dict1}
    with open(output_json, "w") as f:
        f.write(json.dumps(json_dict2))
    # print('%s has been written.' % output_json)


def get_pred(output_gender_tensor):
    softmax = nn.Softmax()
    gen_softmax = softmax(output_gender_tensor)
    output_gender_bool = torch.argmax(gen_softmax).item()
    gender_confidence = round(gen_softmax[output_gender_bool].item(), 4)
    return output_gender_bool, gender_confidence


if __name__ == '__main__':
    # dataset=MyDataset('test_imgs/')
    # dataloader = DataLoader(dataset, batch_size=4, num_workers=1, shuffle=True)
    # for step, data in enumerate(dataloader):
    #     img = data
    #     print(img.shape)

    count = [1, 0, 0, 4]
    print(count)
    sum = 0
    idx = -1
    for num in range(0, 5):
        print('i=', num)
        if num == sum:
            print('this is the start of a frame')
            idx += 1
            while count[idx] == 0:
                idx += 1
            sum += count[idx]
        if num == (sum - 1):
            print('this is the end of a frame')
            # idx += 1

        print()
