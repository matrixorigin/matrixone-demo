import numpy as np
from torch.nn import functional as F
import torch
import cv2
import os
import torch.nn as nn
import sys
from torch.utils.data import DataLoader, Dataset
from .dataset_img import MyDataset
import pickle
# import argparse
from .utils import save_obj, load_obj, get_coordinate, crop


def get_gallery_embedding(predictor, BACKBONE, mode='gen', root_dir='gallery/',
                          gallery_numpy='model/face/gallery_files/gallery.npy',
                          gallery_label_dir='model/face/gallery_files/gallery_label.npy',
                          gender_dict_dir='model/face/gallery_files/gender_dict.pkl',
                          age_dict_dir='model/face/gallery_files/age_dict.pkl'):
    assert mode in ['gen', 'load']
    if mode == 'gen':
        dataset = MyDataset(dir=root_dir)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=True)

        with torch.no_grad():
            gallery_label = []
            gallery_ebd = np.zeros((len(dataset), 512))
            for step, data in enumerate(dataloader):
                img_org, img_640, path, r = data
                img_640 = img_640.cuda()
                id = path[0].split('/')[-1].split('.')[0]

                outputs = predictor.inference(img_640)

                det = outputs[0][0][0:4].cpu() / r.cpu()
                left, right, top, bottom = get_coordinate(np.array(det.cpu()), img_org.shape)
                crop_img_tensor = crop(img_org[0], top, bottom, left, right, [112, 112]).unsqueeze(0)

                gallery_ebd[step] = BACKBONE(crop_img_tensor.cuda()).cpu().numpy()

                gallery_label.append(id)

        gender_dict = {"ck": 1,
                       "gwx": 1,
                       "ccy": 0,
                       "gzynpy": 0,
                       "yjw": 1,
                       "wgx": 1,
                       "shijie": 0,
                       "sc": 0,
                       "plf": 1,
                       "gzy": 1,
                       "wgxnpy": 0,
                       "yjwnpy": 0,
                       "wxx": 0,
                       "gjf": 1,
                       "spg": 1,
                       "lty": 1,
                       "zgq": 1,
                       "hjt": 1,
                       "gwxnpy": 0,
                       "yzp": 1}

        age_dict = {"ck": 22,
                    "gwx": 22,
                    "ccy": 23,
                    "gzynpy": 21,
                    "yjw": 22,
                    "wgx": 22,
                    "shijie": 38,
                    "sc": 23,
                    "plf": 22,
                    "gzy": 22,
                    "wgxnpy": 22,
                    "yjwnpy": 22,
                    "wxx": 22,
                    "gjf": 22,
                    "spg": 22,
                    "lty": 22,
                    "zgq": 22,
                    "hjt": 22,
                    "gwxnpy": 22,
                    "yzp": 22
                    }

        save_obj(gender_dict_dir, gender_dict)
        save_obj(age_dict_dir, age_dict)
        np.save(gallery_numpy, gallery_ebd)
        np.save(gallery_label_dir, np.array(gallery_label))
        print('Gallery generated! There are %d person in the database.' % len(gallery_label))

    else:
        gallery_ebd = np.load(gallery_numpy)
        gallery_label = np.load(gallery_label_dir).tolist()
        gender_dict = load_obj(gender_dict_dir)
        age_dict = load_obj(age_dict_dir)
        print('Gallery loaded! There are %d person in the database.' % len(gallery_label))

    gallery_ebd = torch.from_numpy(gallery_ebd).cuda()
    gallery_ebd = F.normalize(gallery_ebd, dim=1)

    return gallery_ebd, gallery_label, gender_dict, age_dict
