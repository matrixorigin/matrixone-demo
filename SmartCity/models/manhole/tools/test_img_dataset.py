#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time,json
import numpy as np
import base64
from loguru import logger
import cv2
import torch
from preproc import preproc_torch
from coco_classes import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from utils import write_json_data, make_parser, get_image_list
from predictor import Predictor
from dataset import MyDataset
from torch.utils.data import DataLoader

def image_demo(predictor, vis_folder, path, current_time, save_result):
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )

    # if os.path.isdir(path):
    #     files = get_image_list(path)
    # else:
    #     files = [path]
    # files.sort()

    judge = 0
    counter = 0

    #for image_name in files:
    root_dir='testfiles/try/'
    dataset = MyDataset(root_dir)
    BATCH_SIZE = 2
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
    print('len(dataset):', len(dataset))
    print('batch_size:', BATCH_SIZE)

    for step, data in enumerate(dataloader):

        img_org, image_name = data

        #print(image_name)
        #print('img_org.shape',img_org.shape)

        image, ratio = preproc_torch(img_org.cuda())
        #print('image after preprocessing:',image.shape)

        outputs = predictor.inference(image)

        for i in range(img_org.shape[0]):

            #print('img_org[i].numpy()',img_org[i].numpy().shape)

            img_org_i_np = img_org[i].numpy()
            result_image = predictor.visual(ratio, outputs[i], img_org_i_np, predictor.confthre)

            #ratio = r[0]

            if outputs[0]!=None:
                well_detect_num=outputs[0].size()[0]
                for per in range(well_detect_num):
                    bbox_get = outputs[0][per][0:4]
                    bbox_get /= ratio
                    json_manholes = []
                    if outputs[0][per][6].item() == 0 or outputs[0][per][6].item() == 2:#检测到有偏移的井盖
                        judge = 1
                        f_c = outputs[0][per][4].item() * outputs[0][per][5].item()
                        base = "Base64"
                        base64_str = cv2.imencode('.jpg',result_image)[1].tobytes()
                        base="data:image/jpg;base64,"+str(base64_str)[2:-1]

                        json_manhole = {
                            "manhole_cover_x1":bbox_get[0].item(),"manhole_cover_y1":bbox_get[1].item(),"manhole_cover_x2":bbox_get[2].item(),"manhole_cover_y2":bbox_get[3].item(),"manhole_cover_confidence":f_c,
                        }
                        json_manholes.append(json_manhole)
                json_dict1 = {"manhole_cover":json_manholes}
                json_dict2 = {"camera_id":1,"frame_id":2,"time":3,"encoding_picture":" ","environment":json_dict1}
                json_dict3={"frame_info":json_dict2}
                result_name="result_of_"+str(counter)+"th_picture"
                counter = counter + 1

                if save_result:
                    os.makedirs(save_folder, exist_ok=True)
                    save_file_name = os.path.join(save_folder, os.path.basename(image_name[i]))
                    logger.info("Saving detection result in {}".format(save_file_name))
                    cv2.imwrite(save_file_name, result_image)
                    if judge == 1:
                        write_json_data(json_dict3, file_name=os.path.join(save_folder, result_name))
                        judge = 0
                # ch = cv2.waitKey(0)
                # if ch == 27 or ch == ord("q") or ch == ord("Q"):
                #     break

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    predictor = Predictor(model, exp, COCO_CLASSES, args.device)
    current_time = time.localtime()
    image_demo(predictor, vis_folder, args.path, current_time, args.save_result)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
