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
import torch.nn as nn
from preproc import preproc
from coco_classes import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from utils import write_json_data, make_parser
from predictor import Predictor

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)

    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    
    interval=5

    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), (fps/interval), (int(width), int(height))
    )
    counter = 0#单位：帧
    well_counter = 0#单位：个
    cooldown_time = 5#单位：秒
    last_detect = -5*fps#单位：帧
    judge = 0

    while True:
        ret_val, frame = cap.read()
        counter = counter + 1
        if counter % interval == 0:
            # 扩展维度，转tensor
            #frame = np.expand_dims(frame, axis=0)
            #frame = torch.from_numpy(frame)

            if ret_val:

                outputs, img_info = predictor.inference(frame)
                result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
                ratio = img_info["ratio"]

                if outputs[0]!=None:
                    well_detect_num=outputs[0].size()[0]
                    for per in range(well_detect_num):
                        bbox_get = outputs[0][per][0:4]
                        bbox_get /= ratio
                        json_manholes = []
                        if outputs[0][per][6].item() == 0 or outputs[0][per][6].item() == 2:#检测到有偏移的井盖
                            if (counter-last_detect) >= cooldown_time * fps:
                                judge = 1
                                last_detect = counter
                                f_c = outputs[0][per][4].item() * outputs[0][per][5].item()
                                base = "Base64"
                                base64_str = cv2.imencode('.jpg',result_frame)[1].tobytes()
                                base="data:image/jpg;base64,"+str(base64_str)[2:-1]

                                json_manhole = {
                                    "manhole_cover_x1":bbox_get[0].item(),"manhole_cover_y1":bbox_get[1].item(),"manhole_cover_x2":bbox_get[2].item(),"manhole_cover_y2":bbox_get[3].item(),"manhole_cover_confidence":f_c,
                                }
                                json_manholes.append(json_manhole)
                    if judge == 1:
                        json_dict1 = {"manhole_cover":json_manholes}
                        json_dict2 = {"camera_id":1,"frame_id":counter,"time":3,"encoding_picture":" ","environment":json_dict1}
                        json_dict3={"frame_info":json_dict2}
                        result_name="result"+str(well_counter)
                        well_counter = well_counter + 1
                        write_json_data(json_dict3, file_name=os.path.join(save_folder, result_name))
                        judge = 0
                if args.save_result:
                    vid_writer.write(result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break
        else:
            continue

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
    imageflow_demo(predictor, vis_folder, current_time, args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
