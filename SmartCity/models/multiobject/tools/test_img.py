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
from preproc import preproc
from coco_classes import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from utils import write_json_data, make_parser, get_image_list
from predictor import Predictor

def image_demo(predictor, vis_folder, path, current_time, save_result):
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )

    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    judge_asset = 0
    judge_person = 0
    counter = 0

    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)

        ratio = img_info["ratio"]
        json_assets = []
        json_people = []
        people_bbox = []
        if outputs[0]!=None:
            object_detect_num=outputs[0].size()[0]
            for per in range(object_detect_num):
                bbox_get = outputs[0][per][0:4]
                bbox_get /= ratio
                if outputs[0][per][6].item() == 1:#检测到有椅子出现
                    judge_asset = 1
                    f_c = outputs[0][per][4].item() * outputs[0][per][5].item()
                    json_asset = {
                        "assets_category":"chair", "assets_x1":bbox_get[0].item(),"assets_y1":bbox_get[1].item(),"assets_x2":bbox_get[2].item(),"assets_y2":bbox_get[3].item(),"assets_confidence":f_c,
                    }
                    json_assets.append(json_asset)

                else:
                    judge_person = 1
                    f_c = outputs[0][per][4].item() * outputs[0][per][5].item()
                    bbox_temp = [bbox_get[0].item(),bbox_get[1].item(),bbox_get[2].item(),bbox_get[3].item(),f_c]
                    people_bbox.append(bbox_temp)#行人检测列表
        else:
            pass
        base = "Base64"
        base64_str = cv2.imencode('.jpg',result_image)[1].tobytes()
        base="data:image/jpg;base64,"+str(base64_str)[2:-1]

        '''
        第二级工服检测与安全帽检测
        '''

        for kkk in range(len(people_bbox)):#写入行人与工服检测反馈
            person_now = people_bbox[kkk]
            json_person = {#json输出，含安全帽与工服，暂时置为bool=false
                "person_x1":person_now[0],"person_y1":person_now[1],"person_x2":person_now[2],"person_y2":person_now[3],"person_confidence":person_now[4],
                "uniform":{
                    "uniform_state":False,
                    "uniform_confidence":0.0
                },
                "helmet":{
                    "helmet_state":False,
                    "helmet_confidence":0.0
                },
            }
            json_people.append(json_person)

        if save_result:
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        else:
            pass
        json_dict0 = {"person":json_people}
        json_dict1 = {"assets":json_assets}
        json_dict2 = {"camera_id":1,"frame_id":2,"time":3,"encoding_picture":base,"person":json_dict0,"environment":json_dict1}
        json_dict3={"frame_info":json_dict2}
        result_name="result_of_"+str(counter)+"th_picture"
        if (judge_person+judge_asset) > 0:
            write_json_data(json_dict3, file_name=os.path.join(save_folder, result_name))
            judge_person = 0
            judge_asset = 0
        else:
            pass
        counter = counter + 1
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

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
