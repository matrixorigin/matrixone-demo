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
from utils import write_json_data, make_parser, get_image_list
from predictor import Predictor

def locate_items(output, people_bbox, ratio_min, ratio_max, method):
    person_num = len(people_bbox)
    confidence_record = np.zeros(person_num)
    judge_person_wear = np.zeros(person_num)
    statues = np.zeros(person_num)
    if output != None:
        cls = output[:, 6]
        for i in range(len(cls)):
            if int(cls[i].item()) == (2 + 2 * method) or int(cls[i].item()) == (3 + 2 * method):
                items_bbox = output[i, 0:4].cpu().numpy()
                items_area = (items_bbox[2] - items_bbox[0]) * (items_bbox[3] - items_bbox[1])

                for j in range(person_num):
                    if judge_person_wear[j] == 1:
                        continue
                    else:
                        person_bbox = people_bbox[j]
                        person_area = abs(person_bbox[2] - person_bbox[0]) * abs(person_bbox[3] - person_bbox[1])

                        if (person_bbox[2] > items_bbox[0]) & (person_bbox[0] < items_bbox[2]) & (
                                person_bbox[3] > items_bbox[1]) & (person_bbox[1] < items_bbox[3]):
                            x1_max = max(items_bbox[0], person_bbox[0])
                            y1_max = max(items_bbox[1], person_bbox[1])
                            x2_min = min(items_bbox[2], person_bbox[2])
                            y2_min = min(items_bbox[3], person_bbox[3])
                            intersection = (x2_min - x1_max) * (y2_min - y1_max)

                            rat = intersection / items_area

                            if (rat > ratio_min) & (rat < ratio_max):#匹配成功
                                confidence_record[j] = output[i][4].item() * output[i][5].item()
                                judge_person_wear[j] = 1
                                if int(cls[i].item()) == (2 + 2 * method):
                                    statues[j] = 1
                                else:
                                    statues[j] = 2
                            else:
                                continue
                        else:
                            continue
            else:
                continue
    return [confidence_record, statues]

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
        outputs, img_info = predictor.multi_inference(image_name, 2)
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

                elif outputs[0][per][6].item() == 0:#检测到有人出现
                    judge_person = 1
                    f_c = outputs[0][per][4].item() * outputs[0][per][5].item()
                    bbox_temp = [bbox_get[0].item(),bbox_get[1].item(),bbox_get[2].item(),bbox_get[3].item(),f_c]
                    people_bbox.append(bbox_temp)#行人检测列表
                    
                else:
                    pass
        else:
            pass
        
        clothratio_th = [0.7, 2.0]
        hatratio_th = [0.7, 2.0]
        [cloth_confidence,cloth_sta] = locate_items(outputs[0], people_bbox, clothratio_th[0], clothratio_th[1], 0)
        [hat_confidence,hat_sta] = locate_items(outputs[0], people_bbox, hatratio_th[0], hatratio_th[1], 1)

        base = " "

        for kkk in range(len(people_bbox)):#写入行人与工服检测反馈
            person_now = people_bbox[kkk]

            uni_status = 0#未知
            hat_status = 0#未知

            if cloth_confidence[kkk] > 0.3:
                if cloth_sta[kkk] == 1:
                    uni_status = 1#穿了
                elif cloth_sta[kkk] == 2:
                    uni_status = 2#没穿

            if hat_confidence[kkk] > 0.3:
                if hat_sta[kkk] == 1:
                    hat_status = 1#穿了
                elif hat_sta[kkk] == 2:
                    hat_status = 2#没穿

            json_person = {#json输出，含安全帽与工服
                "person_x1":('%.2f' % person_now[0]),"person_y1":('%.2f' % person_now[1]),"person_x2":('%.2f' % person_now[2]),"person_y2":('%.2f' % person_now[3]),"person_confidence":('%.4f' % person_now[4]),
                "uniform":{
                    "uniform_state":uni_status,
                    "uniform_confidence":('%.4f' % cloth_confidence[kkk])
                },
                "helmet":{
                    "helmet_state":hat_status,
                    "helmet_confidence":('%.4f' % hat_confidence[kkk])
                },
                "dangerous_action":{#时序动作，对于目标检测时如何记录，需要进一步确认，因此暂时置0
                    "start_time":0,
                    "end_time":0,
                    "dangerous_confidence":0.0
                }
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

class concate_model(nn.Module):
    #def __init__(self, model1, model2):
    def __init__(self, args):
        super().__init__()

        multi_model_name = "./yolox_voc_s_p_c.py"
        clothes_model_neme = "./yolox_tiny-clothes.py"
        hat_model_neme = "./yolox_tiny-hat.py"

        multi_exp = get_exp(multi_model_name, None)
        clothes_exp = get_exp(clothes_model_neme, None)
        hat_exp = get_exp(hat_model_neme, None)

        self.multi_model = multi_exp.get_model()
        self.clothes_model = clothes_exp.get_model()
        self.hat_model = hat_exp.get_model()

        ckpt1 = torch.load("./best_first_step_detect.pth", map_location="cpu")
        ckpt2 = torch.load("./best_clothes_ckpt.pth.tar", map_location="cpu")
        ckpt3 = torch.load("./best_hat_ckpt.pth.tar", map_location="cpu")

        self.multi_model.load_state_dict(ckpt1["model"])
        self.clothes_model.load_state_dict(ckpt2["model"])
        self.hat_model.load_state_dict(ckpt3["model"])

        #self.model1 = model1
        #self.model2 = model2
        if args.fuse:
            logger.info("\tFusing model...")
            self.multi_model = fuse_model(self.multi_model)
            self.clothes_model = fuse_model(self.clothes_model)
            self.hat_model = fuse_model(self.hat_model)

    def forward(self, x, method):
        if method == 1:
            out_multi = self.multi_model(x)
            return out_multi

        elif method == 2 :
            out_multi = self.multi_model(x)
            out_clothes = self.clothes_model(x)
            out3_hat = self.hat_model(x)
            #out = torch.cat((out1, out2), 1)
            #print(out1.shape)
            #print(out2.shape)
            return [out_multi, out_clothes, out3_hat]

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

    model = concate_model(args)
    #logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

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
        #model.load_state_dict(ckpt["model"])
        #logger.info("loaded checkpoint done.")

    predictor = Predictor(model, exp, COCO_CLASSES, args.device)
    current_time = time.localtime()
    image_demo(predictor, vis_folder, args.path, current_time, args.save_result)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
