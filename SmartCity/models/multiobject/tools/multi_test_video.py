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
    assets_counter = 0#单位：（累计）个
    person_counter = 0#单位：（累计）个
    assets_cooldown_time = 0#单位：秒
    person_cooldown_time = 0#单位：秒
    last_assets_detect = -2 * fps#单位：帧
    last_person_detect = -1 * fps#单位：帧
    judge = 0

    x_min_dan = 100
    x_max_dan = 1000
    y_min_dan = 300
    y_max_dan = 600

    while True:
        ret_val, frame = cap.read()
        counter = counter + 1
        if counter % interval == 0:
            # 扩展维度，转tensor
            #frame = np.expand_dims(frame, axis=0)
            #frame = torch.from_numpy(frame)
            if ret_val:
                outputs, img_info = predictor.multi_inference(frame, 2)
                result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
                ratio = img_info["ratio"]

                json_assets = []
                json_people = []
                people_bbox = []

                if outputs[0]!=None:
                    object_detect_num=outputs[0].size()[0]
                    judge_asset = 0
                    judge_person = 0
                    for per in range(object_detect_num):
                        bbox_get = outputs[0][per][0:4]
                        bbox_get /= ratio
                        if outputs[0][per][6].item() == 1:#检测到有椅子出现
                            if (counter - last_assets_detect) > assets_cooldown_time * fps:
                                assets_counter = assets_counter + 1
                                judge_asset = 1
                                f_c = outputs[0][per][4].item() * outputs[0][per][5].item()
                                json_asset = {
                                    "assets_category":"chair", "assets_x1":('%.2f' % bbox_get[0].item()),"assets_y1":('%.2f' % bbox_get[1].item()),"assets_x2":('%.2f' % bbox_get[2].item()),"assets_y2":('%.2f' % bbox_get[3].item()),"assets_confidence":('%.4f' % f_c),
                                }
                                json_assets.append(json_asset)
                            else:
                                pass

                        elif outputs[0][per][6].item() == 0:#检测到有人出现
                            if (counter - last_person_detect) > person_cooldown_time * fps:
                                person_counter = person_counter + 1
                                judge_person = 1
                                f_c = outputs[0][per][4].item() * outputs[0][per][5].item()
                                bbox_temp = [bbox_get[0].item(),bbox_get[1].item(),bbox_get[2].item(),bbox_get[3].item(),f_c]
                                people_bbox.append(bbox_temp)#行人检测列表
                            else:
                                pass
                        else:
                            pass
                    if judge_asset == 1:
                        last_assets_detect = counter
                    else:
                        pass
                    if judge_person == 1:
                        last_person_detect = counter
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

                if args.save_result:
                    vid_writer.write(result_frame)
                else:
                    pass

                if (judge_person+judge_asset) > 0:#json写入
                    json_dict1 = {"assets":json_assets}
                    json_dict2 = {"camera_id":1,"frame_id":counter,"time":counter/fps,"encoding_picture":base,"person":json_people,"environment":json_dict1}
                    json_dict3={"frame_info":json_dict2}
                    result_name="result_with_total_"+str(assets_counter)+"th_assets_and_"+str(person_counter)+"th_person"
                    write_json_data(json_dict3, file_name=os.path.join(save_folder, result_name))
                    judge_person = 0
                    judge_asset = 0
                else:
                    pass

                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
                else:
                    pass
            else:
                break
        else:
            continue

class concate_model(nn.Module):
    #def __init__(self, model1, model2):
    def __init__(self, args):
        super().__init__()

        #multi_model_name = "/home/zjr/multiple_detect/cloth_chair_detect_first_v1_test/yolox_voc_s_p_c.py"
        #clothes_model_neme = "/home/zjr/multiple_detect/cloth_chair_detect_first_v1_test/yolox_tiny-clothes.py"
        #hat_model_neme = "/home/zjr/multiple_detect/cloth_chair_detect_first_v1_test/yolox_tiny-hat.py"
        multi_model_name = "./yolox_voc_s_p_c.py"
        clothes_model_neme = "./yolox_tiny-clothes.py"
        hat_model_neme = "./yolox_tiny-hat.py"

        multi_exp = get_exp(multi_model_name, None)
        clothes_exp = get_exp(clothes_model_neme, None)
        hat_exp = get_exp(hat_model_neme, None)

        self.multi_model = multi_exp.get_model()
        self.clothes_model = clothes_exp.get_model()
        self.hat_model = hat_exp.get_model()

        #ckpt1 = torch.load("/home/zjr/multiple_detect/cloth_chair_detect_first_v1_test/best_first_step_detect.pth", map_location="cpu")
        #ckpt2 = torch.load("/home/zjr/multiple_detect/cloth_chair_detect_first_v1_test/best_clothes_ckpt.pth.tar", map_location="cpu")
        #ckpt3 = torch.load("/home/zjr/multiple_detect/cloth_chair_detect_first_v1_test/best_hat_ckpt.pth.tar", map_location="cpu")
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

    predictor = Predictor(model, exp, COCO_CLASSES, args.device)
    current_time = time.localtime()
    imageflow_demo(predictor, vis_folder, current_time, args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
