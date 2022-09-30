#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
import json
import numpy as np
import base64
from loguru import logger
import cv2
import torch
import torch.nn as nn
from .tools.preproc import preproc
from .tools.coco_classes import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from .tools.utils import write_json_data, make_parser111
from .tools.predictor import Predictor
import pymysql


def test_multiobject(frame, predictor, counter, assets_counter, person_counter, last_assets_detect, last_person_detect,iscuda,db):
    interval = 5
    assets_cooldown_time = 0  # 单位：秒
    person_cooldown_time = 0  # 单位：秒
    fps=30
    if counter % interval == 0:
        outputs, img_info = predictor.multi_inference(frame, 2)
        result_frame = predictor.visual(
            outputs[0], img_info, predictor.confthre)
        ratio = img_info["ratio"]

        json_assets = []
        json_people = []
        people_bbox = []
        judge_asset = 0
        judge_person = 0
        if outputs[0] != None:
            object_detect_num = outputs[0].size()[0]
            for per in range(object_detect_num):
                bbox_get = outputs[0][per][0:4]
                bbox_get /= ratio
                if outputs[0][per][6].item() == 1:  # 检测到有椅子出现
                    if (counter - last_assets_detect) > assets_cooldown_time * fps:
                        assets_counter = assets_counter + 1
                        judge_asset = 1
                        f_c = outputs[0][per][4].item(
                        ) * outputs[0][per][5].item()
                        json_asset = {
                            "assets_category": "chair", "assets_x1": ('%.2f' % bbox_get[0].item()), "assets_y1": ('%.2f' % bbox_get[1].item()), "assets_x2": ('%.2f' % bbox_get[2].item()), "assets_y2": ('%.2f' % bbox_get[3].item()), "assets_confidence": ('%.4f' % f_c),
                        }
                        json_assets.append(json_asset)
                    else:
                        pass

                elif outputs[0][per][6].item() == 0:  # 检测到有人出现
                    if (counter - last_person_detect) > person_cooldown_time * fps:
                        person_counter = person_counter + 1
                        judge_person = 1
                        f_c = outputs[0][per][4].item(
                        ) * outputs[0][per][5].item()
                        bbox_temp = [bbox_get[0].item(), bbox_get[1].item(
                        ), bbox_get[2].item(), bbox_get[3].item(), f_c]
                        people_bbox.append(bbox_temp)  # 行人检测列表
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
        [cloth_confidence, cloth_sta] = locate_items(
            outputs[0], people_bbox, clothratio_th[0], clothratio_th[1], 0)
        [hat_confidence, hat_sta] = locate_items(
            outputs[0], people_bbox, hatratio_th[0], hatratio_th[1], 1)

        base = " "

        for kkk in range(len(people_bbox)):  # 写入行人与工服检测反馈
            person_now = people_bbox[kkk]

            uni_status = 0  # 未知
            hat_status = 0  # 未知

            if cloth_confidence[kkk] > 0.3:
                if cloth_sta[kkk] == 1:
                    uni_status = 1  # 穿了
                elif cloth_sta[kkk] == 2:
                    uni_status = 2  # 没穿

            if hat_confidence[kkk] > 0.3:
                if hat_sta[kkk] == 1:
                    hat_status = 1  # 穿了
                elif hat_sta[kkk] == 2:
                    hat_status = 2  # 没穿

            json_person = {  # json输出，含安全帽与工服
                "person_x1": ('%.2f' % person_now[0]), "person_y1": ('%.2f' % person_now[1]), "person_x2": ('%.2f' % person_now[2]), "person_y2": ('%.2f' % person_now[3]), "person_confidence": ('%.4f' % person_now[4]),
                "uniform": {
                    "uniform_state": uni_status,
                    "uniform_confidence": ('%.4f' % cloth_confidence[kkk])
                },
                "helmet": {
                    "helmet_state": hat_status,
                    "helmet_confidence": ('%.4f' % hat_confidence[kkk])
                },
                "dangerous_action": {  # 时序动作，对于目标检测时如何记录，需要进一步确认，因此暂时置0
                    "start_time": 0,
                    "end_time": 0,
                    "dangerous_confidence": 0.0
                }
            }
            json_people.append(json_person)

        if (judge_person+judge_asset) > 0:  # json写入
            json_dict1 = {"assets": json_assets}
            json_dict2 = {"camera_id": 1, "frame_id": counter, "time": counter/fps,
                "encoding_picture": base, "person": json_people, "environment": json_dict1}
            json_dict3 = {"frame_info": json_dict2}
            result_name = "result_with_total_" + \
                str(assets_counter) + "th_assets_and_" + \
                    str(person_counter) + "th_person"
            cursor = db.cursor()
            # 这里的person是个字典，可以在表格中加项目继续解析
            # encoding picture数据过长被省略
            byte_data = cv2.imencode('.jpg', result_frame)[1].tobytes()
            base64_str = base64.b64encode(byte_data).decode("ascii") #转换为base64
            base="data:image/jpg;base64,"+str(base64_str)
            cursor.execute('INSERT INTO multiobject(camera_id,frame_id,`time`,raw,person,environment) values(%s,%s,%s,%s,%s,%s)',
                            (json_dict2["camera_id"], json_dict2["frame_id"], json_dict2["time"], str(base64_str), str(json_dict2["person"]), str(json_dict2["environment"])))
            db.commit()  # 务必commit，否则不会修改数据库

            judge_person = 0
            judge_asset = 0
        else:
            pass
    return assets_counter, person_counter, last_assets_detect, last_person_detect


def load_multiobject_model(exp_file,exp_dir, iscuda):
    exp = get_exp(exp_file, None)
    exp.test_conf = 0.6
    exp.nmsthre = 0.4
    model = concate_model(exp_dir)
    model.eval()
    if iscuda:
        multiobject_predictor = Predictor(model, exp, COCO_CLASSES, 'gpu')
    else:
        multiobject_predictor = Predictor(model, exp, COCO_CLASSES, 'cpu')
    return multiobject_predictor


class concate_model(nn.Module):
    #def __init__(self, model1, model2):
    def __init__(self, exp_dir):
        super().__init__()

        #multi_model_name = "/home/zjr/multiple_detect/cloth_chair_detect_first_v1_test/yolox_voc_s_p_c.py"
        #clothes_model_neme = "/home/zjr/multiple_detect/cloth_chair_detect_first_v1_test/yolox_tiny-clothes.py"
        #hat_model_neme = "/home/zjr/multiple_detect/cloth_chair_detect_first_v1_test/yolox_tiny-hat.py"
        multi_model_name = exp_dir + "/multiobject/yolox_voc_s_p_c.py"
        clothes_model_neme = exp_dir + "/multiobject/yolox_tiny-clothes.py"
        hat_model_neme = exp_dir + "/multiobject/yolox_tiny-hat.py"

        multi_exp = get_exp(multi_model_name, None)
        clothes_exp = get_exp(clothes_model_neme, None)
        hat_exp = get_exp(hat_model_neme, None)

        self.multi_model = multi_exp.get_model()
        self.clothes_model = clothes_exp.get_model()
        self.hat_model = hat_exp.get_model()

        #ckpt1 = torch.load("/home/zjr/multiple_detect/cloth_chair_detect_first_v1_test/best_first_step_detect.pth", map_location="cpu")
        #ckpt2 = torch.load("/home/zjr/multiple_detect/cloth_chair_detect_first_v1_test/best_clothes_ckpt.pth.tar", map_location="cpu")
        #ckpt3 = torch.load("/home/zjr/multiple_detect/cloth_chair_detect_first_v1_test/best_hat_ckpt.pth.tar", map_location="cpu")
        ckpt1 = torch.load(
            exp_dir + "/multiobject/best_first_step_detect.pth", map_location="cpu")
        ckpt2 = torch.load(
            exp_dir + "/multiobject/best_clothes_ckpt.pth.tar", map_location="cpu")
        ckpt3 = torch.load(
            exp_dir + "/multiobject/best_hat_ckpt.pth.tar", map_location="cpu")

        self.multi_model.load_state_dict(ckpt1["model"])
        self.clothes_model.load_state_dict(ckpt2["model"])
        self.hat_model.load_state_dict(ckpt3["model"])

    def forward(self, x, method):
        if method == 1:
            out_multi = self.multi_model(x)
            return out_multi

        elif method == 2:
            out_multi = self.multi_model(x)
            out_clothes = self.clothes_model(x)
            out3_hat = self.hat_model(x)
            return [out_multi, out_clothes, out3_hat]



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
                items_area = (items_bbox[2] - items_bbox[0]) * \
                    (items_bbox[3] - items_bbox[1])

                for j in range(person_num):
                    if judge_person_wear[j] == 1:
                        continue
                    else:
                        person_bbox = people_bbox[j]
                        person_area = abs(
                            person_bbox[2] - person_bbox[0]) * abs(person_bbox[3] - person_bbox[1])

                        if (person_bbox[2] > items_bbox[0]) & (person_bbox[0] < items_bbox[2]) & (
                                person_bbox[3] > items_bbox[1]) & (person_bbox[1] < items_bbox[3]):
                            x1_max = max(items_bbox[0], person_bbox[0])
                            y1_max = max(items_bbox[1], person_bbox[1])
                            x2_min = min(items_bbox[2], person_bbox[2])
                            y2_min = min(items_bbox[3], person_bbox[3])
                            intersection = (x2_min - x1_max) * \
                                (y2_min - y1_max)

                            rat = intersection / items_area

                            if (rat > ratio_min) & (rat < ratio_max):  # 匹配成功
                                confidence_record[j] = output[i][4].item(
                                ) * output[i][5].item()
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
