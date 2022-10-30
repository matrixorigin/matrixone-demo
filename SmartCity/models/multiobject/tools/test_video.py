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
    assets_counter = 0#单位：（累计）个
    person_counter = 0#单位：（累计）个
    assets_cooldown_time = 2#单位：秒
    person_cooldown_time = 1#单位：秒
    last_assets_detect = -2 * fps#单位：帧
    last_person_detect = -1 * fps#单位：帧
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

                json_assets = []
                json_people = []
                people_bbox = []

                if outputs[0]!=None:
                    object_detect_num=outputs[0].size()[0]
                    for per in range(object_detect_num):
                        bbox_get = outputs[0][per][0:4]
                        bbox_get /= ratio
                        if outputs[0][per][6].item() == 1:#检测到有椅子出现
                            if (counter - last_assets_detect) > assets_cooldown_time * fps:
                                assets_counter = assets_counter + 1
                                judge_asset = 1
                                last_assets_detect = counter
                                f_c = outputs[0][per][4].item() * outputs[0][per][5].item()
                                json_asset = {
                                    "assets_category":"chair", "assets_x1":bbox_get[0].item(),"assets_y1":bbox_get[1].item(),"assets_x2":bbox_get[2].item(),"assets_y2":bbox_get[3].item(),"assets_confidence":f_c,
                                }
                                json_assets.append(json_asset)
                            else:
                                pass

                        else:
                            if (counter - last_person_detect) > person_cooldown_time * fps:
                                person_counter = person_counter + 1
                                judge_person = 1
                                last_person_detect = counter
                                f_c = outputs[0][per][4].item() * outputs[0][per][5].item()
                                bbox_temp = [bbox_get[0].item(),bbox_get[1].item(),bbox_get[2].item(),bbox_get[3].item(),f_c]
                                people_bbox.append(bbox_temp)#行人检测列表
                            else:
                                pass
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
                    '''
                    行人危险区域监测
                    '''
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
                        "dangerous_action":{#时序动作，对于目标检测时如何记录
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
                    json_dict0 = {"person":json_people}
                    json_dict1 = {"assets":json_assets}
                    json_dict2 = {"camera_id":1,"frame_id":counter,"time":counter/fps,"encoding_picture":base,"person":json_dict0,"environment":json_dict1}
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

        multi_model_name = "/home/gwx/YOLOX/weights/multi_model/yolox_voc_s_multi.py"
        clothes_model_neme = "/home/gwx/YOLOX/exps/example/yolox_tiny-clothes.py"
        hat_model_neme = "/home/gwx/YOLOX/exps/example/yolox_tiny-hat.py"

        multi_exp = get_exp(multi_model_name, None)
        clothes_exp = get_exp(clothes_model_neme, None)
        hat_exp = get_exp(hat_model_neme, None)

        self.multi_model = multi_exp.get_model()
        self.clothes_model = clothes_exp.get_model()
        self.hat_model = hat_exp.get_model()

        ckpt1 = torch.load("/home/gwx/YOLOX/weights/multi_model/best_ckpt.pth", map_location="cpu")
        ckpt2 = torch.load("/home/gwx/YOLOX/YOLOX_outputs/yolox_tiny-clothes/best_ckpt.pth.tar", map_location="cpu")
        ckpt3 = torch.load("/home/gwx/YOLOX/YOLOX_outputs/yolox_tiny-hat/best_ckpt.pth.tar", map_location="cpu")

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

    ######################################################模型替换###################################
    #model = exp.get_model()
    model = concate_model()

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

    ########################################下面三行注释掉#################################################
    # if args.fuse:
    #     logger.info("\tFusing model...")
    #     model = fuse_model(model)

    predictor = Predictor(model, exp, COCO_CLASSES, args.device)
    current_time = time.localtime()
    imageflow_demo(predictor, vis_folder, current_time, args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
