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
from utils import write_json_data, make_parser, cen_2_cen_pair, cen_pair_2_cen, coor_2_cen, dis
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
    counter = 0
    fall_counter = 0
    f_c = np.zeros(20)
    bbo_cen = np.zeros(20)
    fall_judge = np.zeros(20)
    fall_up_bound = np.zeros(20)
    detect_fall = np.zeros(20)
    bbo_all_num = -1
    lasting_fall_judge = 0.3
    fall_time_judge = 3
    clear_time_judge = 30

    while True:
        ret_val, frame = cap.read()
        counter = counter + 1
        if counter%interval == 0:
            # 扩展维度，转tensor
            #frame = np.expand_dims(frame, axis=0)
            #frame = torch.from_numpy(frame)

            if ret_val:
                outputs, img_info = predictor.inference(frame)
                result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
                ratio = img_info["ratio"]
                for tt1 in range(20):#储存空间内部合并清洗位置相近框
                    for tt2 in range(tt1+1,20):
                        if dis(cen_2_cen_pair(bbo_cen[tt1]), cen_2_cen_pair(bbo_cen[tt2])) < dis([0,0],[100,100]):
                            f_c[tt2] = 0
                            bbo_cen[tt2] = 0
                            fall_judge[tt2] = 0
                            fall_up_bound[tt2] = 0
                            detect_fall[tt2] = counter
                temp_judge = 0
                last_last = max(detect_fall)
                if (counter - last_last) > round(clear_time_judge * fps):#当缓存超过一定时间限度时，进行自我清理
                    f_c = np.zeros(20)
                    bbo_cen = np.zeros(20)
                    fall_judge = np.zeros(20)
                    fall_up_bound = np.zeros(20)
                    detect_fall = np.ones(20) * counter
                for tt in range(20):#检查存储空间是否有缓存
                    if bbo_cen[tt] != 0:
                        temp_judge = 1
                if temp_judge ==0:
                    bbo_all_num = -1
                if outputs[0]!=None:
                    person_num=outputs[0].size()[0]
                    for per in range(person_num):
                        bbox_get = outputs[0][per][0:4]
                        bbox_get /= ratio
                        b_c_t = coor_2_cen(bbox_get[0].item(),bbox_get[2].item(),bbox_get[1].item(),bbox_get[3].item())
                        if outputs[0][per][6].item() == 0:#先看本帧摔倒的人与之前进行匹配
                            if bbo_all_num == -1:
                                bbo_all_num = 0
                                bbo_cen[bbo_all_num] = b_c_t
                                fall_up_bound[bbo_all_num] = counter
                                f_c[bbo_all_num] = outputs[0][per][4].item() * outputs[0][per][5].item()
                            else:
                                temp_dis = np.zeros(20)
                                for kk in range(20):
                                    temp_dis[kk] = dis(cen_2_cen_pair(b_c_t),cen_2_cen_pair(bbo_cen[kk]))
                                temp_dis=temp_dis.tolist()
                                min_dis_index = temp_dis.index(min(temp_dis))
                                min_dis = temp_dis[min_dis_index]
                                if min_dis < dis([0,0],[100,100]):#匹配到了之前的人
                                    bbo_cen[min_dis_index] = b_c_t
                                    if fall_judge[min_dis_index] == 0:#之前未输出的人
                                        if (counter-fall_up_bound[min_dis_index])< round(lasting_fall_judge*fps):#未通过判决
                                            f_c[min_dis_index] = f_c[min_dis_index]+ outputs[0][per][4].item() * outputs[0][per][5].item()
                                        else:#通过平滑判决
                                            f_c[min_dis_index] = f_c[min_dis_index]+ outputs[0][per][4].item() * outputs[0][per][5].item()
                                            fall_judge[min_dis_index] = 1
                                            fall_counter = fall_counter + 1
                                            detect_fall[min_dis_index] = counter
                                            per_fc = outputs[0][per][4].item() * outputs[0][per][5].item()
                                            base = "Base64"
                                            base64_str = cv2.imencode('.jpg',result_frame)[1].tobytes()
                                            base="data:image/jpg;base64,"+str(base64_str)[2:-1]
                                            json_people=[]
                                            json_fall_person = {
                                                "person_x1":bbox_get[0].item(),"person_y1":bbox_get[1].item(),"person_x2":bbox_get[2].item(),"person_y2":bbox_get[3].item(),"person_confidence":per_fc,
                                                "fall":
                                                    {"start_time": counter/fps, "fall_confidence":per_fc }
                                            }
                                            json_people.append(json_fall_person)
                                            json_dict1 = {"camera_id":1,"frame_id":counter,"time":3,"encoding_picture":base,"person":json_people}
                                            json_dict2={"frame_info":json_dict1}
                                            result_name="result"+str(fall_counter)
                                            write_json_data(json_dict2, file_name=os.path.join(save_folder, result_name))
                                    else:#之前已输出的人
                                        continue
                                else:#未匹配到摔倒记录的人
                                    for kk in range(20):#找到一个空位置插入
                                        if bbo_cen[kk] == 0:
                                            break
                                    bbo_cen[kk] = b_c_t
                                    fall_up_bound[kk] = counter
                                    f_c[kk] = outputs[0][per][4].item() * outputs[0][per][5].item()

                        else:#再看之前摔倒的人是否在本帧站起
                            temp_dis = np.zeros(20)
                            for kk in range(20):
                                temp_dis[kk] = dis(cen_2_cen_pair(b_c_t),cen_2_cen_pair(bbo_cen[kk]))
                            temp_dis=temp_dis.tolist()
                            min_dis_index = temp_dis.index(min(temp_dis))
                            min_dis = temp_dis[min_dis_index]
                            if min_dis < dis([0,0],[100,100]):#匹配到了之前摔倒的人
                                if detect_fall[min_dis_index] > 0:#并且该摔倒的人已经做过输出，也就是被判定为摔倒已经开始
                                    if (counter-detect_fall[min_dis_index]) > (fall_time_judge * fps):#距离判断摔倒的时长已经超过了fall_time_judge阈值
                                        f_c[min_dis_index] = 0
                                        bbo_cen[min_dis_index] = 0
                                        fall_judge[min_dis_index] = 0
                                        fall_up_bound[min_dis_index] = 0
                                        detect_fall[min_dis_index] = 0
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
