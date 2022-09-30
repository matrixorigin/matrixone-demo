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

from .tools.preproc import preproc
from .tools.coco_classes import COCO_CLASSES
from .yolox.exp import get_exp
from .yolox.utils import fuse_model, get_model_info, postprocess, vis
from .tools.utils import write_json_data, make_parser111, cen_2_cen_pair, cen_pair_2_cen, coor_2_cen, dis
from .tools.predictor import Predictor
import pymysql

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]



def load_falldown_model(exp_file,iscuda):
    exp=get_exp(exp_file,None)
    model = exp.get_model()
    model.eval()
    if iscuda:
        falling_predictor = Predictor(model, exp, COCO_CLASSES, 'gpu')
    else:
        falling_predictor = Predictor(model, exp, COCO_CLASSES, 'cpu')
    return falling_predictor

def test_falldown(frame,predictor,counter,fall_counter,f_c,bbo_cen,fall_judge,fall_up_bound,detect_fall,bbo_all_num,iscuda,db):
    fps=30
    interval=5
    fall_time_judge = 3
    clear_time_judge = 30
    lasting_fall_judge = 0.3
    if counter%interval == 0:
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

                                    byte_data = cv2.imencode('.jpg', result_frame)[1].tobytes()
                                    base64_str = base64.b64encode(byte_data).decode("ascii") #转换为base64
                                    base="data:image/jpg;base64,"+str(base64_str)
                                    json_people=[]
                                    json_fall_person = {
                                        "person_x1":bbox_get[0].item(),"person_y1":bbox_get[1].item(),"person_x2":bbox_get[2].item(),"person_y2":bbox_get[3].item(),"person_confidence":per_fc,
                                        "fall":
                                            {"start_time": counter/fps, "fall_confidence":per_fc }
                                    }
                                    json_people.append(json_fall_person)
                                    json_dict1 = {"camera_id": 1, "frame_id": counter, "time": 3, "raw": str(base64_str), "person": json_people}
                                    json_dict2={"frame_info":json_dict1}
                                    result_name = "result" + str(fall_counter)
                                    
                                    json_db_data = {"camera_id": 1, "frame_id": counter, "time": time.time(), "raw": byte_data, "person": json_people}

                                    # write_db_data(json_db_data,db)
                                    # write_json_data(json_dict2, file_name=os.path.join(save_folder, result_name))
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
    return fall_counter,f_c,bbo_cen,fall_judge,fall_up_bound,detect_fall,bbo_all_num

def write_db_data(write_info,db):
    cursor = db.cursor()
    # 这里的person是个字典，可以在表格中加项目继续解析
    # encoding picture数据过长被省略
    cursor.execute('INSERT INTO falldown(camera_id,frame_id,`time`,raw,person) values(%s,%s,%s,%s,%s)',
                        (write_info["camera_id"],write_info["frame_id"],write_info["time"],write_info["raw"],str(write_info["person"])))
    db.commit() # 务必commit，否则不会修改数据库