#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright 2022 Matrix Origin
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http: // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time,json
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


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def load_manhole_model(exp_file,iscuda):
    exp=get_exp(exp_file,None)
    model = exp.get_model()
    model.eval()
    if iscuda:
        manhole_predictor = Predictor(model, exp, COCO_CLASSES, 'gpu')
    else:
        manhole_predictor = Predictor(model, exp, COCO_CLASSES, 'cpu')
    return manhole_predictor


def test_manhole(frame,predictor,counter,well_counter,last_detect,judge,iscuda,db):
    
    # # Open a MatrixOne connection
    # db = pymysql.connect(host='127.0.0.1',
    #              port=6001,
    #              user='dump',
    #              password='111',
    #              database='park')  # 数据库名称，暂定park
    # cursor = db.cursor()
    # sql='create table if not exists well(\
    #         camera_id int(4) NOT NULL,\
    #         frame_id int(10) NOT NULL,\
    #         `time` int(10) NOT NULL,\
    #         `raw` blob,\
    #         manhole_cover varchar(5000));'
    # cursor.execute(sql)
    fps = 30
    interval=5
    cooldown_time = 5#单位：秒
    if counter % interval == 0:
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
                        byte_data = cv2.imencode('.jpg', result_frame)[1].tobytes()
                        # base64_str = base64.b64encode(byte_data).decode("ascii") #转换为base64
                        # base="data:image/jpg;base64,"+str(base64_str)

                        json_manhole = {
                            "manhole_cover_x1":bbox_get[0].item(),"manhole_cover_y1":bbox_get[1].item(),"manhole_cover_x2":bbox_get[2].item(),"manhole_cover_y2":bbox_get[3].item(),"manhole_cover_confidence":f_c,
                        }
                        json_manholes.append(json_manhole)
            if judge == 1:
                json_dict1 = {"manhole_cover":json_manholes}
                json_dict2 = {"camera_id": 1, "frame_id": counter,
                              "time": 3, "raw": byte_data, "environment": json_dict1}
                json_dict3={"frame_info":json_dict2}
                result_name="result"+str(well_counter)
                well_counter = well_counter + 1
                cursor = db.cursor()
                # 这里的person是个字典，可以在表格中加项目继续解析
                cursor.execute('INSERT INTO well(camera_id,frame_id,`time`,raw,manhole_cover) values(%s,%s,%s,%s,%s)',
                               (json_dict2["camera_id"], json_dict2["frame_id"], json_dict2["time"], json_dict2["raw"], str(json_dict2["environment"]["manhole_cover"])))
                db.commit()  # 务必commit，否则不会修改数据库
                
                # write_json_data(json_dict3, file_name=os.path.join(save_folder, result_name))
                judge = 0
    return well_counter,last_detect,judge

