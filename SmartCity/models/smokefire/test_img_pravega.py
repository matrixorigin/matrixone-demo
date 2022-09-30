import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torch.nn import init
from tqdm import tqdm
import os
import numpy as np
import sys
import cv2
import json
from .dataset import MyDataset
from .model import classifier
import time
import pymysql
import base64


def get_pred(output_gender_tensor):
    softmax = nn.Softmax(dim=0)
    gen_softmax = softmax(output_gender_tensor)
    output_gender_bool = torch.argmax(gen_softmax).item()
    gender_confidence = round(gen_softmax[1].item(), 4)
    return output_gender_bool, gender_confidence


def img_preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')
    img = img.transpose(2, 0, 1)
    img = img - np.mean(img)
    img = img / np.max(np.abs(img))

    return img


def load_smokefire_model(classfier_root,iscuda):
    DEVICE = torch.device("cuda:0" if iscuda else "cpu")
    CLASSIFIER = classifier()
    CLASSIFIER = CLASSIFIER.to(DEVICE)
    CLASSIFIER_RESUME_ROOT = classfier_root
    if iscuda:
        CLASSIFIER.load_state_dict(torch.load(CLASSIFIER_RESUME_ROOT), strict=False)
    else:
        CLASSIFIER.load_state_dict(torch.load(CLASSIFIER_RESUME_ROOT,map_location=torch.device('cpu')), strict=False)
        CLASSIFIER.eval()
    return CLASSIFIER


def test_smokefire(img_org, CLASSIFIER,counter,iscuda,db):
    with torch.no_grad():
        img = img_preprocess(img_org)
        img = torch.from_numpy(img).unsqueeze(0)
        if iscuda:
            DEVICE = torch.device("cuda:0")
            inputs = img.cuda().to(DEVICE)
        else:
            inputs = img
        fire_out, smoke_out = CLASSIFIER(inputs)
        # print('%.4f' % (end_time - start_time))

        for i in range(img.shape[0]):
            fire_bool, fire_conf = get_pred(fire_out[i])
            smoke_bool, smoke_conf = get_pred(smoke_out[i])

            if fire_bool or smoke_bool:
                json_people = []
                if fire_bool:
                    json_person = {"fire": fire_conf}
                    json_people.append(json_person)
                if smoke_bool:
                    json_smoke = {"smoke": smoke_conf}
                    json_people.append(json_smoke)
                print(json_people)
                byte_data = cv2.imencode('.jpg', result_frame)[1].tobytes()
                base64_str = base64.b64encode(byte_data).decode("ascii") #转换为base64
                base="data:image/jpg;base64,"+str(base64_str)
                write_db_data(json_people, frame_count, str(base64_str), db)
                # write_json(json_people, save_dir + str(frame_count).zfill(5) + '.json', frame_count)
             
                
def write_db_data(json_people, frame_count, base64_str, db):
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    json_dict1 = {"camera_id": 1, "frame_id": frame_count, "time": time_str, "encoding_picture": base,
                  "environment": json_people}
    cursor = db.cursor()
    # 这里的person是个字典，可以在表格中加项目继续解析
    # encoding picture数据过长被省略
    cursor.execute('INSERT INTO smokefire(camera_id,frame_id,`time`,raw,environment) values(%s,%s,%s,%s,%s)',
                   (json_dict1["camera_id"], json_dict1["frame_id"], json_dict1["time"], str(base64_str), str(json_dict1["environment"])))
    db.commit() # 务必commit，否则不会修改数据库