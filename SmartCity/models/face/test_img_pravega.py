from email.mime import image
import pymysql
from .utils.load_models import load_face_models
from .utils.preproc import preproc
from .utils.make_parser import make_parser111
from .utils.get_gallery import get_gallery_embedding
from .utils.utils import locate, if_start, if_end, compute_cos_similarity, crop, get_coordinate2, write_json, get_pred
import torch
import cv2
import os
import numpy as np
import torch.nn.functional as F
import cv2
import time
import warnings
import json
import base64

warnings.filterwarnings('ignore')


def test_face(img_org, predictor,counter,BACKBONE, classifier,iscuda,db):
    t0 = time.time()
    td, tc, tr = 0, 0, 0
    frame_height=img_org.shape[0]
    frame_width=img_org.shape[1]
    img_640, r = preproc(img_org)

    # 扩展维度，转tensor
    img_640 = np.expand_dims(img_640, axis=0)
    img_org_np = np.expand_dims(img_org, axis=0)
    img_640 = torch.from_numpy(img_640)
    img_org = torch.from_numpy(img_org_np)

    tds = time.time()

    # yolo人脸检测
    if iscuda:
        outputs = predictor.inference(img_640.cuda())
    else:
        outputs = predictor.inference(img_640)

    count = [0] * img_org.shape[0]
    det = []
    try:
        for i in range(img_org.shape[0]):
            for j in range(outputs[i].shape[0]):
                score = outputs[i][j][4] * \
                    outputs[i][j][5]  # 第i张图中的第j个人脸
                if score > 0.6:
                    d = outputs[i][j][0:4] / r
                    det.append(np.array(d.detach().cpu()))
                    count[i] += 1

    except:
        # 这个batch没检测到人脸
        pass

    tde = time.time()
    td += tde - tds

    if det != []:
        batch_renshu = len(det)
        face_224 = torch.zeros((batch_renshu, 3, 224, 224))
        for i in range(batch_renshu):
            frame_num = locate(count, i)
            left, right, top, bottom = get_coordinate2(
                det[i], frame_width, frame_height)
            face_224[i] = crop(img_org[frame_num],
                                top, bottom, left, right, [224, 224])
        face_112 = F.interpolate(face_224, size=[112, 112])

        # 人脸识别
        #trs = time.time()
        #emb_batch = BACKBONE(face_112.cuda()).cuda()
        #cos_similarity = compute_cos_similarity(emb_batch.double(), gallery_ebd.double())
        #max_cos_similarity, idx = torch.max(cos_similarity, dim=1)
        #tre = time.time()
        #tr += tre - trs

        # 分类
        tcs = time.time()
        if iscuda:
            output_glass_tensor, output_gender_tensor, output_mask_tensor = classifier(
                face_224.cuda())
        else:
            output_glass_tensor, output_gender_tensor, output_mask_tensor = classifier(
                face_224)
        tce = time.time()
        tc += tce - tcs

        for i in range(batch_renshu):

            output_gender_bool, gender_confidence = get_pred(
                output_gender_tensor[i])
            output_glass_bool, glass_confidence = get_pred(
                output_glass_tensor[i])
            output_mask_bool, mask_confidence = get_pred(
                output_mask_tensor[i])

            output_gender_str = "Male" if output_gender_bool == 1 else "Female"
            output_glass_str = "Glasses" if output_glass_bool == 1 else "No glasses"
            output_mask_str = "Masked" if output_mask_bool == 1 else "Unmasked"

            #pred_name = gallery_label[idx[i]]
            #name_confidence = round(1 - max_cos_similarity[i].item(), 4)

            left, right, top, bottom = get_coordinate2(
                det[i], frame_width, frame_height)

            # 现在没有年龄模型，先占个坑
            output_age_out = 22
            age_confidence = 0.9

            json_person = {"face_x1": left, "face_y1": top, "face_x2": right, "face_y2": bottom,
                            "sex": {"sex_category": output_gender_str, "sex_confidence": gender_confidence},
                            "age": {"number": output_age_out, "age_confidence": age_confidence},
                            "mask": {"mask_state": output_mask_bool, "mask_confidence": mask_confidence},
                            "glasses": {"glasses_state": output_glass_bool,
                                        "glasses_confidence": glass_confidence}
                            }

            # 每帧的第一个人
            if if_start(count, i):
                json_people = []
                frame_num = locate(count, i)
                img_save = img_org_np[frame_num]

            # 每个人
            json_people.append(json_person)
            print(json_people)
            
            red = (0, 0, 255)
            green = (0, 255, 0)
            pink = (147, 20, 255)
            blue = (255, 191, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            x1 = (left, top)
            x2 = (right, bottom)
            zihao = frame_width / 800
            color = green if output_mask_bool else red
            cv2.rectangle(img_save, x1, x2, color, 3)

            string = output_glass_str
            color = blue if output_gender_bool else pink
            cv2.putText(img_save, string, (left, top),
                        font, zihao, color, round(zihao * 3))


            byte_data = cv2.imencode('.jpg', img_save)[1].tobytes()
            base64_str = base64.b64encode(byte_data).decode("ascii") #转换为base64
            base="data:image/jpg;base64,"+str(base64_str)

            # sql insert
            cursor = db.cursor()
            # 这里的person是个字典，可以在表格中加项目继续解析
            # encoding picture数据过长被省略
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cursor.execute('INSERT INTO face(camera_id,frame_id,`time`,raw,face_x1,face_y1,face_x2,face_y2,sex_category,sex_confidence,age_number,age_confidence,mask_state,mask_confidence,glasses_state,glasses_confidence) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                           (1, counter, time_str, +str(base64_str), json_person["face_x1"], json_person["face_y1"], json_person["face_x2"], json_person["face_y2"], json_person["sex"]["sex_category"], json_person["sex"]["sex_confidence"], json_person["age"]["number"], json_person["age"]["age_confidence"], json_person["mask"]["mask_state"], json_person["mask"]["mask_confidence"], json_person["glasses"]["glasses_state"], json_person["glasses"]["glasses_confidence"]))
            db.commit()  # 务必commit，否则不会修改数据库

                        #if if_end(count, i):
                        #writer.write(img_save)  # 写入视频

            # for i in range(len(count)):
            #     if count[i] == 0:
            #         img_save = img_org[i]
            #         json_people = []
            #         output_file = "test_json_out/%s" % path[i].split('/')[-1].replace('jpg', 'json')
            #         write_json(img_save=None, json_people=json_people, output_json=output_file)

        t1 = time.time()