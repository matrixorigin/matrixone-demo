import base64
import torch
import cv2
import os, sys
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import cv2
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader, Dataset
from utils.utils import locate, if_start, if_end, compute_cos_similarity, crop, get_coordinate, \
    write_json, get_pred
# from utils.make_parser import make_parser111
from utils.get_gallery import get_gallery_embedding
# from face_models_tmp.utils.dataset_img import MyDataset
from utils.load_models import load_models
from classifier import classifier
import json




class Face_recognition(object):
    """docstring for Darknet"""

    def __init__(self):
        # 初始化YOLOX
        self.path = "./"
        self.exp_file = self.path + "YOLOX/exps/example/custom/yolox_tiny.py"
        self.ckpt = self.path + "YOLOX/YOLOX_outputs/yolox-tiny/best_ckpt.pth"

        # 人脸识别特征提取模型
        self.BACKBONE_RESUME_ROOT = self.path + 'trained_models/backbone_ir50_asia.pth'

        # # 分类模型
        self.classifier_root = self.path + 'trained_models/CLASSIFIER-B0-224-Epoch_170.pth'

        self.predictor, self.BACKBONE, self.classifier = load_models(self.exp_file, self.ckpt,
                                                                     self.BACKBONE_RESUME_ROOT, self.classifier_root)

        # 生成gallery
        self.gallery_ebd, self.gallery_label, self.gender_dict, self.age_dict = get_gallery_embedding(self.predictor,
                                                                                                      self.BACKBONE,
                                                                                                      mode='load')

    def preproc_torch(self, image, input_size=(416, 416), mean=torch.Tensor([0.406, 0.485, 0.456]),
                      std=torch.Tensor([0.225, 0.229, 0.224])):
        
        padded_img = torch.ones(image.shape[0], 3, input_size[0], input_size[1]) * 114.0  # 4, 3, 416(h), 416(w)
        r = min(input_size[0] / image.shape[1], input_size[1] / image.shape[2])  # min(416/1080,416/1920)=416/1920
        # 将 4,h,w,3 转成 4,3,h,w
        image = image.permute(0, 3, 1, 2)  # 4, 3, h, w
        resized_img = F.interpolate(image, size=(int(image.shape[2] * r), int(image.shape[3] * r)))  # 4, 3, h*r, w*r
        padded_img[:, :, :resized_img.shape[2], :resized_img.shape[3]] = resized_img  # 顶着左上角填充
        # BGR转RGB
        index = [2, 1, 0]
        padded_img = padded_img[:, index, :, :]
        padded_img /= 255.0
        if mean is not None:
            mean = mean.unsqueeze(-1).unsqueeze(-1)
            padded_img -= mean
        if std is not None:
            std = std.unsqueeze(-1).unsqueeze(-1)
            padded_img /= std
        return padded_img, r

    def detect(self, img_org):
        n = 0
        if 1:
            #t1 = time.time()
            #img_org = np.array(detect_imgs)
            #t5 = time.time()
            # print("---------列表变成array-------",t5-t1)
            #print(img_org.shape)
            #img_org = torch.from_numpy(img_org)
            # print(img_tensor.shape) # 1, 1080, 1920, 3
            img_640, r = self.preproc_torch(img_org.cuda())
            json_people_list = [0] * img_org.shape[0]
            # json_people_dic = {}
            json_people_dic = []
            #t6 = time.time()
            outputs = self.predictor.inference(img_640.cuda())
            #t3 = time.time()
            # print("---------人脸检测-------",outputs)
            # print('************ r1 ************',r1)
            count = [0] * img_org.shape[0]
            det = []
            for i in range(img_org.shape[0]):
                if outputs[i] != None:
                    for j in range(outputs[i].shape[0]):
                        score = outputs[i][j][4] * outputs[i][j][5]  # 第i张图中的第j个人脸
                        if score > 0.6:
                            d = outputs[i][j][0:4] / r
                            det.append(np.array(d.detach().cpu()))
                            count[i] += 1
                else:
                    pass
            # 整个batch都没有人脸
            if det == []:
                #print('未检测到人脸')
                for i in range(img_org.shape[0]):
                    json_dict1 = {"camera_id": i, "frame_id": i,
                                  "time": 2, "face_info": []}
                    # json_people_dic["frame%d" % i] = json_dict1
                    json_people_dic.append({"frame_info": json_dict1})
                json_people_list = []
                for i in range(img_org.shape[0]):
                    json_people_list.append([])
                # write_json_dir = "test_json_out/%d.json" % (batch + 1)
                # with open(write_json_dir, "w") as f:
                #    f.write(json.dumps(json_people_dic))
            else:
                # 这个batch有人脸
                batch_renshu = len(det)
                face_224 = torch.zeros((batch_renshu, 3, 224, 224))
                n += batch_renshu
                for i in range(batch_renshu):
                    frame_num = locate(count, i)
                    left, right, top, bottom = get_coordinate(det[i], img_org.shape)
                    face_224[i] = crop(img_org[frame_num], top, bottom, left, right, [224, 224])
                    # tmp=face_224[i].permute(1,2,0)
                    # tmp=np.array(tmp)
                    # cv2.imwrite('probe_crop/' + path[frame_num].split('/')[-1], tmp * 127 + 127)  # 加127之后正常显示
                #t7 = time.time()
                face_112 = F.interpolate(face_224, size=[112, 112])
                emb_batch = self.BACKBONE(face_112.cuda()).cuda()
                face_224 = face_224-torch.mean(face_224)
                face_224 = face_224/torch.max(torch.abs(face_224))
                output_glass_tensor, output_gender_tensor, output_mask_tensor = self.classifier(face_224.cuda())

                cos_similarity = compute_cos_similarity(emb_batch.double(), self.gallery_ebd.double())
                max_cos_similarity, idx = torch.max(cos_similarity, dim=1)
                #t4 = time.time()
                # print("---------分类时间-------",t4-t7)
                for i in range(len(det)):
                    output_gender_bool, gender_confidence = get_pred(output_gender_tensor[i])
                    output_glass_bool, glass_confidence = get_pred(output_glass_tensor[i])
                    output_mask_bool, mask_confidence = get_pred(output_mask_tensor[i])

                    output_gender_str = "Male" if output_gender_bool == 1 else "Female"
                    output_glass_str = "Glasses" if output_glass_bool == 1 else "No glasses"
                    output_mask_str = "Masked" if output_mask_bool == 1 else "Unmasked"

                    pred_name = self.gallery_label[idx[i]]
                    name_confidence = round(max_cos_similarity[i].item(), 4)
                    # print(name_confidence)

                    frame_num = locate(count, i)
                    left, right, top, bottom = get_coordinate(det[i], img_org.shape)
                    output_age_out = 22
                    age_confidence = 0.9
                    json_person = {"face_x1": left, "face_y1": top, "face_x2": right, "face_y2": bottom,
                                   "name": {"name_category": pred_name, "name_confidence": name_confidence},
                                   "sex": {"sex_category": output_gender_str, "sex_confidence": gender_confidence},
                                   "age": {"number": output_age_out, "age_confidence": age_confidence},
                                   "mask": {"mask_state": output_mask_bool, "mask_confidence": mask_confidence},
                                   "glasses": {"glasses_state": output_glass_bool,
                                               "glasses_confidence": glass_confidence}
                                   }
                    # print(json_person)
                    # 每帧的第一个人
                    if if_start(count, i):
                        json_people = []
                        # img_save = img_org[frame_num].cpu().numpy()
                    json_people.append(json_person)
                    """
                    # 每个人
                    red = (0, 0, 255)
                    green = (0, 255, 0)
                    pink = (147, 20, 255)
                    blue = (255, 191, 0)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    x1 = (left, top)
                    x2 = (right, bottom)
                    zihao = img_save.shape[1] / 800
                    color = green if output_mask_bool else red
                    cv2.rectangle(img_save, x1, x2, color, 2)

                    string = output_glass_str + ',' + pred_name
                    color = blue if output_gender_bool else pink
                    cv2.putText(img_save, string, (left, top), font, zihao, color, round(zihao * 3))
                    """
                    if if_end(count, i):
                        # save_path = 'test_imgs_out/' + path[frame_num].split('/')[-1]
                        # cv2.imwrite(save_path, img_save)

                        frame_num = locate(count, i)
                        json_people_list[frame_num] = json_people  # 每帧是一个元素

                for i in range(len(json_people_list)):
                    if json_people_list[i] == 0:  # 如果那一帧没有人脸
                        # json_dict1 = {"camera_id": i, "frame_id": i,
                        #               "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "face_info": []}
                        json_dict1 = {"camera_id": i, "frame_id": i,
                                      "time": 2, "face_info": []}
                    else:
                        json_dict1 = {"camera_id": i, "frame_id": i,
                                      "time": 2,
                                      "face_info": json_people_list[i]}

                    # json_people_dic["frame%d" % i] = json_dict1
                    json_people_dic.append({"frame_info": json_dict1})
                # write_json_dir = "test_json_out/%d.json" % (batch + 1)
                # with open(write_json_dir, "w") as f:
                #    f.write(json.dumps(json_people_dic))
                # print('----------json_people_dic-------',json_people_dic)
                #print('++++++++++',json_people_list)
            #t2 = time.time()
            #print("总时延%.4f" % (t2 - t1))
            return json_people_list, json_people_dic
            # batch += 1

 
    def get_coordinate2(self,det, frame_width, frame_height):
        left, top, right, bottom = det
        left = max(0, round(left))
        right = min(frame_width, round(right))
        top = max(0, round(top))
        bottom = min(frame_height, round(bottom))

        w = right - left
        h = bottom - top
        left = max(0, int(left - (h - w) / 2))
        right = min(frame_width, int(right + (h - w) / 2))
        return int(left), int(right), int(top), int(bottom)

if __name__ == '__main__':

    # import resource
    # resource.setrlimit(resource.RLIMIT_CPU, (100,-1))


    face = Face_recognition()
    #path = ['test_imgs/wgx_glass_masked.jpg']
    path = ['/data1/data_raw/cloth_raw/IMG_7527.MOV']
    batchsize=4
    count=[0]*batchsize
    interval =1
    cap = cv2.VideoCapture(path[0])
    while True:
        # path = ['test_imgs/wgx_glass_masked.jpg']
        time1 = time.time()
        _,cv_img =cap.read()
        cv_imgs = [cv_img]*batchsize
        frames = np.array(cv_imgs)
        res_tensor = torch.from_numpy(frames)
        #print(cv_imgs)
        # print(cv_imgs)
        if _!=False:

            face.detect(res_tensor)

            print('间隔 %.3f s'%(time.time()-time1))
            time1=time.time()
        if _==False:
             #break
            cap.release()