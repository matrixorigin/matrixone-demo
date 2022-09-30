#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger
import numpy as np
import cv2

import torch
import torch.nn.functional as F

from yolox.data.data_augment import preproc
from .yolox.data.datasets import Vehicle_CLASSES
from yolox.exp import get_exp
from .yolox.utils import fuse_model, get_model_info, postprocess, vis, _COLORS
from .LPRNet.model import build_lprnet
from .LPRNet.load_data import CHARS
import argparse
import os
import time
import pymysql

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
LPRpretrained_model_load = True
LPRNet_weight_file = "./model/vehicle/weights/Final_LPRNet_model.pth"

def make_parser111(path):
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "-demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default=path, help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="./model/vehicle/yolox_m_V2.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default="./model/vehicle/best_ckpt.pth.tar", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.6, type=float, help="test conf")
    parser.add_argument("--nms", default=0.4, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        LPRNet,
        exp,
        cls_names=Vehicle_CLASSES,
        LP_names = CHARS,
        trt_file=None,
        decoder=None,
        device="cpu",
    ):
        self.model = model
        self.LPRNet = LPRNet
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.output_json = {}

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None


        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
                #print(outputs.shape)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))

        #outputs = outputs[0]
        #if not outputs is None:
        #    outputs = outputs.cpu()
        return outputs, img_info

    def LPRecognition(self, output, img_info):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return 0

        #bboxes = output[:, 0:4].numpy().copy()
        bboxes = output[:, 0:4].cpu().numpy().copy()
        cls = output[:, 6]
        #print(cls)
        bboxes = bboxes / ratio
        self.pred_LP = []

        for i in range(len(bboxes)):
            if int(cls[i].item()) == 5:
                box = bboxes[i]
                # img_cutting = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
                # img_cut = np.ascontiguousarray(img_cutting, dtype=np.float32)
                # img_cut = torch.from_numpy(img_cut)
                # if self.device == 'gpu':
                #     img_cut = img_cut.cuda()
                # img_cut = (img_cut - 127.5) * 0.0078125
                # img_cut = img_cut.permute(2, 0, 1).unsqueeze(0)
                # img_cut = F.interpolate(img_cut, size=[24, 94])

                img_cut = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
                img_cut = cv2.resize(img_cut, (94, 24))
                img_cut = (img_cut - 127.5) * 0.0078125
                img_cut = np.transpose(img_cut, (2, 0, 1))
                img_cut = np.ascontiguousarray(img_cut, dtype=np.float32)
                img_cut = torch.from_numpy(img_cut).unsqueeze(0)

                if self.device == 'gpu':
                    img_cut = img_cut.cuda()

                rec_output = self.LPRNet(img_cut)
                rec_output = rec_output.cpu().detach().numpy()
                rec_labels = list()
                plate = {}
                for i in range(rec_output.shape[0]):
                    rec_output = rec_output[i, :, :]
                    rec_labels = list()
                    for j in range(rec_output.shape[1]):
                        rec_labels.append(np.argmax(rec_output[:, j], axis=0))

                    no_repeat_blank_label = list()
                    rec_c = rec_labels[0]
                    if rec_c != len(CHARS) - 1:
                        no_repeat_blank_label.append(rec_c)
                    for c in rec_labels:  # dropout repeate label and blank label
                        if (rec_c == c) or (c == len(CHARS) - 1):
                            if c == len(CHARS) - 1:
                                rec_c = c
                            continue
                        no_repeat_blank_label.append(c)
                        rec_c = c
                    rec_labels.append(no_repeat_blank_label)

                #pred_LP = [CHARS[x] for x in no_repeat_blank_label]
                self.pred_LP.append([CHARS[x] for x in no_repeat_blank_label])
                print(self.pred_LP)
                self.locate_LPs_car(output)
        return self.pred_LP

    def locate_LPs_car(self, output):
        self.LP_car_info = []

        if output != None:
            cls = output[:, 6]
            cat_num = 0
            for i in range(len(cls)):
                if int(cls[i].item()) == 5:
                    LP_bbox = output[i, 0:4].numpy()
                    LP_area = (LP_bbox[2] - LP_bbox[0]) * (LP_bbox[3] - LP_bbox[1])

                    for j in range(len(cls)):
                        if int(cls[j].item()) != 5:
                            vehicle_bbox = output[j, 0:4].numpy()
                            vehicle_area = (vehicle_bbox[2] - vehicle_bbox[0]) * (vehicle_bbox[3] - vehicle_bbox[1])

                            if (vehicle_bbox[2] > LP_bbox[0]) & (vehicle_bbox[0] < LP_bbox[2]) & (
                                    vehicle_bbox[3] > LP_bbox[1]) & (vehicle_bbox[1] < LP_bbox[3]):
                                x1_max = max(LP_bbox[0], vehicle_bbox[0])
                                y1_max = max(LP_bbox[1], vehicle_bbox[1])
                                x2_min = min(LP_bbox[2], vehicle_bbox[2])
                                y2_min = min(LP_bbox[3], vehicle_bbox[3])
                                intersection = (x2_min - x1_max) * (y2_min - y1_max)

                                rat = intersection / LP_area

                                if (rat > 0.99) & (rat < 1.01):
                                    str_info = f"vehicle id: {j} - {cls[j].item()}, area: {vehicle_area}; LP id {i} - {cls[i].item()}, area: {LP_area}; intersection: {intersection}; ratio: {rat}"
                                    try:
                                        print(cat_num)
                                        print(self.pred_LP[cat_num])
                                        concate_info = {
                                            "vehicel_num": j,
                                            "License_num": i,
                                            "License_Plate": self.pred_LP[cat_num]

                                        }
                                        cat_num = cat_num + 1
                                        self.LP_car_info.append(concate_info)

                                    except:
                                        cat_num = cat_num - 1
                                        concate_info = {
                                            "vehicel_num": j,
                                            "License_num": i,
                                            "License_Plate": self.pred_LP[cat_num]
                                        }
                                        self.LP_car_info.append(concate_info)
                        else:
                            continue
                else:
                    continue
        #print(self.LP_car_info)

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        #print(output)
        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

    def visual_cut_img(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        for i in range(len(bboxes)):
            box = bboxes[i]
            img_cut = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            img_cut = cv2.resize(img_cut, (384,128))
            img_blur = cv2.Canny(img_cut,100,200,3)
            cv2.imwrite(f"test_{i+1}.jpg", img_cut)

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)

        return vis_res

    def visual_LPs(self, output, LPs, img_info,db, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        width = img_info["width"]
        height = img_info["height"]
        cal_para = max(width, height)
        if output is None:
            return img

        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res, _ = self.vis_LPs(img, bboxes, scores, cls, LPs, width,db, cls_conf, self.cls_names)
        return vis_res

    def vis_LPs(self, img, boxes, scores, cls_ids, license_plates, cal_para,db, conf=0.5, class_names=None):
        LP_list = []
        vehicle_list = []
        num_LP = 0
        num_veh = 0
        for i in range(len(boxes)):
            if int(cls_ids[i]) != 5:
                box = boxes[i]
                cls_id = int(cls_ids[i])
                score = scores[i]
                if score < conf:
                    continue
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])

                output_veh = {  # 车辆
                    "veh_x1": box[0].cpu().numpy().astype(float),
                    "veh_y1": box[1].cpu().numpy().astype(float),
                    "veh_x2": box[2].cpu().numpy().astype(float),
                    "veh_y2": box[3].cpu().numpy().astype(float),
                    "veh_confidence": score.cpu().numpy().astype(float),
                    "plate": {''},
                    "type": class_names[cls_id],
                }

                if not len(self.pred_LP) == 0:
                    if num_veh < len(self.pred_LP):
                        vehicel_num = self.LP_car_info[num_veh]["vehicel_num"]
                        if vehicel_num == i:
                            LP_info = self.LP_car_info[num_veh]["License_Plate"]
                            License_num = self.LP_car_info[num_veh]["License_num"]
                            #print(License_num)
                            LP_score = scores[i]

                            output_veh["plate"] = {
                                "plate_number": ''.join(LP_info),
                                "plate_confidence": LP_score.numpy().astype(float),
                                }
                            num_veh = num_veh + 1


                    vehicle_list.append(output_veh)

                color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
                text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
                txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX

                txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

                txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
                cv2.rectangle(
                    img,
                    (x0, y0 + 1),
                    (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                    txt_bk_color,
                    -1
                )
                cv2.putText(img, text, (x0, y0 + txt_size[1]), font, cal_para * 0.00025, txt_color,
                            thickness=int(cal_para * 0.0005))

            else:
                box = boxes[i]
                cls_id = int(cls_ids[i])
                score = scores[i]
                if score < conf:
                    continue
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])

                License_Plate = license_plates[num_LP].copy()
                num_LP = num_LP + 1

                # ouput to json
                t0 = time.time()

                output_LP = {  # 车牌
                    "plate_x1": box[0].numpy().astype(float),
                    "plate_y1": box[1].numpy().astype(float),
                    "plate_x2": box[2].numpy().astype(float),
                    "plate_y2": box[3].numpy().astype(float),
                    "plate_number": ''.join(License_Plate),
                    "plate_confidence": score.numpy().astype(float),
                }

                t1 = time.time()
                print(f"LP Json: {t1 - t0}s")
                LP_list.append(output_LP)

                color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
                text = '{}-{:.1f}%'.format(class_names[cls_id], score * 100)
                License_Plate[0] = 'X'
                LPtext = ''.join(License_Plate)
                text = text + "  " + "License: " + LPtext
                txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(text, font, cal_para * 0.0003, int(cal_para * 0.0003))[0]
                # LPtext_size = cv2.getTextSize(text, font, cal_para*0.0003, int(cal_para*0.0003))[0]
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

                txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
                cv2.rectangle(
                    img,
                    (x0, y0 + 1),
                    (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                    txt_bk_color,
                    -1
                )
                cv2.putText(img, text, (x0, y0 + txt_size[1]), font, cal_para * 0.00025, txt_color,
                            thickness=int(cal_para * 0.0005))
        frame_info = {
            "camera_id": 1,
            "frame_id": 0,
            "time": time.strftime('%y-%m-%d-%H:%M:%S', time.localtime()),
            "vehicle": vehicle_list,
            "license_plate": LP_list,
        }
        cursor = db.cursor()
        base64_str = cv2.imencode('.jpg', img)[1].tobytes()
        # 这里的vehicle,license_plate是个列表，可以在表格中加项目继续解析
        # encoding picture数据过长被省略
        cursor.execute('INSERT INTO vehicle(camera_id,frame_id,`time`,raw,vehicle,license_plate) values(%s,%s,%s,%s,%s,%s)',
                            (frame_info["camera_id"],frame_info["frame_id"],frame_info["time"],str(base64_str),"".join(frame_info["vehicle"]),"".join(frame_info["license_plate"])))
        db.commit() # 务必commit，否则不会修改数据库
        self.output_json = {"frame_info": frame_info}
        print(self.output_json)

        return img, self.output_json

    def judge_warning(self, boundary, area_threshold, ratio_threshold, veh_list):
        bound_x1 = boundary[0]
        bound_y1 = boundary[1]

        bound_x2 = boundary[2]
        bound_y2 = boundary[3]

        veh_lib = self.output_json["frame_info"]["vehicle"]

        warning_list = []
        warning_veh_num = 0
        for i in range(len(veh_lib)):

            veh_x1 = veh_lib[i]["veh_x1"]
            veh_y1 = veh_lib[i]["veh_y1"]
            veh_x2 = veh_lib[i]["veh_x2"]
            veh_y2 = veh_lib[i]["veh_y2"]
            veh_type = veh_lib[i]["type"]
            area = (veh_x2 - veh_x1) * (veh_y2 - veh_y1)

            if not ((area >= area_threshold[0]) & (area <= area_threshold[1]) | (veh_type in veh_list)):
                continue

            else:
                if (bound_x2 > veh_x1) & (bound_x1 < veh_x2) & (
                        bound_y2 > veh_y1) & (bound_y1 < veh_y2):

                    x1_max = max(bound_x1, veh_x1)
                    y1_max = max(bound_y1, veh_y1)
                    x2_min = min(bound_x2, veh_x2)
                    y2_min = min(bound_y2, veh_y2)

                    intersection = (x2_min - x1_max) * (y2_min - y1_max)
                    ratio = intersection / area

                    if ratio >= ratio_threshold:
                        warning_list.append(veh_lib)
                        warning_veh_num = warning_veh_num + 1

                    info = f"there have {warning_veh_num} vehicls in the warning area"
                    print(info)
                    print(warning_list)



def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    # Open a MatrixOne connection
    db = pymysql.connect(host='127.0.0.1',
                    port=6001,
                    user='dump',
                    password='111',
                    database='park')  # 数据库名称，暂定park
    # Create db schema(create tables)
    cursor = db.cursor()
    for image_name in files:
        time0 = time.time()
        outputs, img_info = predictor.inference(image_name)
        time1 = time.time()
        LPs = predictor.LPRecognition(outputs, img_info)
        time2 = time.time()
        result_image = predictor.visual_LPs(outputs, LPs, img_info,db, predictor.confthre,db)
        time3 = time.time()

        boundary = [100, 100, 1000,1000]
        area_threshold = [20, 5000]
        ratio_threshold = 0.8
        veh_list = ["car"]
        predictor.judge_warning(boundary, area_threshold, ratio_threshold, veh_list)
        # os._exit(0)
        print(f"LP detect: {time1-time0}s")
        print(f"LP Recognition: {time2-time1}s")
        print(f"LP Visual: {time3 - time2}s")
        #result_image = predictor.visual_cut_img(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

def image_process(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs, img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


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
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    # Open a MatrixOne connection
    db = pymysql.connect(host='127.0.0.1',
                    port=6001,
                    user='dump',
                    password='111',
                    database='park')  # 数据库名称，暂定park
    # Create db schema(create tables)
    cursor = db.cursor()
    sql='create table if not exists vehicle(\
            camera_id int(4) NOT NULL,\
            frame_id int(10) NOT NULL,\
            `time` varchar(20) NOT NULL,\
            raw blob,\
            vehicle varchar(5000),\
            license_plate varchar(5000));'
    cursor.execute(sql)
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            #outputs, img_info = predictor.inference(frame)
            #result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            
            time0 = time.time()
            outputs, img_info = predictor.inference(frame)
            time1 = time.time()
            LPs = predictor.LPRecognition(outputs[0], img_info)
            time2 = time.time()
            result_image = predictor.visual_LPs(outputs[0], LPs, img_info,db, predictor.confthre)
            time3 = time.time()
            print(f"LP detect: {time1-time0}s")
            print(f"LP Recognition: {time2-time1}s")
            print(f"LP Visual: {time3 - time2}s")
            
            #if args.save_result:
            #    vid_writer.write(result_frame)
            vid_writer.write(result_image)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    #if args.save_result:
    #    vis_folder = os.path.join(file_name, "vis_res")
    #    os.makedirs(vis_folder, exist_ok=True)
    vis_folder = os.path.join(file_name, "vis_res")
    os.makedirs(vis_folder, exist_ok=True)
    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    model.eval()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    LPRNet = build_lprnet(lpr_max_len=8, class_num=len(CHARS))

    if args.device == "gpu":
        model.cuda()
        LPRNet.cuda()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
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

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    # load pretrained model
    if LPRpretrained_model_load:
        LPRNet.load_state_dict(torch.load(LPRNet_weight_file))
        print("LPRNet load pretrained model successful!")

    predictor = Predictor(model, LPRNet, exp, Vehicle_CLASSES, CHARS, trt_file, decoder, args.device)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)

def vehicle_model_infer(path):
    args = make_parser111(path).parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
