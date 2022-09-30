import argparse
import os
import time,json
import numpy as np
import base64
from loguru import logger

import cv2

import torch

from .preproc import preproc
from .coco_classes import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        device="cpu",
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
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
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def multi_inference(self, img, method):
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
            if method == 1:
                outputs = self.model(img, method)
                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

            elif method == 2:
                outputs = self.model(img, method)
                outputs_temp = []
                for i in range(len(outputs)):
                    outputs[i] = postprocess(outputs[i], self.num_classes, self.confthre, self.nmsthre)
                    if outputs[i][0] != None:
                        outputs[i][0][:, 6] = outputs[i][0][:, 6] + i * 2
                        outputs_temp.append(outputs[i][0])

                if len(outputs_temp) == 0:
                    outputs = [None]
                else:
                    for i in range(len(outputs_temp)):
                        if i == 0:
                            outputs = outputs_temp[i]
                        else:
                            outputs = torch.cat((outputs, outputs_temp[i]), 0)
                    outputs = [outputs]
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res