#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        #self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.exp_name = "yolox-tiny"

        # Define yourself dataset path
        self.data_dir = "datasets/VOCdevkit"
        self.train_ann = "/data1/littlesc/FR_data/WIDERFACE/annotations/WIDERFaceTrainCOCO.json"
        self.val_ann = "/data1/littlesc/FR_data/WIDERFACE/annotations/WIDERFaceValCOCO.json"

        self.num_classes = 1

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1
