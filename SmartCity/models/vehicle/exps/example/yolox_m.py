#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = "yolox_m-vehicle"

        self.input_size = (640, 640)
        self.random_size = (10, 20)
        self.test_size = (640, 640)
        self.enable_mixup = False

        # Define yourself dataset path
        self.data_dir = ""
        self.train_ann = "/home/gwx/YOLOX/datasets/VehicleTrainCOCO.json"
        self.val_ann = "/home/gwx/YOLOX/datasets/VehicleValCOCO.json"

        self.num_classes = 6

        self.max_epoch = 100
        self.data_num_workers = 8
        self.eval_interval = 1