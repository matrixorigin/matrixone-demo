#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = "yolox_tiny-clothes"

        self.depth = 0.33
        self.width = 0.375

        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.enable_mixup = False

        # Define yourself dataset path
        self.data_dir = ""
        self.train_ann = "/home/gwx/YOLOX/datasets/ClothTrainCOCO.json"
        self.val_ann = "/home/gwx/YOLOX/datasets/ClothValCOCO.json"

        self.num_classes = 2

        self.max_epoch = 300
        self.data_num_workers = 8
        self.eval_interval = 1