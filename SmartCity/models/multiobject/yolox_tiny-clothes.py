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