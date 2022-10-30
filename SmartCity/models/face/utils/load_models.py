import torch
import cv2
import os

import numpy as np
import warnings

from ..YOLOX.yolox.data.data_augment import preproc
from ..YOLOX.yolox.data.datasets import COCO_CLASSES
from ..YOLOX.yolox.exp import get_exp
from ..YOLOX.yolox.utils import fuse_model, get_model_info, postprocess, vis
from ..YOLOX.tools.demo import Predictor
from ..backbone.model_irse import IR_50
from ..classifier import classifier
from .make_parser import make_parser111

warnings.filterwarnings('ignore')


def load_face_models(exp_file, ckpt, BACKBONE_RESUME_ROOT, classifier_root):
    # torch.multiprocessing.set_start_method('spawn')
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化YOLOX
    args = make_parser111(exp_file, ckpt).parse_args()
    exp = get_exp(args.exp_file, args.name)
    model = exp.get_model()
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    ckpt_file = args.ckpt
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    predictor = Predictor(model, exp, COCO_CLASSES, None, args.device)
    print('YOLO predictor initialized!')

    # 人脸识别特征提取模型
    INPUT_SIZE = [112, 112]
    BACKBONE = IR_50(INPUT_SIZE)
    if torch.cuda.is_available():
        BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    else:
        BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT,map_location=torch.device('cpu')))
    BACKBONE = BACKBONE.to(DEVICE)
    BACKBONE = BACKBONE.eval()
    print('BACKBONE loaded!')

    # 分类模型
    CLASSIFIER = classifier()
    if torch.cuda.is_available():
        CLASSIFIER.load_state_dict(torch.load(classifier_root))
    else:
        CLASSIFIER.load_state_dict(torch.load(classifier_root,map_location=torch.device('cpu')))
    CLASSIFIER = CLASSIFIER.to(DEVICE)
    CLASSIFIER = CLASSIFIER.eval()
    print(classifier_root, ' loaded!')

    return predictor, BACKBONE, CLASSIFIER
