import torch
import torch.nn as nn
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import os


class load_model(nn.Module):
    #def __init__(self, model1, model2):
    def __init__(self):
        super().__init__()
        exp1_name = "exps/example/yolox_tiny-hat.py"
        exp2_name = "exps/example/yolox_tiny-clothes.py"
        exp1 = get_exp(exp1_name, None)
        exp2 = get_exp(exp2_name, None)
        self.model1 = exp1.get_model()
        self.model2 = exp2.get_model()
        ckpt1 = torch.load("./YOLOX_outputs/yolox_tiny-hat/best_ckpt.pth.tar", map_location="cpu")
        ckpt2 = torch.load("./YOLOX_outputs/yolox_tiny-clothes/best_ckpt.pth.tar", map_location="cpu")
        self.model1.load_state_dict(ckpt1["model"])
        self.model2.load_state_dict(ckpt2["model"])
        #self.model1 = model1
        #self.model2 = model2

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        #out = torch.cat((out1, out2), 1)
        #print(out1.shape)
        #print(out2.shape)
        return out1, out2
