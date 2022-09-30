import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torch.nn import init
import os
import numpy as np
import random
import sys

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()

        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')

        # for p in self.backbone.parameters():
        #     p.requires_grad = False

        self.head_glass = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(0.2),
            nn.Linear(1280, 2)
        )

        self.head_mask = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(0.2),
            nn.Linear(1280, 2)
        )

        self.head_gender = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(0.2),
            nn.Linear(1280, 2)
        )

    def forward(self, x):
        x = x.float()
        features = self.backbone.extract_features(x)# batchsize, 7, 7, 1280
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1)) #batchsize, 1, 1, 1280

        gender = self.head_gender(features)
        glass = self.head_glass(features)
        mask = self.head_mask(features)

        return glass, gender, mask