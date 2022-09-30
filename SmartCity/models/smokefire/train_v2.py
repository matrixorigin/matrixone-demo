import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torch.nn import init
from tqdm import tqdm
import os
import numpy as np
import sys
import cv2

from dataset_v2 import MyDataset
from model import classifier

sys.path.append('/home/littlesc/FR/face.evoLVe.PyTorch-master/')
from code_train.train_utils import set_seed, create_logger
from util.utils import AverageMeter


def train(train_loader, classifier):
    DISP_FREQ = len(train_loader) // 10
    losses = AverageMeter()
    batch = 0
    classifier.train()

    for img, path, fire_label, smoke_label in tqdm(iter(train_loader)):
        inputs = img.cuda().to(DEVICE)
        fire_label = fire_label.cuda().to(DEVICE)
        smoke_label = smoke_label.cuda().to(DEVICE)

        fire_out, smoke_out = classifier(inputs)

        fire_out = fire_out.cuda()
        smoke_out = smoke_out.cuda()

        loss = LOSS_CE(fire_out.float(), fire_label.long()) + LOSS_CE(smoke_out.float(), smoke_label.long())
        losses.update(loss.data.item(), inputs.size(0))
        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        # if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
        #     print('Epoch {}/{} Batch {}/{}\t'
        #           'Training LOSS {loss.val:.4f} ({loss.avg:.4f})\t'.format(
        #         epoch + 1, NUM_EPOCH, batch + 1, len(train_loader),  loss=losses))
        batch += 1  # batch index

    logger.info('\nEpoch: {}/{}\t'
                'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(epoch + 1, NUM_EPOCH, loss=losses))
    model_name = os.path.join(MODEL_ROOT, "CLASSIFIER-B0-224-Epoch_{}.pth".format(epoch + 1))
    torch.save(classifier.state_dict(), model_name, _use_new_zipfile_serialization=False)
    logger.info(model_name)




if __name__ == '__main__':

    logger = create_logger()
    set_seed(233)
    NUM_EPOCH = 200
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_ROOT = '/data1/littlesc/Fire/ckpt_v2/'

    dataset_train_fire = MyDataset(txt_dir='lists/fire_and_smoke_train.txt')
    train_loader_fire = DataLoader(dataset_train_fire, batch_size=32, num_workers=8, shuffle=True)

    LOSS_CE = nn.CrossEntropyLoss()

    CLASSIFIER = classifier()
    CLASSIFIER = CLASSIFIER.to(DEVICE)

    logger.info('\n***************** Begin Training *****************')

    init_epoch = 0
    logger.info('Train from pretrained b0')


    model_name = 'CLASSIFIER-B0-224-Epoch_26.pth'
    init_epoch = 26
    CLADSSIFIER_RESUME_ROOT = MODEL_ROOT + model_name
    CLASSIFIER.load_state_dict(torch.load(CLADSSIFIER_RESUME_ROOT), strict=False)
    logger.info('%s loaded to classifier!' % model_name)

    # OPTIMIZER = optim.Adam([{'params': CLASSIFIER.parameters()}, ], lr=1e-6)
    OPTIMIZER = torch.optim.Adam([{'params': CLASSIFIER.backbone.parameters(), 'lr': 1e-6}], lr=3e-4)

    print("Optimizer Generated")

    for epoch in range(init_epoch, NUM_EPOCH):
        train(train_loader_fire, CLASSIFIER)
