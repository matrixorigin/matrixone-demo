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

from dataset import MyDataset
from model import classifier

sys.path.append('/home/littlesc/FR/face.evoLVe.PyTorch-master/')
from code_train.train_utils import set_seed, create_logger
from util.utils import AverageMeter

def train(train_loader, head, classifier):
    DISP_FREQ = len(train_loader) // 10
    losses = AverageMeter()
    batch = 0
    classifier.train()

    for img, path, label in tqdm(iter(train_loader)):
        inputs = img.cuda().to(DEVICE)
        label = label.cuda().to(DEVICE)
        if head == 'fire':
            output, _  = classifier(inputs)
        elif head == 'smoke':
            _, output = classifier(inputs)
        output = output.cuda()
        loss = LOSS_CE(output.float(), label.long())
        losses.update(loss.data.item(), inputs.size(0))
        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
            print('Epoch {}/{} Batch {}/{}\t'
                  'Training LOSS_{} {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1, NUM_EPOCH, batch + 1, len(train_loader), head, loss=losses))
        batch += 1  # batch index

    logger.info('\nEpoch: {}/{}\t'
                'Training Loss({}) {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(epoch + 1, NUM_EPOCH, head, loss=losses))
    model_name = os.path.join(MODEL_ROOT, "CLASSIFIER-B0-224-Epoch_{}_{}.pth".format(epoch + 1, head))
    torch.save(classifier.state_dict(), model_name, _use_new_zipfile_serialization=False)
    logger.info(model_name)
    valid(dataset_val_fire, val_loader_fire, 'fire', classifier)
    valid(dataset_val_smoke, val_loader_smoke, 'smoke', classifier)


def valid(dataset_val, val_loader, head, classifier):
    with torch.no_grad():
        print('validating on the validation set...')
        print(len(dataset_val), len(val_loader), head)
        right_count = 0
        classifier.eval()

        for img, path, label in tqdm(iter(val_loader)):
            inputs = img.cuda().to(DEVICE)
            label = label.cuda().to(DEVICE)

            if head == 'fire':
                output, _ = classifier(inputs)
            elif head == 'smoke':
                _, output = classifier(inputs)

            s = nn.Softmax(dim=1)
            output_softmax = s(output)
            output = torch.argmax(output_softmax, dim=1)  # 输出为0或1的单值
            # confidence = output_softmax[output]

            for i in range(len(img)):
                if output[i].item() == label[i].item():
                    right_count += 1

        acc = right_count / len(dataset_val)
        logger.info('在验证集上预测%s的ACC是%.4f' % (head, acc))
        print()


if __name__ == '__main__':

    logger = create_logger()
    set_seed(233)
    NUM_EPOCH = 80
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_ROOT = '/data1/littlesc/Fire/ckpt/'

    dataset_train_fire = MyDataset(txt_dir='lists/fire_train.txt')
    dataset_train_smoke = MyDataset(txt_dir='lists/smoke_train.txt')

    logger.info('dataset_train_fire:%d' % len(dataset_train_fire))
    logger.info('dataset_train_smoke: %d' % len(dataset_train_smoke))

    dataset_val_fire = MyDataset(txt_dir='lists/fire_val.txt')
    dataset_val_smoke = MyDataset(txt_dir='lists/smoke_val.txt')

    logger.info('dataset_val_fire:%d' % len(dataset_val_fire))
    logger.info('dataset_val_smoke:%d' % len(dataset_val_smoke))


    train_loader_fire = DataLoader(dataset_train_fire, batch_size=32, num_workers=8, shuffle=True)
    train_loader_smoke = DataLoader(dataset_train_smoke, batch_size=32, num_workers=8, shuffle=True)

    val_loader_fire = DataLoader(dataset_val_fire, batch_size=32, num_workers=8, shuffle=True)
    val_loader_smoke = DataLoader(dataset_val_smoke, batch_size=32, num_workers=8, shuffle=True)

    LOSS_CE = nn.CrossEntropyLoss()

    CLASSIFIER = classifier()
    CLASSIFIER = CLASSIFIER.to(DEVICE)

    logger.info('\n***************** Begin Training *****************')

    init_epoch = 0
    logger.info('Train from pretrained b0')

    model_name = 'CLASSIFIER-B0-224-Epoch_50_smoke.pth'
    init_epoch = int(model_name.split('_')[1])
    CLADSSIFIER_RESUME_ROOT = MODEL_ROOT + model_name
    CLASSIFIER.load_state_dict(torch.load(CLADSSIFIER_RESUME_ROOT), strict=False)
    logger.info('%s loaded to classifier!' % model_name)



    #OPTIMIZER = optim.Adam([{'params': CLASSIFIER.parameters()}, ], lr=1e-6)
    OPTIMIZER = torch.optim.Adam([{'params': CLASSIFIER.backbone.parameters(), 'lr': 1e-6}], lr=1e-4)

    print("Optimizer Generated")

    for epoch in range(init_epoch, NUM_EPOCH):
        train(train_loader_fire, 'fire', CLASSIFIER)
        train(train_loader_smoke, 'smoke', CLASSIFIER)