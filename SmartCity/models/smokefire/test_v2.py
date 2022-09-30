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
import json
from dataset import MyDataset
from model import classifier
import time


def write_json(json_people, output_json):
    base = "Base64"
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    json_dict1 = {"camera_id": 1, "frame_id": 2, "time": time_str, "encoding_picture": base,
                  "environment": json_people}
    json_dict2 = {"frame_info": json_dict1}
    with open(output_json, "w") as f:
        f.write(json.dumps(json_dict2))


def get_pred(output_gender_tensor):
    softmax = nn.Softmax(dim=0)
    gen_softmax = softmax(output_gender_tensor)
    output_gender_bool = torch.argmax(gen_softmax).item()
    gender_confidence = round(gen_softmax[output_gender_bool].item(), 4)
    return output_gender_bool, gender_confidence


def img_loader(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32')
    img = img.transpose(2, 0, 1)
    img = img - np.mean(img)
    img = img / np.max(np.abs(img))

    return img


if __name__ == '__main__':

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CLASSIFIER = classifier()
    CLASSIFIER = CLASSIFIER.to(DEVICE)
    CLADSSIFIER_RESUME_ROOT = 'model/CLASSIFIER-B0-224-Epoch_122.pth'
    CLASSIFIER.load_state_dict(torch.load(CLADSSIFIER_RESUME_ROOT), strict=False)
    print('model loaded.')
    CLASSIFIER.eval()

    test_img_dir = 'test_imgs'

    with torch.no_grad():
        for root2, dirs2, files2 in os.walk(test_img_dir):
            for f in range(len(files2)):

                path = [os.path.join(root2, files2[f])]
                img = img_loader(path[0])
                img = torch.from_numpy(img).unsqueeze(0)
                inputs = img.cuda().to(DEVICE)

                start_time = time.time()
                fire_out, smoke_out = CLASSIFIER(inputs)
                end_time = time.time()
                #print('%.4f' % (end_time - start_time))

                for i in range(img.shape[0]):
                    fire_bool, fire_conf = get_pred(fire_out[i])
                    smoke_bool, smoke_conf = get_pred(smoke_out[i])

                    if fire_bool or smoke_bool:
                        json_people = []
                        if fire_bool:
                            json_person = {"fire": fire_conf}
                            json_people.append(json_person)
                        if smoke_bool:
                            json_smoke = {"smoke": smoke_conf}
                            json_people.append(json_smoke)
                        img_name = path[i].split('/')[-1][:-4]
                        write_json(json_people, 'output_json/' + img_name + '.json')