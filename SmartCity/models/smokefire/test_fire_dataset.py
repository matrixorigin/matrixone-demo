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
    # if img_save is not None:
    #     base64_str = cv2.imencode('.jpg', img_save)[1].tobytes()
    #     base64_str = base64.b64encode(base64_str)
    #     base = "data:image/jpg;base64," + str(base64_str)[2:-1]  # 写json的时候应该加前面那段
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    json_dict1 = {"camera_id": 1, "frame_id": 2, "time": time_str, "encoding_picture": base,
                  "environment": json_people}
    json_dict2 = {"frame_info": json_dict1}
    with open(output_json, "w") as f:
        f.write(json.dumps(json_dict2))
    # print('%s has been written.' % output_json)


def get_pred(output_gender_tensor):
    softmax = nn.Softmax(dim=0)
    gen_softmax = softmax(output_gender_tensor)
    output_gender_bool = torch.argmax(gen_softmax).item()
    gender_confidence = round(gen_softmax[output_gender_bool].item(), 4)
    return output_gender_bool, gender_confidence


if __name__ == '__main__':

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_ROOT = 'model/'

    CLASSIFIER = classifier()
    CLASSIFIER = CLASSIFIER.to(DEVICE)

    model_name = 'CLASSIFIER-B0-224-Epoch_122.pth'
    CLADSSIFIER_RESUME_ROOT = MODEL_ROOT + model_name
    CLASSIFIER.load_state_dict(torch.load(CLADSSIFIER_RESUME_ROOT), strict=False)
    print('model loaded.')

    dataset_val_fire = MyDataset(txt_dir='lists/fire_test.txt')
    dataset_val_smoke = MyDataset(txt_dir='lists/smoke_test.txt')

    val_loader_fire = DataLoader(dataset_val_fire, batch_size=32, num_workers=8, shuffle=False)
    val_loader_smoke = DataLoader(dataset_val_smoke, batch_size=32, num_workers=8, shuffle=False)
    print('dataset initialized.')

    # s = nn.Softmax(dim=1)

    with torch.no_grad():
        print('validating on the validation set...')
        # print(len(dataset_val), len(val_loader), head)
        right_count = 0
        CLASSIFIER.eval()

        for img, path, _ in tqdm(iter(val_loader_fire)):
            inputs = img.cuda().to(DEVICE)

            start_time = time.time()

            fire_out, smoke_out = CLASSIFIER(inputs)

            end_time = time.time()
            print('%.4f' % (end_time - start_time))

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
                    # print(img_name)
                    write_json(json_people, 'output_json/' + img_name + '.json')

        # print((end_time-start_time)/len(val_loader_fire))
