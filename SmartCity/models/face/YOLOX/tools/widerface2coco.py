# coding=utf-8
import os
import cv2
import sys
import json
import numpy as np
#import shutil

dataset = { "info": {
            "description": "WIDER face in COCO format.",
            "url": "",
            "version": "1.1",
            "contributor": "aimhabo",
            "date_created": "2020-09-29"},
            "images": [],
            "annotations": [],
            "categories": [{'id': 1, 'name': 'face'}],
}

outputpath = "/data1/littlesc/FR_data/WIDERFACE/annotations/"
image_root = '/data1/littlesc/FR_data/WIDERFACE/WIDER_train/images/'
phase = "WIDERFaceTrainCOCO"

with open('/data1/littlesc/FR_data/wider_face_split/wider_face_train_bbx_gt.txt','r') as f:
    lines = f.readlines()
    num_lines = len(lines)
    i_l=0
    img_id=1
    anno_id=1
    imagepath=None
    while i_l < num_lines:
        print(num_lines, '\\', i_l, '\t-', img_id)
        if len(lines[i_l]) < 1:
            break
        if '--' in lines[i_l]:
            imagepath=lines[i_l].strip()
            im_path=image_root+imagepath
            im = cv2.imread(im_path)
            height, width, channels = im.shape
            dataset["images"].append({"file_name": im_path, "coco_url": "local", "height": height, "width": width, "flickr_url": "local", "id": img_id})
            i_l+=1
            num_gt=int(lines[i_l])
            while num_gt>0:
                i_l+=1
                x1,y1,wid,hei=list(map(int, lines[i_l].split()))[:4]
                num_gt-=1
                dataset["annotations"].append({
                    "segmentation": [],
                    "iscrowd": 0,
                    "area": wid * hei,
                    "image_id": img_id,
                    "bbox": [x1, y1, wid, hei],
                    "category_id": 1,
                    "id": anno_id})
                anno_id = anno_id + 1
                #if im is not None:
                #    cv2.rectangle(im,(x1,y1),(x1+wid,y1+hei), (0,0,0), 3)
                #    cv2.rectangle(im,(x1,y1),(x1+wid,y1+hei), (255,255,255), 1)
            img_id+=1
            #if im is not None:
            #    cv2.imshow('img', im)
            #    cv2.waitKey(0)
        i_l+=1

json_name = os.path.join(outputpath, "{}.json".format(phase))

with open(json_name, 'w') as f:
    json.dump(dataset, f)