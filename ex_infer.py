print('Starting')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import time
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from random import randint

from ssd import SSD
from models.models import VGGBase, AuxiliaryConvolutions, PredictionConvolutions
from utils.utils import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy




all_img_folder = os.listdir('/home/users/b/bozianu/work/SSD/SSD/data/input/Images')

all_img_name = []
for img_folder in all_img_folder:
    img_folder_path = '/home/users/b/bozianu/work/SSD/SSD/data/input/Images/' + img_folder
    all_img_name += list(map(lambda x: img_folder + '/'+ x, os.listdir(img_folder_path)))




tsfm = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])




model_save_path = "/home/users/b/bozianu/work/SSD/SSD/models/SSD_model_7.pth"
save_at = "/home/users/b/bozianu/work/SSD/SSD/inference/" + "SSD_model_7/" + time.strftime("%Y%m%d-%H%M%S") + "/"
if not os.path.exists(save_at):
    os.makedirs(save_at)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_images = 15



model = SSD()
model.load_state_dict(torch.load(model_save_path))
model.eval()


for i in range(n_images):
    
    img_number = -1*randint(1,len(all_img_name))
    print('Check here for sanity: ',all_img_name[img_number])
    print('Save loc',len(all_img_name)-img_number)
    origin_img = Image.open('/home/users/b/bozianu/work/SSD/SSD/data/input/Images/' + all_img_name[img_number]).convert('RGB')
    img = tsfm(origin_img)
    img = img.to(device)


    with open('/home/users/b/bozianu/work/SSD/SSD/data/input/Annotation/'+all_img_name[img_number][:-4]) as f:
        reader = f.read()

    xmin = list(map(int,re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', reader)))
    xmax = list(map(int,re.findall('(?<=<xmax>)[0-9]+?(?=</xmax>)', reader)))
    ymin = list(map(int,re.findall('(?<=<ymin>)[0-9]+?(?=</ymin>)', reader)))
    ymax = list(map(int,re.findall('(?<=<ymax>)[0-9]+?(?=</ymax>)', reader)))

    predicted_locs, predicted_scores = model(img.unsqueeze(0))
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=0.2, max_overlap=0.5, top_k=200)
    det_boxes = [box.to(device) for box in det_boxes]
    # print('det_boxes',det_boxes)
    # print('det_labels',det_labels)
    # print('det_scores',det_scores)
    
    origin_dims = torch.FloatTensor([origin_img.width, origin_img.height, origin_img.width, origin_img.height]).unsqueeze(0).to(device)
    det_boxes = [box * origin_dims for box in det_boxes]




    draw = ImageDraw.Draw(origin_img)

    #PREDICTIONS
    pred_boxes = det_boxes[0].tolist()
    print(' # pred_boxes',len(pred_boxes),'!')
    for boxp in pred_boxes:
        draw.rectangle(xy=boxp, outline='red',width=3)

    #TRUTH
    boxes_true = list(map(list, zip(xmin, ymin, xmax,ymax)))
    for boxt in boxes_true:
        draw.rectangle(xy=boxt,outline='limegreen',width=3)

    plt.figure(figsize=(5, 5))
    plt.imshow(origin_img)
    plt.savefig(save_at+'test_infer_img_{}.png'.format(len(all_img_name)-img_number))
    print()

