print('Starting')
import numpy as np
import pandas as pd
import os
import re
import time
from tqdm import tqdm
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt
from random import randint

from ssd import SSD
from models.models import VGGBase, AuxiliaryConvolutions, PredictionConvolutions
from utils.utils import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy
from utils.dataset import SSDCOCODataset





das = SSDCOCODataset(
    root_folder="/home/users/b/bozianu/work/data/train2017",
    annotation_json="/home/users/b/bozianu/work/data/annotations/instances_train2017.json",
)
print('Images in dataset:',len(das))

BS = 4
train_size = int(0.8 * len(das))
val_size = int(0.15 * len(das))
test_size = len(das) - train_size - val_size
print('\ttrain / val / test size : ',train_size,'/',val_size,'/',test_size)

torch.random.manual_seed(1)
train_das, val_das, test_das = torch.utils.data.random_split(das,[train_size,val_size,test_size])


train_dal = DataLoader(train_das, batch_size=BS, shuffle=True, collate_fn=das.collate_fn)
val_dal = DataLoader(val_das,batch_size=BS,shuffle=True,collate_fn=das.collate_fn)



tsfm = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


invtsfm = transforms.Compose([ 
    transforms.Normalize(mean = [ 0., 0., 0. ],
                         std = [ 1/0.229, 1/0.224, 1/0.225 ]),
    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                         std = [ 1., 1., 1. ]),
])



model_save_path = "/home/users/b/bozianu/work/SSD/SSD/models/SSD_model_9_coco.pth"
save_at = "/home/users/b/bozianu/work/SSD/SSD/inference/" + "SSD_model_9_coco/" + time.strftime("%Y%m%d-%H%M") + "/"
if not os.path.exists(save_at):
    os.makedirs(save_at)

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_images = 15


model = SSD(device=device)
model.load_state_dict(torch.load(model_save_path,map_location=torch.device(device)))
model.eval()


n_batches = 3

for step, (img, tru_boxes, labels) in enumerate(val_dal):

    img_tensor = img.to(device)
    img = invtsfm(img_tensor)
    tru_boxes = [box.detach().to(device) for box in tru_boxes]
    labels = [label.detach().to(device) for label in labels]

    predicted_locs, predicted_scores = model(img)
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=0.2, max_overlap=0.5, top_k=200)
    det_boxes = [box.detach().to(device) for box in det_boxes]

    #revert to original images
    B, C, H, W = img.shape
    img_shape = torch.FloatTensor([H,W,H,W]).unsqueeze(0)
    det_boxes = [box * img_shape for box in det_boxes]
    tru_boxes = [box * img_shape for box in tru_boxes]


    for i in range(len(img)):
        image_i = img[i].cpu()
        gt_boxes_i = tru_boxes[i]
        boxes_i = det_boxes[i]

        f,ax = plt.subplots()
        ax.imshow(image_i.permute(1,2,0))
        for bbx in gt_boxes_i:
            x,y=float(bbx[0]),float(bbx[1])
            w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
            bb = matplotlib.patches.Rectangle((x,y),w,h,lw=2,ec='limegreen',fc='none')
            ax.add_patch(bb)
        
        for pred_box in boxes_i:
            x,y=float(pred_box[0]),float(pred_box[1])
            w,h=float(pred_box[2])-float(pred_box[0]),float(pred_box[3])-float(pred_box[1])  
            bb = matplotlib.patches.Rectangle((x,y),w,h,lw=2,ec='red',fc='none')
            ax.add_patch(bb)


        f.savefig(save_at + 'testing{}-{}.png'.format(step,i))


    if step==n_batches:
        break



