import numpy as np 
import pandas as pd
import os
import re
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from random import randint

from ssd import SSD
from models.models import VGGBase, AuxiliaryConvolutions, PredictionConvolutions
from utils.utils import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy
from utils.dataset import SSDDataset





all_img_folder = os.listdir('/home/users/b/bozianu/work/SSD/SSD/data/input/Images')

all_img_name = []
for img_folder in all_img_folder:
    img_folder_path = '/home/users/b/bozianu/work/SSD/SSD/data/input/Images/' + img_folder
    all_img_name += list(map(lambda x: img_folder + '/'+ x, os.listdir(img_folder_path)))


model_save_path = "/home/users/b/bozianu/work/SSD/SSD/models/SSD_model_7.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BS = 8

train_ds = SSDDataset(all_img_name[:int(len(all_img_name)*0.8)])
train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=train_ds.collate_fn)

valid_ds = SSDDataset(all_img_name[int(len(all_img_name)*0.8):int(len(all_img_name)*0.9)])
valid_dl = DataLoader(valid_ds, batch_size=BS, shuffle=True, collate_fn=valid_ds.collate_fn)

test_ds = SSDDataset(all_img_name[int(len(all_img_name)*0.9):])
test_dl = DataLoader(test_ds, batch_size=BS, shuffle=False, collate_fn=test_ds.collate_fn)



model = SSD()
model.load_state_dict(torch.load(model_save_path,map_location=torch.device(device)))
model.eval()




for step, (img, tru_boxes, labels) in enumerate(tqdm(test_dl)):

    img = img.to(device)
    tru_boxes = [box.detach().to(device) for box in tru_boxes]
    labels = [label.detach().to(device) for label in labels]


    predicted_locs, predicted_scores = model(img)
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=0.2, max_overlap=0.5, top_k=200)
    det_boxes = [box.detach().to(device) for box in det_boxes]


    #associate each prediction with a GT box that it overlaps with the most

    for i in range(len(img)):
        truth_boxes_this_image = tru_boxes[i]
        pred_boxes_this_image = det_boxes[i]

        # 1. delta n_boxes

        n_objects_per_image = len(truth_boxes_this_image)
        n_predicted_objects_per_image = len(pred_boxes_this_image)
        delta_n_objects = n_objects_per_image - n_predicted_objects_per_image

        # 2. delta centres, widths, heights

        iou_mat = torchvision.ops.boxes.box_iou(truth_boxes_this_image,pred_boxes_this_image)
        matched_vals, matches = iou_mat.max(dim=0)
        matched_truth_boxes_this_image = truth_boxes_this_image[matches]
        #turn into cx, cy, w, h coords
        tru_cxcywh = xy_to_cxcy(matched_truth_boxes_this_image)
        det_cxcywh = xy_to_cxcy(pred_boxes_this_image)

        #squared L2 norm for difference in centers
        cxcy_diff = torch.sum((tru_cxcywh[:,:2]-det_cxcywh[:,:2])**2,dim=1)
        #signed difference in width and height
        w_diff = tru_cxcywh[:,2] - det_cxcywh[:,2]
        h_diff = tru_cxcywh[:,3] - det_cxcywh[:,3]
        print(w_diff)
        print(h_diff)


    quit()




    #homemade matcher
    #will need to use IoU matrix for this
    #max over gt elems to find the best G candidate for each prediction, which is all we want here
    matched_vals, matches = match_quality_matrix.max(dim=0)




    
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
























