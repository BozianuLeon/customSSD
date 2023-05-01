import numpy as np 
import pandas as pd
import os
import re
import time
from tqdm import tqdm
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from utils import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy



#functions to calculate comparable metrics between predicted and truth boxes



### Geometric!
def delta_n(truth_boxes, predicted_boxes):
    #the difference between the number of boxes predicted and the number of true boxes
    n_objects_per_image = len(truth_boxes)
    n_predicted_objects_per_image = len(predicted_boxes)
    delta_n_objects = n_objects_per_image - n_predicted_objects_per_image
    return delta_n_objects

def n_unmatched(truth_boxes,predicted_boxes):
    #how many truth boxes have no prediction "matched" to them - based on IoU
    iou_mat = torchvision.ops.boxes.box_iou(truth_boxes,predicted_boxes)
    matched_vals, matches = iou_mat.max(dim=0)
    matched_truth_boxes_this_image = truth_boxes[matches]
    print(truth_boxes)
    print(matched_truth_boxes_this_image)
    print()

    matched_GT_boxes_ =  [tuple(trbox) for trbox in matched_truth_boxes_this_image.cpu().tolist()]
    n_unique_gt_boxes_matched_with = len(list(set(matched_GT_boxes_)))
    delta_n_matched = len(truth_boxes) - n_unique_gt_boxes_matched_with
    
    return delta_n_matched


def centre_diffs(truth_boxes,predicted_boxes):
    #distance between the centre of predicted box and its "closest" (matched) truth box
    iou_mat = torchvision.ops.boxes.box_iou(truth_boxes,predicted_boxes)
    matched_vals, matches = iou_mat.max(dim=0)
    matched_truth_boxes_this_image = truth_boxes[matches]
    #turn into cx, cy, w, h coords
    tru_cxcywh = xy_to_cxcy(matched_truth_boxes_this_image)
    det_cxcywh = xy_to_cxcy(predicted_boxes)

    #squared L2 norm for difference in centers
    cxcy_diff = torch.sqrt(torch.sum((tru_cxcywh[:,:2]-det_cxcywh[:,:2])**2,dim=1))
    return cxcy_diff.tolist()


def hw_diffs(truth_boxes,predicted_boxes,ratio=False):
    #distance between the centre of predicted box and its "closest" (matched) truth box
    iou_mat = torchvision.ops.boxes.box_iou(truth_boxes,predicted_boxes)
    matched_vals, matches = iou_mat.max(dim=0)
    matched_truth_boxes_this_image = truth_boxes[matches]
    #turn into cx, cy, w, h coords
    tru_cxcywh = xy_to_cxcy(matched_truth_boxes_this_image)
    det_cxcywh = xy_to_cxcy(predicted_boxes)

    if ratio:
        #ratio of truth/predicted
        w_diff = tru_cxcywh[:,2]/det_cxcywh[:,2]
        h_diff = tru_cxcywh[:,3]/det_cxcywh[:,3]
    else:
        #difference
        w_diff = tru_cxcywh[:,2] - det_cxcywh[:,2]
        h_diff = tru_cxcywh[:,3] - det_cxcywh[:,3]

    return h_diff.tolist(), w_diff.tolist()


def area_covered(truth_boxes,predicted_boxes):
    iou_mat = torchvision.ops.boxes.box_iou(truth_boxes,predicted_boxes)
    matched_vals, matches = iou_mat.max(dim=0)
    matched_truth_boxes_this_image = truth_boxes[matches]
    print(iou_mat)

    for gt_box, pred_box in zip(matched_truth_boxes_this_image,predicted_boxes):
        print('\n',pred_box[:2])
        dx = min(gt_box[2],pred_box[2]) - max(gt_box[0],pred_box[0])
        dy = min(gt_box[3],pred_box[3]) - max(gt_box[1],pred_box[1])
        if (dx>0) and (dy>0):
            print(dx*dy,dx*dy/((gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])))


    
    return 0






if __name__=="__main__":
    tboxes = torch.tensor([[ 0.2000,  0.4387,  0.8500,  0.9327],
                           [ 1.2000, -0.7363,  2.1500, -0.0491],
                           [ 1.5250, -4.2706,  2.4500, -2.3071],
                           [ 1.5250,  2.0126,  2.4500,  3.9761]])

    pboxes = torch.tensor([[ 1.1950, -0.7484,  2.0343, -0.0121],
                           [ 1.5079, -4.1209,  2.1564, -3.2197],
                           [ 1.5047,  2.1246,  2.1543,  3.0333],
                           [ 1.8304,  3.4052,  2.4032,  4.0322],
                           [ 0.2163,  0.3306,  0.8147,  0.8914],
                           [ 1.7208, -2.8509,  2.3841, -2.3033]])

    print(area_covered(tboxes,pboxes))






































