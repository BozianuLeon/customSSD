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

#from utils import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy
from utils.utils import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy


#functions to calculate comparable metrics between predicted and truth boxes



### Geometric!
def delta_n(truth_boxes, predicted_boxes):
    #the difference between the number of boxes predicted and the number of true boxes
    n_objects_per_image = len(truth_boxes)
    n_predicted_objects_per_image = len(predicted_boxes)
    delta_n_objects = n_objects_per_image - n_predicted_objects_per_image
    return delta_n_objects

def n_unmatched(truth_boxes,predicted_boxes):
    #how many truth boxes have no prediction "matched" to them - based on gIoU
    #iou_mat = torchvision.ops.boxes.box_iou(truth_boxes,predicted_boxes)
    iou_mat = torchvision.ops.boxes.distance_box_iou(truth_boxes,predicted_boxes)
    matched_vals, matches = iou_mat.max(dim=0)
    matched_truth_boxes_this_image = truth_boxes[matches]

    matched_GT_boxes_ =  [tuple(trbox) for trbox in matched_truth_boxes_this_image.cpu().tolist()]
    n_unique_gt_boxes_matched_with = len(list(set(matched_GT_boxes_)))
    delta_n_matched = len(truth_boxes) - n_unique_gt_boxes_matched_with
    
    return delta_n_matched


def centre_diffs(truth_boxes,predicted_boxes):
    #distance between the centre of predicted box and its "closest" (matched) truth box
    iou_mat = torchvision.ops.boxes.distance_box_iou(truth_boxes,predicted_boxes)
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
    #returns two lists of the differences in height and width
    iou_mat = torchvision.ops.boxes.distance_box_iou(truth_boxes,predicted_boxes)
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
    #calculate the area of matched! truth boxes covered by our predictions
    #as a fraction of the true area
    iou_mat = torchvision.ops.boxes.distance_box_iou(truth_boxes,predicted_boxes)
    matched_vals, matches = iou_mat.max(dim=0)
    matched_truth_boxes_this_image = truth_boxes[matches]

    true_area_covered = []
    for gt_box, pred_box in zip(matched_truth_boxes_this_image,predicted_boxes):
        dx = min(gt_box[2],pred_box[2]) - max(gt_box[0],pred_box[0])
        dy = min(gt_box[3],pred_box[3]) - max(gt_box[1],pred_box[1])
        if (dx>0) and (dy>0):
            area_of_intersection = dx*dy
            area_of_truth = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
            true_area_covered.append(area_of_intersection.item()/area_of_truth.item())

    return torch.mean(torch.tensor(true_area_covered))



### Physical!







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



    tboxes = torch.tensor([[-1.6500, -2.9943, -1.5125, -2.7980],
                            [-1.6500,  3.2889, -1.5125,  3.4852],
                            [ 4.2431, -3.0457,  4.5888, -2.7053],
                            [ 4.2431,  3.2375,  4.5888,  3.5779],
                            [-1.6750, -1.1290, -1.2500, -0.9286],
                            [-1.9750,  4.0783, -1.5500,  4.3278],
                            [-0.4500,  1.0308,  0.2500,  1.7181],
                            [-1.9750, -2.2048, -1.0000, -1.3213]])

    pboxes = torch.tensor([[-0.4826,  0.9420,  0.2278,  1.7408],
                            [-1.9768, -2.1506, -0.9706, -1.1996],
                            [ 4.2272,  3.1612,  4.6302,  3.5746],
                            [ 4.2222, -3.0234,  4.6303, -2.5995],
                            [ 2.1292,  2.3016,  2.3423,  2.4964],
                            [-1.8452, -1.5355, -1.3698, -0.8584],
                            [-1.7328,  3.8827, -1.5029,  4.1173],
                            [-1.5159,  3.3068, -1.2927,  3.7263]])
    print(centre_diffs(tboxes,pboxes))






































