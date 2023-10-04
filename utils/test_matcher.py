import numpy as np 
import os
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
from matplotlib import pyplot as plt
import matplotlib



if __name__=="__main__":
    # tboxes = torch.tensor([[ -3.8046, 1.7498,  -3.3890,  2.1706],
    #                         [ -1.1313,  0.7137,  -0.5888,  1.1269],
    #                         [ 2.4375, -0.4909,  3.0000,  0.0982],
    #                         [-2.8500, -2.1067, -2.3500, -1.4726],
    #                         [ 0.6750, -2.6047,  1.7500, -1.6199]])

    # pboxes = torch.tensor([[-1.0627,  0.6790, -0.1945,  1.7026],
    #                         [ 0.7602, -2.5536,  1.6347, -1.6474],
    #                         [ 1.2602, -2.1536,  1.8347, -1.3474],
    #                         [ -4.0046, 1.5498,  -3.6890,  2.1706],
    #                         [ 4.0911,  0.5808,  4.7361,  1.3391],
    #                         [-2.9182, -2.1412, -2.2930, -1.3508]])


    tboxes = torch.tensor([[ 1.1125, -0.5430,  1.2125, -0.1503],
                            [ 4.1313,  0.7137,  4.5888,  1.1269],
                            [-4.3383, -2.8606, -3.6834, -2.2605],
                            [-4.3383,  3.4226, -3.6834,  4.0227],
                            [-2.8500,  4.1765, -2.3500,  4.3278],
                            [ 2.4375, -0.4909,  3.0000,  0.0982],
                            [-1.0500,  0.7363, -0.2000,  1.7181],
                            [-2.8500, -2.1067, -2.3500, -1.4726],
                            [ -4.0046, 1.75498,  -3.6890,  2.3706],
                            [ -0.0046, 0.498,  0.6890,  0.706],
                            [ 1.0046, 1.498,  1.6890,  2.2706],
                            [ 0.6750, -2.6047,  1.7500, -1.6199]])

    pboxes = torch.tensor([[-1.0627,  0.6790, -0.1945,  1.7026],
                            [ 0.7602, -2.5536,  1.6347, -1.6474],
                            [ 0.8676,  3.7909,  1.7102,  4.3490],
                            [ 4.0911,  0.5808,  4.7361,  1.3391],
                            [ 2.4046, -0.5498,  2.9890,  0.1706],
                            [ 2.7046, -0.5498,  3.1890,  0.1706],
                            [ 1.0046, 1.5498,  1.6890,  2.4706],
                            [ -4.0046, 1.5498,  -3.6890,  2.1706],
                            [-2.9182, -2.1412, -2.2930, -1.3508]])


    iou_mat = torchvision.ops.boxes.box_iou(pboxes, tboxes)
    print(iou_mat.shape,'tboxes: ',tboxes.shape,'pboxes: ',pboxes.shape)
    print(iou_mat)

    matched_vals, matches = iou_mat.max(dim=1)
    wc_truth_boxes = tboxes[matches[np.nonzero(matched_vals)]].reshape(-1,4)
    wc_pred_boxes = pboxes[np.nonzero(matched_vals)].reshape(-1,4)
    print(wc_truth_boxes,wc_truth_boxes.shape)
    wc_truth_boxes = torch.unique(wc_truth_boxes, dim=0)
    wc_pred_boxes = torch.unique(wc_pred_boxes, dim=0)
    print(wc_truth_boxes,wc_truth_boxes.shape)
    print(wc_pred_boxes,wc_pred_boxes.shape)
    print()
    print()
    unmatched_idxs = np.where(matched_vals==0)
    wc_pred_boxes = pboxes[unmatched_idxs].reshape(-1,4)
    wc_truth_boxes = np.delete(tboxes,matches[np.nonzero(matched_vals)],axis=0)
    print(wc_truth_boxes,wc_truth_boxes.shape)
    print(wc_pred_boxes,wc_pred_boxes.shape)









    quit()
    iou_mat = torchvision.ops.boxes.box_iou(pboxes, tboxes)
    print(iou_mat)
    matched_vals, matches = iou_mat.max(dim=1)
    print(matched_vals)
    print(matches)
    print()
    print(matches[matched_vals>0.01])
    print('N preds',len(pboxes))
    print('N matched preds',len(matches[matched_vals>0.01]))
    print('N unmatched preds',len(pboxes) - len(matches[matched_vals>0.01]))
    print('Unique truth boxes matched to',torch.unique(matches[matched_vals>0.01]),len(torch.unique(matches[matched_vals>0.01])))

    
    iou_matt = torchvision.ops.boxes.box_iou(tboxes, pboxes)
    matched_valst, matchest = iou_matt.max(dim=1)
    print()
    print('N truth',len(tboxes))
    print('N matched truth',len(matchest[matched_valst>0.01]))
    print('N unmatched truth',len(tboxes) - len(matchest[matched_valst>0.01]))
    print('Unique pred boxes matched to',torch.unique(matchest[matched_valst>0.01]),len(torch.unique(matchest[matched_valst>0.01])))
    print()
    print()
    print(matched_vals)
    print(matches[matched_vals>0.01])
    print(tboxes[matches[matched_vals>0.01]].shape)
    print(tboxes[matches[matched_vals>0.01]])
    print(pboxes[matched_vals>0.01])

    matched_ts = tboxes[matches[matched_vals>0.01]]
    matched_ps = pboxes[matched_vals>0.01]


    def area1(a, b):  # returns None if rectangles don't intersect
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        if (dx>=0) and (dy>=0):
            return dx*dy

    def area(a, b):  # returns None if rectangles don't intersect
        dx = np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])
        dy = np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])
        intersection_area = np.where((dx >= 0) & (dy >= 0), dx * dy, None)
        return intersection_area
    
    area_covered_by_each_pred = area(matched_ts,matched_ps)
    area_covered_by_overlapping_preds = area(matched_ps,matched_ps)


    area_each_truth = torchvision.ops.box_area(matched_ts).numpy()
    percent_true_covered = area_covered_by_each_pred / area_each_truth
    print(area(matched_ts,matched_ps))
    print('areas',area_each_truth)
    print('percent true covered',percent_true_covered)
    print(np.average(percent_true_covered))
    print('by hand WA',sum(percent_true_covered[g] * area_each_truth[g] / sum(area_each_truth) for g in range(len(percent_true_covered))))
    print(np.average(percent_true_covered,weights=1/area_each_truth))
    print(np.average(percent_true_covered,weights=area_each_truth))






    ext = [-4.82349586,  4.82349586, -6.21738815,  6.21801758]
    # print((ext[1] - ext[0]) * (ext[3] - ext[2]))
    area_check = torchvision.ops.box_area(pboxes)
    # print(area_check)
    # print(sum(area_check))
    # print(sum(area_check)/((ext[1] - ext[0]) * (ext[3] - ext[2])))


    # print(torch.unique(matches[matched_vals>0.02]),len(torch.unique(matches[matched_vals>0.02])))
    # print(len(tboxes)-len(torch.unique(matches[matched_vals>0.02])))
    # matched_truth_boxes_this_image = tboxes[matches[matched_vals>0.02]]

    # matched_GT_boxes_ =  [tuple(trbox) for trbox in matched_truth_boxes_this_image.cpu().tolist()]
    # n_unique_gt_boxes_matched_with = len(list(set(matched_GT_boxes_)))
    # delta_n_matched = len(tboxes) - n_unique_gt_boxes_matched_with
    # print(matched_truth_boxes_this_image)
    # print(torch.tensor(list(set(matched_GT_boxes_))))
    # print(torch.tensor(list(set(matched_GT_boxes_))).shape,matched_truth_boxes_this_image.shape)
    # print(len(torch.tensor(list(set(matched_GT_boxes_)))),len(matched_truth_boxes_this_image))
    # print(delta_n_matched)


    f,ax = plt.subplots(1,1,figsize=(12,8))
    a = (pboxes[:,2:] + pboxes[:,:2]) * 0.5
    b = (tboxes[:,2:] + tboxes[:,:2]) * 0.5
    ax.plot(a[:,0],a[:,1],'.')
    ax.plot(b[:,0],b[:,1],'.')
    ax.plot(pboxes[:,0],pboxes[:,1],'.')
    ax.plot(pboxes[:,2],pboxes[:,3],'.')
    for bbx in tboxes:
        x,y=float(bbx[0]),float(bbx[1])
        w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
        bb = matplotlib.patches.Rectangle((x,y),w,h,lw=2,ec='limegreen',fc='none')
        ax.add_patch(bb)
    for bbx in pboxes:
        x,y=float(bbx[0]),float(bbx[1])
        w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
        bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='red',fc='none')
        ax.add_patch(bb)
    ax.set(xlim=(-6,6),ylim=(-6,6))
    f.savefig('testing-matcher.png')






def find_matched_truth_boxes(truth_boxes, predicted_boxes, iou_threshold=0.5):
    """
    Find the ground truth boxes that match the predicted boxes based on IoU.

    Args:
    truth_boxes (Tensor): Ground truth bounding boxes as a PyTorch tensor of shape (N, 4).
    predicted_boxes (Tensor): Predicted bounding boxes as a PyTorch tensor of shape (M, 4).
    iou_threshold (float): IoU threshold for considering a match.

    Returns:
    Tensor: A tensor containing matched ground truth boxes as PyTorch tensors.
    """
    # Ensure truth_boxes and predicted_boxes are tensors
    if not isinstance(truth_boxes, torch.Tensor):
        truth_boxes = torch.tensor(truth_boxes)

    if not isinstance(predicted_boxes, torch.Tensor):
        predicted_boxes = torch.tensor(predicted_boxes)

    # Calculate the IoU matrix
    iou_mat = torchvision.ops.boxes.box_iou(truth_boxes, predicted_boxes)

    # Find matched truth boxes based on IoU threshold
    max_iou_values, max_iou_indices = iou_mat.max(dim=1)
    matched_truth_boxes = truth_boxes[max_iou_values >= iou_threshold]

    return matched_truth_boxes