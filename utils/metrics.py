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


try:
    from utils.utils import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy
except ModuleNotFoundError:
    from utils import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy


#functions to calculate comparable metrics between predicted and truth boxes



### Geometric!
def delta_n(truth_boxes, predicted_boxes):
    #the difference between the number of boxes predicted and the number of true boxes
    n_objects_per_image = len(truth_boxes)
    n_predicted_objects_per_image = len(predicted_boxes)
    delta_n_objects = n_objects_per_image - n_predicted_objects_per_image
    return delta_n_objects

def n_unmatched_truth(truth_boxes,predicted_boxes):
    #how many truth boxes have no prediction "matched" to them - based on dIoU
    #iou_mat = torchvision.ops.boxes.box_iou(truth_boxes,predicted_boxes)
    #measure of the accuracy
    if isinstance(truth_boxes,np.ndarray):
        truth_boxes = torch.tensor(truth_boxes)
    
    if isinstance(predicted_boxes,np.ndarray):
        predicted_boxes = torch.tensor(predicted_boxes)    
    
    iou_mat = torchvision.ops.boxes.distance_box_iou(truth_boxes,predicted_boxes)
    matched_vals, matches = iou_mat.max(dim=0)
    matched_truth_boxes_this_image = truth_boxes[matches]

    matched_GT_boxes_ =  [tuple(trbox) for trbox in matched_truth_boxes_this_image.cpu().tolist()]
    n_unique_gt_boxes_matched_with = len(list(set(matched_GT_boxes_)))
    delta_n_matched = len(truth_boxes) - n_unique_gt_boxes_matched_with
    
    return delta_n_matched

def n_unmatched_preds(truth_boxes,predicted_boxes):
    #how many predicted boxes have no truth "matched" to them - based on dIoU
    #measure of false positive rate
    rev_iou_mat = torchvision.ops.boxes.distance_box_iou(predicted_boxes,truth_boxes)
    _, matches = rev_iou_mat.max(dim=0)
    matched_preds_boxes_this_image = predicted_boxes[matches]

    matched_p_boxes_ =  [tuple(pbox) for pbox in matched_preds_boxes_this_image.cpu().tolist()]
    n_unique_p_boxes_matched_with = len(list(set(matched_p_boxes_)))
    delta_n_matched_preds = len(predicted_boxes) - n_unique_p_boxes_matched_with
    
    return delta_n_matched_preds


def centre_diffs(truth_boxes,predicted_boxes,filter=False):
    if filter:
        iou = torchvision.ops.boxes.box_iou(truth_boxes,predicted_boxes)
        matched_vals, matches = iou.max(dim=0)
        filtered_matches = matches[matched_vals>0]
        matched_truth_boxes_this_image = truth_boxes[filtered_matches]
        predicted_boxes = predicted_boxes[matched_vals>0]

    elif filter=='rad':
        pcentres = (pboxes[:,2:] + pboxes[:,:2]) * 0.5
        tcentres = (tboxes[:,2:] + tboxes[:,:2]) * 0.5
        dist_mat = np.linalg.norm((pcentres[:,None] - tcentres), axis=-1)
        thresh_mat = np.where(dist_mat < 0.4, dist_mat, 0).reshape(len(pboxes),len(tboxes))
        first_dim,second_dim = np.nonzero(thresh_mat)
        predicted_boxes = predicted_boxes[first_dim]
        matched_truth_boxes_this_image = truth_boxes[second_dim]

    else:
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



def hw_diffs(truth_boxes,predicted_boxes,ratio=False,filter=False):
    #distance between the centre of predicted box and its "closest" (matched) truth box
    #returns two lists of the differences in height and width

    if filter:
        iou = torchvision.ops.boxes.box_iou(truth_boxes,predicted_boxes)
        matched_vals, matches = iou.max(dim=0)
        filtered_matches = matches[matched_vals>0]
        matched_truth_boxes_this_image = truth_boxes[filtered_matches]
        predicted_boxes = predicted_boxes[matched_vals>0]

    elif filter=='rad':
        pcentres = (pboxes[:,2:] + pboxes[:,:2]) * 0.5
        tcentres = (tboxes[:,2:] + tboxes[:,:2]) * 0.5
        dist_mat = np.linalg.norm((pcentres[:,None] - tcentres), axis=-1)
        thresh_mat = np.where(dist_mat < 0.4, dist_mat, 0).reshape(len(pboxes),len(tboxes))
        first_dim,second_dim = np.nonzero(thresh_mat)
        predicted_boxes = predicted_boxes[first_dim]
        matched_truth_boxes_this_image = truth_boxes[second_dim]

    else:
        #distance between the centre of predicted box and its "closest" (matched) truth box
        iou_mat = torchvision.ops.boxes.distance_box_iou(truth_boxes,predicted_boxes)
        matched_vals, matches = iou_mat.max(dim=0)
        matched_truth_boxes_this_image = truth_boxes[matches]

    if len(matched_truth_boxes_this_image) != 0:
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
    
    else:
        return [0]


def area_covered(truth_boxes,predicted_boxes,filter=False):
    #calculate the area of matched! truth boxes covered by our predictions
    #as a fraction of the true area
    if filter:
        iou = torchvision.ops.boxes.box_iou(truth_boxes,predicted_boxes)
        matched_vals, matches = iou.max(dim=0)
        filtered_matches = matches[matched_vals>0]
        matched_truth_boxes_this_image = truth_boxes[filtered_matches]
        predicted_boxes = predicted_boxes[matched_vals>0]
        
    elif filter=='rad':
        pcentres = (pboxes[:,2:] + pboxes[:,:2]) * 0.5
        tcentres = (tboxes[:,2:] + tboxes[:,:2]) * 0.5
        dist_mat = np.linalg.norm((pcentres[:,None] - tcentres), axis=-1)
        thresh_mat = np.where(dist_mat < 0.4, dist_mat, 0).reshape(len(pboxes),len(tboxes))
        first_dim,second_dim = np.nonzero(thresh_mat)
        predicted_boxes = predicted_boxes[first_dim]
        matched_truth_boxes_this_image = truth_boxes[second_dim]

    else:
        #distance between the centre of predicted box and its "closest" (matched) truth box
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







#testing
if __name__=="__main__":
    #always check whether the boxes need to be mutliplied by extents
    tboxes = torch.tensor([[ 1.1125, -0.5430,  1.2125, -0.1503],
                            [ 4.1313,  0.7137,  4.5888,  1.1269],
                            [-4.3383, -2.8606, -3.6834, -2.2605],
                            [-4.3383,  3.4226, -3.6834,  4.0227],
                            [-2.8500,  4.1765, -2.3500,  4.3278],
                            [ 2.4375, -0.4909,  3.0000,  0.0982],
                            [-1.0500,  0.7363, -0.2000,  1.7181],
                            [-2.8500, -2.1067, -2.3500, -1.4726],
                            [ 0.6750,  3.6785,  1.7500,  4.3278],
                            [ 0.6750, -2.6047,  1.7500, -1.6199]])

    pboxes = torch.tensor([[-1.0627,  0.6790, -0.1945,  1.7026],
                            [ 0.7602, -2.5536,  1.6347, -1.6474],
                            [ 0.8676,  3.7909,  1.7102,  4.3490],
                            [ 4.0911,  0.5808,  4.7361,  1.3391],
                            [ 2.4046, -0.5498,  2.9890,  0.1706],
                            [-2.9182, -2.1412, -2.2930, -1.3508]])
    from matplotlib import pyplot as plt
    import matplotlib
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
    ax.set(xlim=(0,5),ylim=(-5,5))
    f.savefig('reg-p-t-boxes.png')

    iou_mat = torchvision.ops.boxes.box_iou(tboxes,pboxes)
    matched_vals, matches = iou_mat.max(dim=0)
    hmm = matches[matched_vals>0]
    print(matched_vals,'\n',matches,'\n',hmm)
    matched_truth_boxes_this_image = tboxes[matches]
    #print(matched_truth_boxes_this_image)
    print(tboxes[hmm])
    print(pboxes[matched_vals>0])
    print()
    a = (pboxes[:,2:] + pboxes[:,:2]) * 0.5
    b = (tboxes[:,2:] + tboxes[:,:2]) * 0.5
    #dist_mat = np.sum((a[:,None] - b)**2, axis=-1)**.5
    dist_mat = np.linalg.norm((a[:,None] - b), axis=-1)
    print(dist_mat)
    thresh_mat = np.where(dist_mat < 0.4, dist_mat, 0).reshape(len(pboxes),len(tboxes))
    first_dim,second_dim = np.nonzero(thresh_mat)
    print()
    print(thresh_mat)
    print(first_dim,second_dim)
    print(pboxes[first_dim])
    print(tboxes[second_dim])
    # print(((1.1950-1.2)**2+(-0.7484--0.7363)**2)**.5)
    quit()

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
















