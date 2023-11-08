import numpy as np 
from numpy.lib.recfunctions import structured_to_unstructured as stu
import os
from itertools import compress

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import time

try:
    from utils.utils import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy
    from utils.utils import wrap_check_NMS, wrap_check_truth, remove_nan, get_cells_from_boxes
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


def n_matched_preds(truth_boxes,predicted_boxes,iou_thresh=0.02):
    #calculate the number of predicted boxes lie on top of a GT box
    iou_mat = torchvision.ops.boxes.box_iou(predicted_boxes, truth_boxes)
    matched_vals, matches = iou_mat.max(dim=1)
    # print('N preds',len(predicted_boxes))
    # print('N matched preds',len(matches[matched_vals>0.01]))
    # print('N unmatched preds',len(pboxes) - len(matches[matched_vals>0.01]))
    #how many distinct truth boxes we are matching to. All boxes match to 1 GT?
    #or more distributed (In practice redundant due to NMS)
    # print('Unique truth boxes matched to',torch.unique(matches[matched_vals>0.01]))
    return len(matches[matched_vals>iou_thresh])

def n_unmatched_preds(truth_boxes,predicted_boxes,iou_thresh=0.02):
    iou_mat = torchvision.ops.boxes.box_iou(predicted_boxes, truth_boxes)
    matched_vals, matches = iou_mat.max(dim=1)
    return len(predicted_boxes) - len(matches[matched_vals>iou_thresh])

def n_matched_truth(truth_boxes,predicted_boxes,iou_thresh=0.02):
    iou_matt = torchvision.ops.boxes.box_iou(truth_boxes, predicted_boxes)
    matched_valst, matchest = iou_matt.max(dim=1)
    return len(matchest[matched_valst>iou_thresh])

def n_unmatched_truth(truth_boxes,predicted_boxes,iou_thresh=0.02):
    iou_matt = torchvision.ops.boxes.box_iou(truth_boxes, predicted_boxes)
    matched_valst, matchest = iou_matt.max(dim=1)
    return len(truth_boxes) - len(matchest[matched_valst>iou_thresh])

def percentage_total_area_covered_by_boxes(boxes,extent):
    total_area = (extent[1] - extent[0]) * (extent[3] - extent[2])
    box_areas = torchvision.ops.box_area(boxes)
    return torch.sum(box_areas) / total_area


def intersection_area(a, b):  # returns None if rectangles don't intersect
    dx = np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])
    dy = np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])
    int_area = np.where((dx >= 0) & (dy >= 0), dx * dy, None)
    return int_area


def percentage_truth_area_covered(preds,truths,iou_thresh=0.01):
    #find the average percentage area of all truth boxes that
    #are covered by predictions
    #TODO: cover case of many overlapping pred boxes (mitigated by NMS)

    #get matched truth and predictions
    iou_mat = torchvision.ops.boxes.box_iou(preds, truths)
    matched_vals, matches = iou_mat.max(dim=1)
    matched_ts = truths[matches[matched_vals>iou_thresh]]
    matched_ps = preds[matched_vals>iou_thresh]
    
    #calculate area of intersection and divide by truth box area
    area_covered_by_each_pred = intersection_area(matched_ts,matched_ps)
    area_each_truth = torchvision.ops.box_area(matched_ts).numpy()
    percent_true_covered = area_covered_by_each_pred / area_each_truth
    
    return np.average(percent_true_covered)



# def n_unmatched_truth(truth_boxes,predicted_boxes):
#     #how many truth boxes have no prediction "matched" to them - based on dIoU
#     #iou_mat = torchvision.ops.boxes.box_iou(truth_boxes,predicted_boxes)
#     #measure of the accuracy
#     if isinstance(truth_boxes,np.ndarray):
#         truth_boxes = torch.tensor(truth_boxes)
    
#     if isinstance(predicted_boxes,np.ndarray):
#         predicted_boxes = torch.tensor(predicted_boxes)    
    
#     iou_mat = torchvision.ops.boxes.distance_box_iou(truth_boxes,predicted_boxes)
#     matched_vals, matches = iou_mat.max(dim=0)
#     matched_truth_boxes_this_image = truth_boxes[matches]

#     matched_GT_boxes_ =  [tuple(trbox) for trbox in matched_truth_boxes_this_image.cpu().tolist()]
#     n_unique_gt_boxes_matched_with = len(list(set(matched_GT_boxes_)))
#     delta_n_matched = len(truth_boxes) - n_unique_gt_boxes_matched_with
    
#     return delta_n_matched

# def n_unmatched_preds(truth_boxes,predicted_boxes):
#     #how many predicted boxes have no truth "matched" to them - based on dIoU
#     #measure of false positive rate
#     rev_iou_mat = torchvision.ops.boxes.distance_box_iou(predicted_boxes,truth_boxes)
#     _, matches = rev_iou_mat.max(dim=0)
#     matched_preds_boxes_this_image = predicted_boxes[matches]

#     matched_p_boxes_ =  [tuple(pbox) for pbox in matched_preds_boxes_this_image.cpu().tolist()]
#     n_unique_p_boxes_matched_with = len(list(set(matched_p_boxes_)))
#     delta_n_matched_preds = len(predicted_boxes) - n_unique_p_boxes_matched_with
    
#     return delta_n_matched_preds


# def centre_diffs(truth_boxes,predicted_boxes,filter=False):
#     if filter:
#         iou = torchvision.ops.boxes.box_iou(truth_boxes,predicted_boxes)
#         matched_vals, matches = iou.max(dim=0)
#         filtered_matches = matches[matched_vals>0]
#         matched_truth_boxes_this_image = truth_boxes[filtered_matches]
#         predicted_boxes = predicted_boxes[matched_vals>0]

#     elif filter=='rad':
#         pcentres = (pboxes[:,2:] + pboxes[:,:2]) * 0.5
#         tcentres = (tboxes[:,2:] + tboxes[:,:2]) * 0.5
#         dist_mat = np.linalg.norm((pcentres[:,None] - tcentres), axis=-1)
#         thresh_mat = np.where(dist_mat < 0.4, dist_mat, 0).reshape(len(pboxes),len(tboxes))
#         first_dim,second_dim = np.nonzero(thresh_mat)
#         predicted_boxes = predicted_boxes[first_dim]
#         matched_truth_boxes_this_image = truth_boxes[second_dim]

#     else:
#         #distance between the centre of predicted box and its "closest" (matched) truth box
#         iou_mat = torchvision.ops.boxes.distance_box_iou(truth_boxes,predicted_boxes)
#         matched_vals, matches = iou_mat.max(dim=0)
#         matched_truth_boxes_this_image = truth_boxes[matches]


#     #turn into cx, cy, w, h coords
#     tru_cxcywh = xy_to_cxcy(matched_truth_boxes_this_image)
#     det_cxcywh = xy_to_cxcy(predicted_boxes)

#     #squared L2 norm for difference in centers
#     cxcy_diff = torch.sqrt(torch.sum((tru_cxcywh[:,:2]-det_cxcywh[:,:2])**2,dim=1))
#     return cxcy_diff.tolist()



# def hw_diffs(truth_boxes,predicted_boxes,ratio=False,filter=False):
#     #distance between the centre of predicted box and its "closest" (matched) truth box
#     #returns two lists of the differences in height and width

#     if filter:
#         iou = torchvision.ops.boxes.box_iou(truth_boxes,predicted_boxes)
#         matched_vals, matches = iou.max(dim=0)
#         filtered_matches = matches[matched_vals>0]
#         matched_truth_boxes_this_image = truth_boxes[filtered_matches]
#         predicted_boxes = predicted_boxes[matched_vals>0]

#     elif filter=='rad':
#         pcentres = (pboxes[:,2:] + pboxes[:,:2]) * 0.5
#         tcentres = (tboxes[:,2:] + tboxes[:,:2]) * 0.5
#         dist_mat = np.linalg.norm((pcentres[:,None] - tcentres), axis=-1)
#         thresh_mat = np.where(dist_mat < 0.4, dist_mat, 0).reshape(len(pboxes),len(tboxes))
#         first_dim,second_dim = np.nonzero(thresh_mat)
#         predicted_boxes = predicted_boxes[first_dim]
#         matched_truth_boxes_this_image = truth_boxes[second_dim]

#     else:
#         #distance between the centre of predicted box and its "closest" (matched) truth box
#         iou_mat = torchvision.ops.boxes.distance_box_iou(truth_boxes,predicted_boxes)
#         matched_vals, matches = iou_mat.max(dim=0)
#         matched_truth_boxes_this_image = truth_boxes[matches]

#     if len(matched_truth_boxes_this_image) != 0:
#         #turn into cx, cy, w, h coords
#         tru_cxcywh = xy_to_cxcy(matched_truth_boxes_this_image)
#         det_cxcywh = xy_to_cxcy(predicted_boxes)

#         if ratio:
#             #ratio of truth/predicted
#             w_diff = tru_cxcywh[:,2]/det_cxcywh[:,2]
#             h_diff = tru_cxcywh[:,3]/det_cxcywh[:,3]
#         else:
#             #difference
#             w_diff = tru_cxcywh[:,2] - det_cxcywh[:,2]
#             h_diff = tru_cxcywh[:,3] - det_cxcywh[:,3]

#         return h_diff.tolist(), w_diff.tolist()
    
#     else:
#         return [0]


# def area_covered(truth_boxes,predicted_boxes,filter=False):
#     #calculate the area of matched! truth boxes covered by our predictions
#     #as a fraction of the true area
#     if filter:
#         iou = torchvision.ops.boxes.box_iou(truth_boxes,predicted_boxes)
#         matched_vals, matches = iou.max(dim=0)
#         filtered_matches = matches[matched_vals>0]
#         matched_truth_boxes_this_image = truth_boxes[filtered_matches]
#         predicted_boxes = predicted_boxes[matched_vals>0]
        
#     elif filter=='rad':
#         pcentres = (pboxes[:,2:] + pboxes[:,:2]) * 0.5
#         tcentres = (tboxes[:,2:] + tboxes[:,:2]) * 0.5
#         dist_mat = np.linalg.norm((pcentres[:,None] - tcentres), axis=-1)
#         thresh_mat = np.where(dist_mat < 0.4, dist_mat, 0).reshape(len(pboxes),len(tboxes))
#         first_dim,second_dim = np.nonzero(thresh_mat)
#         predicted_boxes = predicted_boxes[first_dim]
#         matched_truth_boxes_this_image = truth_boxes[second_dim]

#     else:
#         #distance between the centre of predicted box and its "closest" (matched) truth box
#         iou_mat = torchvision.ops.boxes.distance_box_iou(truth_boxes,predicted_boxes)
#         matched_vals, matches = iou_mat.max(dim=0)
#         matched_truth_boxes_this_image = truth_boxes[matches]

#     true_area_covered = []
#     for gt_box, pred_box in zip(matched_truth_boxes_this_image,predicted_boxes):
#         dx = min(gt_box[2],pred_box[2]) - max(gt_box[0],pred_box[0])
#         dy = min(gt_box[3],pred_box[3]) - max(gt_box[1],pred_box[1])
#         if (dx>0) and (dy>0):
#             area_of_intersection = dx*dy
#             area_of_truth = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
#             true_area_covered.append(area_of_intersection.item()/area_of_truth.item())

#     return torch.mean(torch.tensor(true_area_covered))



def circular_mean(phi_values):
    """
    Calculate the circular mean (average) of a list of phi_values.
    Handles the periodicity of phi_values correctly.
    
    :param phi_values: List of phi_values in radians
    :return: Circular mean in radians
    """
    sin_sum = np.sum(np.sin(phi_values))
    cos_sum = np.sum(np.cos(phi_values))
    circular_mean = np.arctan2(sin_sum, cos_sum)
    return circular_mean


def weighted_circular_mean(phi_values, energy_values):
    """
    Calculate the weighted circular mean (average) of a list of angles.
    Handles the periodicity of phi correctly. http://palaeo.spb.ru/pmlibrary/pmbooks/mardia&jupp_2000.pdf

    :param phi_values: List of angles in radians
    :param energy_values: List of weights corresponding to each phi value
    :return: Weighted circular mean in radians
    """
    if len(phi_values) != len(energy_values):
        raise ValueError("phi_values and energy_values must have the same length")

    weighted_sin_sum = np.sum(energy_values * np.sin(phi_values))
    weighted_cos_sum = np.sum(energy_values * np.cos(phi_values))
    weighted_circular_mean = np.arctan2(weighted_sin_sum, weighted_cos_sum)
    return weighted_circular_mean


# TODO : Future use should prioritise saving cell_IdCells rather than a list, then extracting using
# np.isin to find cells from cells
# also find out how many negative cells are in the boxes!

def grab_cells_from_boxes(pred_boxes,scores,truth_boxes,cells,mode='match',wc=False):
    # arguments are:
    # pred_boxes, model output, no augmentation/wrap checking [n_preds,4] NO WRAP CHECK
    # truth_boxes, from .npy file, [n_objs, 4]
    # mode, tells us whether we should look at matched predictions, unmatched, or all (None) ['match','unmatch']
    # wc, if the boxes have been wrap checked before this function

    wc_pred_boxes = pred_boxes
    wc_truth_boxes = truth_boxes
    if not wc:
        wc_pred_boxes = wrap_check_NMS(pred_boxes,scores,min(cells['cell_phi']),max(cells['cell_phi']),threshold=0.2)
        wc_truth_boxes = wrap_check_truth(truth_boxes,min(cells['cell_phi']),max(cells['cell_phi']))

    if mode=='match':
        iou_mat = torchvision.ops.boxes.box_iou(torch.tensor(wc_truth_boxes),torch.tensor(wc_pred_boxes))
        matched_vals, matches = iou_mat.max(dim=0)
        wc_truth_boxes = wc_truth_boxes[matches[np.nonzero(matched_vals)]].reshape(-1,4)
        wc_pred_boxes = wc_pred_boxes[np.nonzero(matched_vals)].reshape(-1,4)
        wc_truth_boxes = torch.unique(torch.tensor(wc_truth_boxes), dim=0)
        wc_pred_boxes = torch.unique(torch.tensor(wc_pred_boxes), dim=0)
        wc_pred_boxes = wc_pred_boxes.numpy()
        wc_truth_boxes = wc_truth_boxes.numpy()    
    elif mode=='unmatch':
        iou_mat = torchvision.ops.boxes.box_iou(torch.tensor(wc_truth_boxes),torch.tensor(wc_pred_boxes))
        matched_vals, matches = iou_mat.max(dim=0)
        unmatched_idxs = np.where(matched_vals==0)
        wc_pred_boxes = wc_pred_boxes[unmatched_idxs].reshape(-1,4)
        wc_truth_boxes = np.delete(wc_truth_boxes,matches[np.nonzero(matched_vals)],axis=0)

    list_pred_cl_cells = get_cells_from_boxes(wc_pred_boxes,cells)
    list_tru_cl_cells = get_cells_from_boxes(wc_truth_boxes,cells)   

    # Check that neither list using placeholder values has an entry with no cells
    # zero_cells_mask tells us that this box contains more than 0 cells
    pred_zero_cells_mask = [sum(data['cell_BadCells']) >= 0 for data in list_pred_cl_cells]
    list_pred_cl_cells = list(compress(list_pred_cl_cells, pred_zero_cells_mask))
    true_zero_cells_mask = [sum(data['cell_BadCells']) >= 0 for data in list_tru_cl_cells]
    list_tru_cl_cells = list(compress(list_tru_cl_cells, true_zero_cells_mask))

    return list_pred_cl_cells, list_tru_cl_cells



def extract_physics_variables(list_pred_box_cells, list_tru_box_cells, target='energy'):
    if target == 'energy':
        list_pred_cl_energies = [sum(x['cell_E']) for x in list_pred_box_cells]
        list_tru_cl_energies = [sum(x['cell_E']) for x in list_tru_box_cells]
        return list_pred_cl_energies, list_tru_cl_energies

    def calc_cl_eta(cl_array):
        return np.dot(cl_array['cell_eta'],np.abs(cl_array['cell_E'])) / sum(np.abs(cl_array['cell_E']))
    def calc_cl_phi(cl_array): 
        # return np.dot(cl_array['cell_phi'],np.abs(cl_array['cell_E'])) / sum(np.abs(cl_array['cell_E']))
        return weighted_circular_mean(cl_array['cell_phi'],cl_array['cell_E'])

    if target  == 'eta':
        list_pred_cl_etas = [calc_cl_eta(x) for x in list_pred_box_cells]
        list_tru_cl_etas = [calc_cl_eta(x) for x in list_tru_box_cells]
        return list_pred_cl_etas, list_tru_cl_etas
    
    if target == 'phi':
        list_pred_cl_phis = [calc_cl_phi(x) for x in list_pred_box_cells]
        list_tru_cl_phis = [calc_cl_phi(x) for x in list_tru_box_cells]
        return list_pred_cl_phis, list_tru_cl_phis
    
    if target in ['eT','et']:
        list_pred_cl_et = [sum(x['cell_E'])/np.cosh(calc_cl_eta(x)) for x in list_pred_box_cells]
        list_tru_cl_et = [sum(x['cell_E'])/np.cosh(calc_cl_eta(x)) for x in list_tru_box_cells]
        return list_pred_cl_et, list_tru_cl_et
    
    if target == 'n_cells':
        list_pred_cl_ns = [len(x) for x in list_pred_box_cells]
        list_tru_cl_ns = [len(x) for x in list_tru_box_cells]
        return list_pred_cl_ns, list_tru_cl_ns  



def event_cluster_estimates(pred_boxes, scores, truth_boxes, cells, mode='match',target='energy',wc=False):
    #arguments are:
    #pred_boxes, model output, no augmentation/wrap checking [n_preds,4] NO WRAP CHECK
    #truth_boxes, from .npy file, [n_objs, 4]
    #mode, tells us whether we should look at matched predictions, unmatched, or all (None) ['match','unmatch']
    #target, cluster statistic to check ['energy','eta','phi']
    #wc, if the boxes have been wrap checked before this function

    wc_pred_boxes = pred_boxes
    wc_truth_boxes = truth_boxes
    if not wc:
        wc_pred_boxes = wrap_check_NMS(pred_boxes,scores,min(cells['cell_phi']),max(cells['cell_phi']),threshold=0.2)
        wc_truth_boxes = wrap_check_truth(truth_boxes,min(cells['cell_phi']),max(cells['cell_phi']))

    if mode=='match':
        iou_mat = torchvision.ops.boxes.box_iou(torch.tensor(wc_truth_boxes),torch.tensor(wc_pred_boxes))
        matched_vals, matches = iou_mat.max(dim=0)
        wc_truth_boxes = wc_truth_boxes[matches[np.nonzero(matched_vals)]].reshape(-1,4)
        wc_pred_boxes = wc_pred_boxes[np.nonzero(matched_vals)].reshape(-1,4)
        wc_truth_boxes = torch.unique(torch.tensor(wc_truth_boxes), dim=0)
        wc_pred_boxes = torch.unique(torch.tensor(wc_pred_boxes), dim=0)
        wc_pred_boxes = wc_pred_boxes.numpy()
        wc_truth_boxes = wc_truth_boxes.numpy()    
    elif mode=='unmatch':
        iou_mat = torchvision.ops.boxes.box_iou(torch.tensor(wc_truth_boxes),torch.tensor(wc_pred_boxes))
        matched_vals, matches = iou_mat.max(dim=0)
        unmatched_idxs = np.where(matched_vals==0)
        wc_pred_boxes = wc_pred_boxes[unmatched_idxs].reshape(-1,4)
        wc_truth_boxes = np.delete(wc_truth_boxes,matches[np.nonzero(matched_vals)],axis=0)


    list_pred_cl_cells = get_cells_from_boxes(wc_pred_boxes,cells)
    list_tru_cl_cells = get_cells_from_boxes(wc_truth_boxes,cells)
    
    # Check that neither list using placeholder values has an entry with no cells
    # zero_cells_mask tells us that this box contains more than 0 cells
    pred_zero_cells_mask = [sum(data['cell_BadCells']) >= 0 for data in list_pred_cl_cells]
    list_pred_cl_cells = list(compress(list_pred_cl_cells, pred_zero_cells_mask))
    true_zero_cells_mask = [sum(data['cell_BadCells']) >= 0 for data in list_tru_cl_cells]
    list_tru_cl_cells = list(compress(list_tru_cl_cells, true_zero_cells_mask))
    
    if target == 'energy':
        list_pred_cl_energies = [sum(x['cell_E']) for x in list_pred_cl_cells]
        list_tru_cl_energies = [sum(x['cell_E']) for x in list_tru_cl_cells]
        return list_pred_cl_energies, list_tru_cl_energies

    def calc_cl_eta(cl_array):
        return np.dot(cl_array['cell_eta'],np.abs(cl_array['cell_E'])) / sum(np.abs(cl_array['cell_E']))
    def calc_cl_phi(cl_array): 
        # return np.dot(cl_array['cell_phi'],np.abs(cl_array['cell_E'])) / sum(np.abs(cl_array['cell_E']))
        return weighted_circular_mean(cl_array['cell_phi'],cl_array['cell_E'])

    if target  == 'eta':
        list_pred_cl_etas = [calc_cl_eta(x) for x in list_pred_cl_cells]
        list_tru_cl_etas = [calc_cl_eta(x) for x in list_tru_cl_cells]
        return list_pred_cl_etas, list_tru_cl_etas
    
    if target == 'phi':
        list_pred_cl_phis = [calc_cl_phi(x) for x in list_pred_cl_cells]
        list_tru_cl_phis = [calc_cl_phi(x) for x in list_tru_cl_cells]
        return list_pred_cl_phis, list_tru_cl_phis
    
    if target == 'eT' or 'et':
        list_pred_cl_et = [sum(x['cell_E'])/np.cosh(calc_cl_eta(x)) for x in list_pred_cl_cells]
        list_tru_cl_et = [sum(x['cell_E'])/np.cosh(calc_cl_eta(x)) for x in list_tru_cl_cells]
        return list_pred_cl_et, list_tru_cl_et
    
    if target == 'n_cells':
        list_pred_cl_ns = [len(x) for x in list_pred_cl_cells]
        list_tru_cl_ns = [len(x) for x in list_tru_cl_cells]
        return list_pred_cl_ns, list_tru_cl_ns  





def is_cluster_enclosed_in_box(cluster_cell_d, cells_this_event, box_xyxy):
    # number of clusters inside a box based on the cells in that cluster
    # can overestimate due to wrapping. Sometimes boxes that go "off the page" 
    # count clusters that are at the other side of the image, this is okay!

    which_clusters_are_inside = []
    for i in range(cluster_cell_d.shape[0]):
        cell_ids = cluster_cell_d[i]['cl_cell_IdCells']
        cell_ids = cell_ids[np.nonzero(cell_ids)]
        wanted_cell_ids = np.isin(cells_this_event['cell_IdCells'],cell_ids)
        desired_cells = cells_this_event[wanted_cell_ids]

        #if x condition satisfied
        if box_xyxy[0] <= np.mean(desired_cells['cell_eta']) <= box_xyxy[2]:
            # print('x satisfied!')
            #if y condition satisfied
            y_mean_circ = circular_mean(desired_cells['cell_phi'])
            if box_xyxy[1] <= y_mean_circ <= box_xyxy[3]:
                # print('y_satisfied!')
                which_clusters_are_inside.append(True)
            elif box_xyxy[1] <= (y_mean_circ + (-1*np.sign(y_mean_circ))*2*np.pi) <= box_xyxy[3]:
                # print('y wrap satisfied')
                which_clusters_are_inside.append(True)  
            else:
                which_clusters_are_inside.append(False)
        else:
            which_clusters_are_inside.append(False)

    return  sum(which_clusters_are_inside)



import multiprocessing

def n_clusters_per_box(truth_boxes,cluster_cell_data,cells_this_event):
    # this function should tell us how many topoclusters are inside each truth box
    # find the number of clusters that make up each truth box.
    # multiprocessing because slow!

    # n_cl_per_box = []
    # for idx,tb in enumerate(truth_boxes):
    #     bo_array = is_cluster_enclosed_in_box(cluster_cell_data, cells_this_event, tb) 
    #     n_cl_per_box.append(sum(bo_array))
    # n_cl_per_box = np.fromiter((is_cluster_enclosed_in_box(cluster_cell_data, cells_this_event,tb) for tb in truth_boxes), dtype=int, count=len(truth_boxes))
    # n_cl_per_box = np.array([is_cluster_enclosed_in_box(cluster_cell_data, cells_this_event,tb) for tb in truth_boxes])
    
    num_workers = 4 #multiprocessing.cpu_count()
    chunk_size = 1 if len(truth_boxes) // num_workers==0 else len(truth_boxes) // num_workers
    print('\tNum workers: ', num_workers, 'Chunk size',chunk_size)
    
    pool = multiprocessing.Pool()
    results = []
    for i in range(0,len(truth_boxes),chunk_size):
        chunk = truth_boxes[i:i + chunk_size]
        # chunk_results = pool.map(func1, [a_constant] * len(chunk), chunk)
        chunk_results = pool.starmap(is_cluster_enclosed_in_box, [(cluster_cell_data, cells_this_event, x) for x in chunk])
        results.extend(chunk_results)
    
    pool.close()
    pool.join()
    
    return results





def clusters_in_box_E_diff(truth_boxes,cluster_data,cluster_cell_data, cells_this_event):
    # this function should compare the truth boxes to ALL clusters contained within them
    # this used the cell_Ids of clusters (NOT cl_cell_ information)
    tot_clus_E_per_box = []
    tot_clus_cell_E_per_box = []
    for idx,tb in enumerate(truth_boxes):
        e1,e2 = total_energy_in_truth_box(cluster_data,cluster_cell_data,cells_this_event,tb)
        tot_clus_E_per_box.append(e1)
        tot_clus_cell_E_per_box.append(e2)
    
    return tot_clus_E_per_box,tot_clus_cell_E_per_box






def number_cluster_in_tboxes(cluster_cell_d, cells_this_event, boxes):
    # number of clusters inside a box based on the cells in that cluster
    # can overestimate due to wrapping. Sometimes boxes that go "off the page" 
    # count clusters that are at the other side of the image, this is okay!

    l_topo_cells = RetrieveCellIdsFromCluster(cells_this_event,cluster_cell_d)
    n_tc_in_box = []
    for truth_box in boxes:
        which_clusters_are_inside = []
        for cl_no in range(len(l_topo_cells)):
            cluster_cells = l_topo_cells[cl_no]
            #if x condition satisfied
            if truth_box[0] <= np.mean(cluster_cells['cell_eta']) <= truth_box[2]:
                #if y condition satisfied
                y_mean_circ = circular_mean(cluster_cells['cell_phi'])
                if truth_box[1] <= y_mean_circ <= truth_box[3]:
                    which_clusters_are_inside.append(True)
                elif truth_box[1] <= (y_mean_circ + (-1*np.sign(y_mean_circ))*2*np.pi) <= truth_box[3]:
                    which_clusters_are_inside.append(True)  
                else:
                    which_clusters_are_inside.append(False)
            else:
                which_clusters_are_inside.append(False)
        n_tc_in_box.append(sum(which_clusters_are_inside))
        
    return n_tc_in_box


def total_energy_in_truth_box(cluster_d, cluster_cell_d, cells_this_event, boxes):
    # number of clusters inside a box based on the cells in that cluster
    # can overestimate due to wrapping. Sometimes boxes that go "off the page" 
    # count clusters that are at the other side of the image, this is okay!

    list_cl_cell_E = []
    list_cl_E = []
    l_topo_cells = RetrieveCellIdsFromCluster(cells_this_event,cluster_cell_d)
    for truth_box in boxes:
        cluster_E_in_tbox = 0
        cluster_cell_E_in_tbox = 0
        for cl_no in range(len(l_topo_cells)):
            cluster_cells = l_topo_cells[cl_no]
            #if x condition satisfied
            if truth_box[0] <= np.mean(cluster_cells['cell_eta']) <= truth_box[2]:
                #if y condition satisfied
                y_mean_circ = circular_mean(cluster_cells['cell_phi'])
                if truth_box[1] <= y_mean_circ <= truth_box[3]:
                    cluster_E_in_tbox += cluster_d[cl_no]['cl_E_em']
                    cluster_E_in_tbox += cluster_d[cl_no]['cl_E_had']
                    cluster_cell_E_in_tbox += sum(cluster_cells['cell_E'])
                elif truth_box[1] <= (y_mean_circ + (-1*np.sign(y_mean_circ))*2*np.pi) <= truth_box[3]:
                    cluster_E_in_tbox += cluster_d[cl_no]['cl_E_em']
                    cluster_E_in_tbox += cluster_d[cl_no]['cl_E_had']
                    cluster_cell_E_in_tbox += sum(cluster_cells['cell_E'])
        
        list_cl_cell_E.append(cluster_cell_E_in_tbox)
        list_cl_E.append(cluster_E_in_tbox)

    return  list_cl_E, list_cl_cell_E













######################################################################################################
# New multiprocessing regime
######################################################################################################


######################################################################################################
# Get Cells from boxes/clusters



def RetrieveCellIdsFromBox(cells,boxes):

    ymin,ymax = min(cells['cell_phi']),max(cells['cell_phi']) # get physical bounds of calo cells
    list_containing_all_cells = []
    for box in boxes:
        eta_min,phi_min,eta_max,phi_max = box
        x_condition = np.logical_and.reduce((cells['cell_eta']>=eta_min, cells['cell_eta']<=eta_max))
        
        #box straddles bottom of image
        if (phi_min < ymin) and (phi_max > ymin):
            modded_box = box + np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            top_of_top_box = min(modded_box[3],ymax)
            y_condtion1 = np.logical_and.reduce((cells['cell_phi']>=ymin, cells['cell_phi']<=phi_max))
            y_condtion2 = np.logical_and.reduce((cells['cell_phi']>=modded_box[1], cells['cell_phi']<=top_of_top_box))
            y_cond = np.logical_or(y_condtion1,y_condtion2)
        
        #box straddles top of image
        elif (phi_max > ymax) and (phi_min < ymax):
            modded_box = box - np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            bottom_of_bottom_box = min(modded_box[1],ymin)
            y_condtion1 = np.logical_and.reduce((cells['cell_phi'] >= phi_min, cells['cell_phi'] <= ymax))
            y_condtion2 = np.logical_and.reduce((cells['cell_phi'] >= bottom_of_bottom_box, cells['cell_phi'] <= modded_box[3]))
            y_cond = np.logical_or(y_condtion1,y_condtion2)

        #box is completely above top
        elif (phi_max < ymin):
            modded_box = box + np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            y_cond = np.logical_and.reduce((cells['cell_phi']>=modded_box[1], cells['cell_phi']>=modded_box[3]))

        elif (phi_min > ymax):
            modded_box = box - np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            y_cond = np.logical_and.reduce((cells['cell_phi']>=modded_box[1], cells['cell_phi']>=modded_box[3]))
        else:
            y_cond = np.logical_and.reduce((cells['cell_phi']>=phi_min, cells['cell_phi']<=phi_max)) #multiple conditions #could use np.all(x,axis)
        
        tot_cond = np.logical_and(x_condition,y_cond)
        cells_here = cells[np.where(tot_cond)][['cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells']]
        if len(cells_here):
            list_containing_all_cells.append(cells_here)
        else:
            # so that we know where there were no cells!
            # placeholder_values = np.array([(-1.0,-99.9,-99.9,-1.0,-10.0)],
            #     dtype=[('cell_E', '<f4'), ('cell_eta', '<f4'), ('cell_phi', '<f4'), ('cell_Sigma', '<f4'), ('cell_IdCells', '<u4')])
            # list_containing_all_cells.append(placeholder_values)
            list_containing_all_cells.append(None)

    # Check that the list does not include None/placeholder values and if it does, remove it
    filtered_list = [x for x in list_containing_all_cells if x is not None]
    # somelist = [x for x in list_containing_all_cells if not min(x['cell_phi']) < -99.0]

    return filtered_list




def RetrieveCellIdsFromCluster(cells,cluster_cell_info):
    # Inputs
    # cluster_cell_info, structured array containing cluster *cell* information
    #                   including cell ids for cells in topocluster
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # cell_ids, array containing cell ids of all cells inside this topocluster

    list_containing_all_cells = []
    for cluster in range(len(cluster_cell_info)):
        cell_mask = np.isin(cells['cell_IdCells'],cluster_cell_info[cluster]['cl_cell_IdCells'])
        desired_cells = cells[cell_mask][['cell_E','cell_eta','cell_phi','cell_Sigma','cell_IdCells']]
        list_containing_all_cells.append(desired_cells)
    # return desired_cells
    return list_containing_all_cells




# def GetCellsFromClusters(cluster_cell_data, cells):
#     # cluster_cell_data cluster cell info after E cut
#     # cells in this event
    
#     num_workers = 4 #multiprocessing.cpu_count()
#     chunk_size = 1 if len(cluster_cell_data) // num_workers==0 else len(cluster_cell_data) // num_workers
#     pool = multiprocessing.Pool(processes=num_workers)
#     results = []
#     for j in range(0,len(cluster_cell_data),chunk_size):
#         mp_chunk = cluster_cell_data[j:j+chunk_size]
#         chunk_results = pool.starmap(RetrieveCellIdsFromCluster, [(cells, y) for y in mp_chunk])
#         results.extend(chunk_results)

#     pool.close()
#     pool.join()

#     return results






################################################################################################
# Get physics things from cell ids



def CalculateEnergyFromCells(all_cells,desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # total energy,  for this collection of cells
    
    if desired_cells is None:
        return np.nan
    return sum(desired_cells['cell_E']) 

def CalculateEtaFromCells(all_cells,desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # eta, (absolute) energy weighted value
    
    if desired_cells is None:
        return np.nan    
    energy_weighted_eta = np.dot(desired_cells['cell_eta'],np.abs(desired_cells['cell_E'])) / sum(np.abs(desired_cells['cell_E']))
    return energy_weighted_eta 

def CalculatePhiFromCells(all_cells,desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # phi, (absolute) energy weighted value
    
    if desired_cells is None:
        return np.nan
    energy_weighted_phi = weighted_circular_mean(desired_cells['cell_phi'],np.abs(desired_cells['cell_E']))
    return energy_weighted_phi 

def CalculateEtFromCells(all_cells,desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # E_T, Transverse energy
    
    if desired_cells is None:
        return np.nan
    total_energy = sum(desired_cells['cell_E']) 
    energy_weighted_eta = np.dot(desired_cells['cell_eta'],np.abs(desired_cells['cell_E'])) / sum(np.abs(desired_cells['cell_E']))
    return total_energy / np.cosh(energy_weighted_eta) 

def CalculateNCellsFromCells(all_cells,desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # Number of cells in the box/cluster

    if desired_cells is None:
        return np.nan
    return len(desired_cells) #if cell_ids is not None else np.nan

def CalculateNegativeEnergyFromCells(all_cells,desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # fraction of total energy that is negative

    if desired_cells is None:
        return np.nan
    # total_energy = sum(abs(desired_cells['cell_E'])) 
    negative_energy_cells = desired_cells[desired_cells['cell_E']<0]['cell_E']
    positive_energy_cells = desired_cells[desired_cells['cell_E']>0]['cell_E']

    return abs(sum(negative_energy_cells)) / sum(positive_energy_cells) 

def CalculateMaxEnergyFracFromCells(all_cells,desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # fraction of total energy in one cell
    
    if desired_cells is None:
        return np.nan
    total_energy = sum(desired_cells['cell_E'])
    return max(desired_cells['cell_E']) / total_energy 

def CalculateNoiseFromCells(all_cells,desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # total noise of the cluster/box (see TC paper p. 26)
    
    if desired_cells is None:
        return np.nan
    total_noise = np.sqrt(sum(desired_cells['cell_Sigma']**2))
    return total_noise

def CalculateSignficanceFromCells(all_cells,desired_cells):
    # Inputs
    # cell_ids, list of cell ids that we want to calculate 
    # cells, structured array containing all cells in event ['cell_*'] properties
    # Outputs
    # total significance of cells

    if desired_cells is None:
        return np.nan    
    total_energy = sum(desired_cells['cell_E']) 
    total_noise = np.sqrt(sum(desired_cells['cell_Sigma']**2))

    return total_energy / total_noise



# def GetPhysicsFromCells(all_cell_ids, cells):
#     # Inputs
#     # all_boxes, array of boxes we would like to find cell Ids for and return
#     # cells, structured array containing all cells in THIS event ['cell_*'] properties
#     # Multiprocessing for speed
    
#     num_workers = 4 #multiprocessing.cpu_count()
#     chunk_size = 1 if len(all_cell_ids) // num_workers==0 else len(all_cell_ids) // num_workers
#     pool = multiprocessing.Pool(processes=num_workers)
#     results = []
#     for j in range(0,len(all_cell_ids),chunk_size):
#         mp_chunk = all_cell_ids[j:j+chunk_size]
#         chunk_results = pool.starmap(CalculateEnergyFromCells, [(cells, cids) for cids in mp_chunk])
#         results.extend(chunk_results)
#     pool.close()
#     pool.join()
#     return results

def get_physics_from_cells(all_cell_ids, cells, target='energy'):
    # functions = [CalculateEnergyFromCells, CalculateEtaFromCells, CalculatePhiFromCells, CalculateNCellsFromCells]
    # results = [[] for _ in functions]   
    # for idx in range(len(all_cell_ids)):
    #     for j, func in enumerate(functions):
    #         results[j].append(func(cells,all_cell_ids[idx]))

    if target == 'energy':
        return [CalculateEnergyFromCells(cells,ids_i) for ids_i in all_cell_ids]

    if target  == 'eta':
        return [CalculateEtaFromCells(cells,ids_i) for ids_i in all_cell_ids]
    
    if target == 'phi':
        return [CalculatePhiFromCells(cells,ids_i) for ids_i in all_cell_ids]
    
    if target in ['eT','et','e_T','E_T']:
        return [CalculateEtFromCells(cells,ids_i) for ids_i in all_cell_ids]
    
    if target == 'n_cells':
        return [CalculateNCellsFromCells(cells,ids_i) for ids_i in all_cell_ids]

    if target == 'neg_frac':
        return [CalculateNegativeEnergyFromCells(cells,ids_i) for ids_i in all_cell_ids]

    if target == 'max_frac':
        return [CalculateMaxEnergyFracFromCells(cells,ids_i) for ids_i in all_cell_ids]

    if target in ['noise','sigma','total_noise']:
        return [CalculateNoiseFromCells(cells,ids_i) for ids_i in all_cell_ids]

    if target == 'significance':
        return [CalculateSignficanceFromCells(cells,ids_i) for ids_i in all_cell_ids]


def get_physics_dictionary(all_cell_ids, cells):
    output_dict = {
        'energy'       : [CalculateEnergyFromCells(cells,ids_i) for ids_i in all_cell_ids],
        'eta'          : [CalculateEtaFromCells(cells,ids_i) for ids_i in all_cell_ids],
        'phi'          : [CalculatePhiFromCells(cells,ids_i) for ids_i in all_cell_ids],
        'eT'           : [CalculateEtFromCells(cells,ids_i) for ids_i in all_cell_ids],
        'n_cells'      : [CalculateNCellsFromCells(cells,ids_i) for ids_i in all_cell_ids],
        'noise'        : [CalculateNoiseFromCells(cells,ids_i) for ids_i in all_cell_ids],
        'significance' : [CalculateSignficanceFromCells(cells,ids_i) for ids_i in all_cell_ids],
        'neg_frac'     : [CalculateNegativeEnergyFromCells(cells,ids_i) for ids_i in all_cell_ids],
        'max_frac'     : [CalculateMaxEnergyFracFromCells(cells,ids_i) for ids_i in all_cell_ids],
    }

    return output_dict





































if __name__=="__main__":


    eta_array = np.array([1.5,1.4,1.3,1.2])
    phi_array = np.array([-3.1,-3.05,-2.9,3.1])
    box = [1.1,-3.14,1.6,-2.85]


    is_cluster_enclosed_in_box()















    def circular_mean(angles):
        """
        Calculate the circular mean (average) of a list of angles.
        Handles the periodicity of angles correctly.
        
        :param angles: List of angles in radians
        :return: Circular mean in radians
        """
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        circular_mean = np.arctan2(sin_sum, cos_sum)
        return circular_mean

    # Example usage:
    phi_values = [0.2, 3.0, 2.8, -2.9, 3.1, -3.05, -3.15, 2.9]
    average_phi = circular_mean(phi_values)
    print(f"Average phi: {average_phi}")
    phi_values = [3.1, -3.1, 2.8, -2.8, 3.0, -3.0]
    average_phi = circular_mean(phi_values)
    print(f"Average phi: {average_phi}",np.pi)

    
    energy_values = [2.0, 4.0, 2.0, 4.0, 2.0, 4.0]
    weighted_average_phi = weighted_circular_mean(phi_values, energy_values)
    print(f"Weighted Average phi: {weighted_average_phi}") 


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
















