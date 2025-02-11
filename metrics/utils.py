import torch
import torchvision
import numpy as np
import scipy



MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
# MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496
MIN_CELLS_ETA,MAX_CELLS_ETA = -2.5, 2.5
# EXTENT = [-4.82349586, 4.82349586, -6.21738815, 6.21801758] 
EXTENT = (-2.4999826, 2.4999774, -6.217388274177672, 6.2180176992265)



def iou_box_matching(tees, pees, iou_thresh=0.5):
    # find the indices of the truth/predicted boxes that are matched to each other
    # using hungarian algorithm, linear_sum_assignment based on IoU score
    # returns two tensors, 1 if this target/pred box is matched, 0 if not
    # tees and pees in xyxy coordinates

    iou_scores = -torchvision.ops.box_iou(tees,pees)

    # Find the optimal one-to-one matching between ground truth and predicted boxes.
    gt_indices, pred_indices = scipy.optimize.linear_sum_assignment(iou_scores)
    gt_indices = torch.tensor(gt_indices)
    pred_indices = torch.tensor(pred_indices)

    # print('scores',iou_scores[gt_indices, pred_indices])
    good_scores = iou_scores[gt_indices, pred_indices] < -iou_thresh
    good_gt_indices = gt_indices[good_scores]
    good_pred_indices = pred_indices[good_scores]
    # print(good_scores)
    # print(good_gt_indices,good_pred_indices)
    # print(t_pt[good_gt_indices])
    # print(p_pt[good_pred_indices])

    mod_iou_scores = -torchvision.ops.box_iou(tees, pees + (-1*torch.sign(pees[:,1]).unsqueeze(1)) * torch.tensor([0.0,2*torch.pi,0.0,2*torch.pi]).unsqueeze(0))
    gt_mod_indices, pred_mod_indices = scipy.optimize.linear_sum_assignment(mod_iou_scores)
    gt_mod_indices = torch.tensor(gt_mod_indices)
    pred_mod_indices = torch.tensor(pred_mod_indices)
    good_mod_scores = mod_iou_scores[gt_mod_indices, pred_mod_indices] < -iou_thresh
    # print()
    # print(good_mod_scores)
    good_mod_gt_indices = gt_mod_indices[good_mod_scores]
    good_mod_pred_indices = pred_mod_indices[good_mod_scores]
    # print(good_mod_gt_indices,good_mod_pred_indices)
    # print(t_pt[good_mod_gt_indices])
    # print(p_pt[good_mod_pred_indices])
    # print()
    # print(torch.cat((good_pred_indices,good_mod_pred_indices)))
    good_total_gt_indices = torch.cat((good_gt_indices,good_mod_gt_indices))
    good_total_pred_indices = torch.cat((good_pred_indices,good_mod_pred_indices))

    if torch.isin(good_pred_indices, good_mod_pred_indices).any():
        print("PT     indicies…", good_pred_indices)
        print("GT     indicies…", good_gt_indices)
        print("PT mod indicies…", good_mod_pred_indices)
        print("GT mod indicies…", good_mod_gt_indices)
        print("PT tot indixes...",good_total_pred_indices)
        print("GT tot indixes...",good_total_gt_indices)
        print()


        print("repeated elements (when a box matches twice both initially and once when modded)")
        common_elements_pred = torch.isin(good_pred_indices, good_mod_pred_indices)
        common_elements_mod_pred = torch.isin(good_mod_pred_indices,good_pred_indices)
        print(common_elements_pred,common_elements_mod_pred)
        idx_pred_common = good_pred_indices[common_elements_pred]
        idx_mod_pred_common = good_mod_pred_indices[common_elements_mod_pred]
        idx_gt_common = good_gt_indices[common_elements_pred]
        idx_mod_gt_common = good_mod_gt_indices[common_elements_mod_pred]
        
        iou_common = -iou_scores[good_gt_indices[common_elements_pred], good_pred_indices[common_elements_pred]]
        iou_mod_common = -mod_iou_scores[good_mod_gt_indices[common_elements_mod_pred], good_mod_pred_indices[common_elements_mod_pred]]
        # print(idx_gt_common,idx_pred_common,iou_common)
        # print(idx_mod_gt_common,idx_mod_pred_common,iou_mod_common)
        # print("which match is better? original or modded?")
        mask_pred = iou_common < iou_mod_common
        mask_mod_pred = ~mask_pred

        get_rid_of_the_worst_pred_duplicates = torch.isin(good_total_pred_indices, idx_pred_common[mask_pred])
        # print(get_rid_of_the_worst_pred_duplicates)
        get_rid_of_the_worst_gt_duplicates = torch.isin(good_total_gt_indices, idx_gt_common[mask_pred])
        # print(get_rid_of_the_worst_gt_duplicates)
        combined_get_rid_of = get_rid_of_the_worst_pred_duplicates & get_rid_of_the_worst_gt_duplicates
        
        get_rid_of_the_worst_pred_mod_duplicates = torch.isin(good_total_pred_indices, idx_mod_pred_common[mask_mod_pred])
        # print(get_rid_of_the_worst_pred_mod_duplicates)
        get_rid_of_the_worst_gt_mod_duplicates = torch.isin(good_total_gt_indices, idx_mod_gt_common[mask_mod_pred])
        # print(get_rid_of_the_worst_gt_mod_duplicates)
        combined_mod_get_rid_of = get_rid_of_the_worst_pred_mod_duplicates & get_rid_of_the_worst_gt_mod_duplicates
        # print(combined_get_rid_of)
        # print(combined_mod_get_rid_of)
        final_get_rid_combined_mask = combined_get_rid_of | combined_mod_get_rid_of
        print(final_get_rid_combined_mask)

        print("Fixed PT tot indixes...",good_total_pred_indices[~final_get_rid_combined_mask], "non-duplicate values")
        print("Fixed GT tot indixes...",good_total_gt_indices[~final_get_rid_combined_mask], "non-duplicate values")
        good_total_gt_indices = good_total_gt_indices[~final_get_rid_combined_mask]
        good_total_pred_indices = good_total_pred_indices[~final_get_rid_combined_mask]


    # returns tensor of matched gt and pred pts in order
    return good_total_gt_indices, good_total_pred_indices


def dR_box_matching(t_centre_x, t_centre_y, p_centre_x, p_centre_y, dR_thresh=0.2):
    
    t_centres = torch.stack((t_centre_x, t_centre_y),dim=1)
    p_centres = torch.stack((p_centre_x, p_centre_y),dim=1)

    # dR_mat = torch.cdist(t_centres, p_centres, p=2)
    # print(dR_mat)
    # print()

    diff = t_centres[:, None, :] - p_centres[None, :, :]  # Broadcasting to get the difference matrix
    diff[:, :, 1] = torch.fmod(diff[:, :, 1] + torch.pi, 2 * torch.pi) - torch.pi # Handle the y coordinate difference modulo 2*pi
    distances = torch.norm(diff, dim=2)
    # print(distances)
    
    _, matched_pred_indices = distances.min(dim=1)

    min_distances = distances[torch.arange(len(t_centres)),matched_pred_indices]
    # print("Min distances",min_distances)
    dR_mask = min_distances < dR_thresh
    good_match_gt_indices = torch.arange(len(t_centres))[dR_mask]
    good_match_pred_indices = matched_pred_indices[dR_mask]
    # print(good_match_gt_indices)
    # print(good_match_pred_indices)

    return good_match_gt_indices, good_match_pred_indices








def target_box_matching(gt_boxes, pr_boxes, iou_thresh=0.5):
    # find the indices of the truth/predicted boxes that are matched to each other
    # using hungarian algorithm, linear_sum_assignment based on IoU score
    # returns two tensors, 1 if this target/pred box is matched, 0 if not

    iou_scores = -torchvision.ops.box_iou(gt_boxes,pr_boxes)

    # Find the optimal one-to-one matching between ground truth and predicted boxes.
    gt_indices, pred_indices = scipy.optimize.linear_sum_assignment(iou_scores)
    gt_indices = torch.tensor(gt_indices)
    pred_indices = torch.tensor(pred_indices)

    #Count the number of matches that exceed the IoU threshold.
    total_matched_gt = (iou_scores[gt_indices, pred_indices] < -iou_thresh)
    # print('scores',iou_scores[gt_indices, pred_indices])
    # print(gt_indices,pred_indices)

    matched_pr_boxes = torch.zeros(len(pr_boxes))
    matched_gt_boxes = torch.zeros(len(gt_boxes))
    matched_gt_boxes[gt_indices[total_matched_gt]] = 1
    matched_pr_boxes[pred_indices[total_matched_gt]] = 1

    # do the same with modded boxes
    mod_iou_scores = -torchvision.ops.box_iou(gt_boxes, pr_boxes + (-1*torch.sign(pr_boxes[:,1]).unsqueeze(1)) * torch.tensor([0.0,2*torch.pi,0.0,2*torch.pi]).unsqueeze(0))
    gt_mod_indices, pred_mod_indices = scipy.optimize.linear_sum_assignment(mod_iou_scores)
    # print('mod scores',mod_iou_scores[gt_mod_indices, pred_mod_indices])
    total_matched_mod_gt = (mod_iou_scores[gt_mod_indices, pred_mod_indices] < -iou_thresh)
    # print('gt_mod_indices',gt_mod_indices, 'pred_mod_indices',pred_mod_indices)
    # print('total_matched_mod_gt',total_matched_mod_gt)
    gt_mod_indices = torch.tensor(gt_mod_indices)
    pred_mod_indices = torch.tensor(pred_mod_indices)
    # print('gt_mod_indices[total_matched_mod_gt]',gt_mod_indices[total_matched_mod_gt])
    matched_mod_pr_boxes = torch.zeros(len(pr_boxes))
    matched_mod_gt_boxes = torch.zeros(len(gt_boxes))
    matched_mod_gt_boxes[gt_mod_indices[total_matched_mod_gt]] = 1
    matched_mod_pr_boxes[pred_mod_indices[total_matched_mod_gt]] = 1

    gt_box_match = (matched_gt_boxes.to(dtype=torch.int32) | matched_mod_gt_boxes.to(dtype=torch.int32)).float()
    pr_box_match = (matched_pr_boxes.to(dtype=torch.int32) | matched_mod_pr_boxes.to(dtype=torch.int32)).float()
    # returns 1 or 0 if the truth box is matched or not 
    return gt_box_match, pr_box_match





def wrap_check_NMS3(pred_boxes, pred_scores, pred_pts, iou_thresh=0.5):
    # if a prediction overlaps with another prediction 
    # %2pi then remove the one with less confidence
    # pred_boxes in x1,y1,x2,y2
    #this version includes predicted pt (for the sumpool ofc)

    pred_boxes = torch.tensor(pred_boxes) if type(pred_boxes)==np.ndarray else pred_boxes
    pred_scores = torch.tensor(pred_scores) if type(pred_scores)==np.ndarray else pred_scores
    pred_pts = torch.tensor(pred_pts) if type(pred_pts)==np.ndarray else pred_pts

    wrapping = torch.zeros_like(pred_boxes)
    wrapping[:,1] = torch.ones(len(pred_boxes))*2*torch.pi
    wrapping[:,3] = torch.ones(len(pred_boxes))*2*torch.pi
    mod_boxes_plus = pred_boxes + wrapping
    
    # calculate IoU with +2pi boxes
    plus_ious = torchvision.ops.box_iou(pred_boxes, mod_boxes_plus)
    # find the original/wrapped pairs that interest
    mask = plus_ious > iou_thresh
    indices = torch.nonzero(mask, as_tuple=False)
    
    # isolate these orig/wrapped boxes
    orig_boxes = pred_boxes[indices[:,0]]
    orig_scores = pred_scores[indices[:,0]]
    wrap_boxes = mod_boxes_plus[indices[:,1]]
    wrap_scores = pred_scores[indices[:,1]]
    # is the original or the wrap better?
    orig_better = orig_scores >= wrap_scores
    wrap_better = orig_scores < wrap_scores
    #extract the indices we want to keep (implicitly suppress the losers)
    surviving_orig_boxes = indices[:,0][orig_better]
    surviving_wrap_boxes = indices[:,1][wrap_better]

    final_indices = torch.ones(len(pred_boxes))
    # set ALL the boxes considered to 0, then save only those we want
    final_indices[torch.flatten(indices)] = 0
    final_indices[surviving_orig_boxes] = 1
    final_indices[surviving_wrap_boxes] = 1

    final_boxes = pred_boxes[final_indices.bool()]
    final_scores = pred_scores[final_indices.bool()]
    final_pts = pred_pts[final_indices.bool()]

    mask = final_boxes[:, 3] < -torch.pi
    final_boxes[mask, 1] += 2*torch.pi
    final_boxes[mask, 3] += 2*torch.pi

    mask = final_boxes[:, 1] > torch.pi
    final_boxes[mask, 1] -= 2*torch.pi
    final_boxes[mask, 3] -= 2*torch.pi

    # new:
    # ONE FINAL NMS TO RULE THEM ALL
    final_nms_indices = torchvision.ops.nms(final_boxes, final_scores, iou_threshold=iou_thresh)

    return final_boxes[final_nms_indices], final_scores[final_nms_indices], final_pts[final_nms_indices]





def wrap_check_truth3(boxes,pts,ymin,ymax):
    #here we look at truth boxes, remove the (wrapped) "duplicates" 
    #and mitigate those crossing the discontinuity
    #input is a np.ndarray containing the boxes in xyxy coords, after multiplication of extent
    
    boxes = torch.tensor(boxes) if isinstance(boxes,np.ndarray) else boxes
    pts = torch.tensor(pts) if isinstance(pts,np.ndarray) else pts
    
    suppress = torch.zeros(len(boxes))
    for j in range(len(boxes)):
        # print(j)
        box_j = boxes[j]
        # case (1) the truth box lies entirely outside the true phi range
        # if (box_j[1]>MAX_CELLS_PHI) or (box_j[3]<MIN_CELLS_PHI):
        #     suppress[j] = 1

        # the box is entirely inside
        # if (box_j[3]<MAX_CELLS_PHI) and (box_j[1]>MIN_CELLS_PHI):
        #     print('\t',1)
        #     modded_box_j = box_j + (-1*torch.sign(box_j[1])) * torch.tensor([0.0, 2*torch.pi, 0.0, 2*torch.pi])
        #     overlaps = torchvision.ops.box_iou(modded_box_j.unsqueeze(0), boxes)
        #     wrapped_box = boxes[torch.argmax(overlaps)]
        #     suppress[torch.argmax(overlaps)] = 1.0
        
        # case (2) the truth box has two corners below the true phi range 
        if (box_j[1] < MIN_CELLS_PHI) and suppress[j]==0:
            modded_box_j = box_j + torch.tensor([0.0, 2*torch.pi, 0.0, 2*torch.pi])
            overlaps = torchvision.ops.box_iou(modded_box_j.unsqueeze(0), boxes)
            wrapped_box = boxes[torch.argmax(overlaps)]
            #select the one with more area in the image:
            original_overlap =  MIN_CELLS_PHI - box_j[3]
            modded_overlap = MAX_CELLS_PHI  - wrapped_box[1]
            suppress[j] = abs(original_overlap) < abs(modded_overlap)

        # case (3) the truth box has two corners above the true phi range 
        elif (box_j[3] > MAX_CELLS_PHI) and suppress[j]==0:
            modded_box_j = box_j - torch.tensor([0.0, 2*torch.pi, 0.0, 2*torch.pi])
            overlaps = torchvision.ops.box_iou(modded_box_j.unsqueeze(0), boxes)
            wrapped_box = boxes[torch.argmax(overlaps)]
            #select the one with more area in the image:
            original_overlap = MAX_CELLS_PHI - box_j[1]
            modded_overlap =  MIN_CELLS_PHI - wrapped_box[3]
            suppress[j] = abs(original_overlap) < abs(modded_overlap)

    return boxes[torch.where(suppress==0)], pts[torch.where(suppress==0)]
