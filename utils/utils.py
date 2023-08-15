import torch
import torchvision
import numpy as np


def sel_device(dev):
    """
    Returns a pytorch device given a string (or a device)
    - giving cuda or gpu will run a hardware check first
    """

    # Not from config, but when device is specified already
    if isinstance(dev, torch.device):
        return dev

    # Tries to get gpu if available
    if dev in ["cuda", "gpu"]:
        print("Trying to select cuda based on available hardware")
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Tries to get specific gpu
    elif "cuda" in dev:
        print(f"Trying to select {dev} based on available hardware")
        dev = dev if torch.cuda.is_available() else "cpu"

    print(f"Running on hardware: {dev}")
    return torch.device(dev)




def move_dev(
    tensor, 
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    MATTSTOOLS
    Returns a copy of a tensor on the targetted device. This function calls
    pytorch's .to() but allows for values to be a.

    - list of tensors
    - tuple of tensors
    - dict of tensors
    """

    # Select the pytorch device object if dev was a string
    if isinstance(dev, str):
        dev = sel_device(dev)

    if isinstance(tensor, tuple):
        #return tuple(t.to(dev) for t in tensor)
        return tuple(move_dev(t,dev) for t in tensor)
    elif isinstance(tensor, list):
        #return [t.to(dev) for t in tensor]
        return [move_dev(t,dev) for t in tensor]
    elif isinstance(tensor, dict):
        #return {t: tensor[t].to(dev) for t in tensor}
        return {t: move_dev(tensor[t],dev) for t in tensor}
    elif isinstance(tensor,str):
        return tensor
    else:
        return tensor.to(dev)
    

def remove_nan(array):
    #find the indices where there are not nan values
    good_indices = np.where(array==array) 
    return array[good_indices]



def transform_angle(angle):
    #maps angle to [-pi,pi]
    angle %= 2 * np.pi  # Map angle to [0, 2π]
    if angle >= np.pi:
        angle -= 2 * np.pi  # Map angle to [-π, π]
    return angle




def xyxy2cxcywh(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h)
    Input xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    Return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2, xy[:, 2:] - xy[:, :2]], dim=1)  

def cxcywh2xyxy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    Input cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    Return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], dim=1)  

def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).
    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).
    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.
    In the model, we are predicting bounding box coordinates in this encoded form.
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    # prior_widths = priors_cxcy[:,2]
    # prior_heights = priors_cxcy[:,3]
    # print('In the encode step\n Priors:', priors_cxcy.shape,torch.count_nonzero(prior_widths),torch.count_nonzero(prior_heights),'\n',priors_cxcy)
    # print('trues?',cxcy.shape,cxcy)
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h



def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.
    They are decoded into center-size coordinates.
    This is the inverse of the function above.
    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h








def wrap_check_NMS(boxes,scores,ymin,ymax,threshold=0.35):
    #we now need a function that removes duplicates, ie the boxes we put into the padded region
    #to resolve the discontinuity problem, we must have a maximum of 1 box for each REAL cluster

    #input is a np.ndarray containing the boxes in xyxy coords, after multiplication of extent
    if isinstance(boxes,np.ndarray):
        boxes = torch.tensor(boxes)
    suppress = np.zeros(len(boxes))
    for i in range(len(boxes)):
        #needs further investigation
        box_i = boxes[i]
        #compare to the "un"wrapped box
        modded_box_i = box_i + (-1*torch.sign(box_i[1])) * torch.tensor([0.0, 2*np.pi, 0.0, 2*np.pi])
        overlaps = torchvision.ops.box_iou(modded_box_i.unsqueeze(0), boxes)
        #if overlap too much, take that with the greater confidence
        if torch.max(overlaps) > threshold:
            wrapped_guy = torch.argmax(overlaps)
            suppress[i] = max(suppress[i],scores[i]<scores[wrapped_guy])

    boxes = boxes.numpy()
    boxes = boxes[np.where(suppress==0)]
    final_boxes = []
    for j in range(len(boxes)):
        box_j = boxes[j]
        if (box_j[1] < ymin) or (box_j[3] > ymax):
            modded_box_j = box_j + (-1*np.sign(box_j[1])) * np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            final_boxes.append(modded_box_j)
        else:
            final_boxes.append(box_j)
    return np.array(final_boxes)


def wrap_check_truth(boxes,ymin,ymax):
    #here we look at truth boxes, remove the (wrapped) "duplicates" 
    #and mitigate those crossing the discontinuity
    #input is a np.ndarray containing the boxes in xyxy coords, after multiplication of extent
    if isinstance(boxes,np.ndarray):
        boxes = torch.tensor(boxes)
    suppress = np.zeros(len(boxes))
    for j in range(len(boxes)):
        box_j = boxes[j]

        #case (A) the truth box lies entirely outside the true phi range
        if (box_j[1]>ymax) or (box_j[3]<ymin):
            suppress[j] = 1
        
        #case (B) the truth box has two corners outside the true phi range
        #check the IoU of the truth box with its duplicate, remove just one of these
        elif (box_j[1] < ymin) or (box_j[3] > ymax):
            modded_box_j = box_j + (-1*torch.sign(box_j[1])) * torch.tensor([0.0, 2*np.pi, 0.0, 2*np.pi])
            overlaps = torchvision.ops.box_iou(modded_box_j.unsqueeze(0), boxes)
            wrapped_box = boxes[torch.argmax(overlaps)]

            #keep the truth box with the largest area (can be different due to merging).
            suppress[j] = max(suppress[j],(box_j[2]-box_j[0])*(box_j[3]-box_j[1])<(wrapped_box[2]-wrapped_box[0])*(wrapped_box[3]-wrapped_box[1]))

    boxes = boxes.numpy()
    return boxes[np.where(suppress==0)]



def get_cells_from_boxes(boxes,cells):
    #boxes in xyxy
    ymin,ymax = min(cells['cell_phi']),max(cells['cell_phi'])

    list_o_cells = []
    for i in range(len(boxes)):
        box_i = boxes[i]
        x_condition = np.logical_and.reduce((cells['cell_eta']>box_i[0], cells['cell_eta']<box_i[2]))
        #need a check that the corners are inside the true extent:
        #here's where we need to break boxes in 2

        if (box_i[1] < ymin) and (box_i[3] > ymin):
            modded_box_i = box_i + np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            bottom_out = max(box_i[1],ymin)
            top_out = min(modded_box_i[3],ymax)
            y_condtion1 = np.logical_and.reduce((cells['cell_phi']>bottom_out,cells['cell_phi']<box_i[3]))
            y_condtion2 = np.logical_and.reduce((cells['cell_phi']>modded_box_i[1],cells['cell_phi']<top_out))
            y_cond = np.logical_or(y_condtion1,y_condtion2)

        
        elif (box_i[3] > ymax) and (box_i[1] < ymax):
            modded_box_i = box_i - np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            top_top = max(box_i[3],ymax)
            bottom_bottom = min(modded_box_i[1],ymin)
            y_condtion1 = np.logical_and.reduce((cells['cell_phi'] > box_i[1],cells['cell_phi'] < top_top))
            y_condtion2 = np.logical_and.reduce((cells['cell_phi'] > bottom_bottom,cells['cell_phi'] < modded_box_i[3]))
            y_cond = np.logical_or(y_condtion1,y_condtion2)


        elif (box_i[1] < ymin):
            modded_box_i = box_i + np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            y_cond = np.logical_and.reduce((cells['cell_phi']>modded_box_i[1], cells['cell_phi']>modded_box_i[3]))


        elif (box_i[3] > ymax):
            modded_box_i = box_i - np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            y_cond = np.logical_and.reduce((cells['cell_phi']>modded_box_i[1], cells['cell_phi']>modded_box_i[3]))


        else:
            y_cond = np.logical_and.reduce((cells['cell_phi']>box_i[1], cells['cell_phi']<box_i[3])) #multiple conditions #could use np.all(x,axis)
        
        tot_cond = np.logical_and(x_condition,y_cond)
        cells_here = cells[np.where(tot_cond)]
        list_o_cells.append(cells_here)

    return list_o_cells



def event_cluster_estimates(pred_boxes, scores, truth_boxes, cells, mode='match',target='energy'):
    #arguments are:
    #pred_boxes, model output, no augmentation/wrap checking [n_preds,4] NO WRAP CHECK
    #truth_boxes, from .npy file, [n_objs, 4]
    #mode, tells us whether we should look at matched predictions, unmatched, or all (None) ['match','unmatch']
    #target, cluster statistic to check ['energy','eta','phi']

    wc_pred_boxes = wrap_check_NMS(pred_boxes,scores,min(cells['cell_phi']),max(cells['cell_phi']),threshold=0.2)
    wc_truth_boxes = wrap_check_truth(truth_boxes,min(cells['cell_phi']),max(cells['cell_phi']))

    if mode=='match':
        iou_mat = torchvision.ops.boxes.box_iou(torch.tensor(wc_truth_boxes),torch.tensor(wc_pred_boxes))
        matched_vals, matches = iou_mat.max(dim=0)
        wc_truth_boxes = wc_truth_boxes[matches[np.nonzero(matched_vals)]].reshape(-1,4)
        wc_pred_boxes = wc_pred_boxes[np.nonzero(matched_vals)].reshape(-1,4)
    elif mode=='unmatch':
        iou_mat = torchvision.ops.boxes.box_iou(torch.tensor(wc_truth_boxes),torch.tensor(wc_pred_boxes))
        matched_vals, matches = iou_mat.max(dim=0)
        unmatched_idxs = np.where(matched_vals==0)
        wc_pred_boxes = wc_pred_boxes[unmatched_idxs].reshape(-1,4)
        wc_truth_boxes = np.delete(wc_truth_boxes,matches[np.nonzero(matched_vals)],axis=0)

    list_pred_cl_cells = get_cells_from_boxes(wc_pred_boxes,cells)
    list_tru_cl_cells = get_cells_from_boxes(wc_truth_boxes,cells)
    
    if target == 'energy':
        list_pred_cl_energies = [sum(x['cell_E']) for x in list_pred_cl_cells]
        list_tru_cl_energies = [sum(x['cell_E']) for x in list_tru_cl_cells]
        return list_pred_cl_energies, list_tru_cl_energies

    def calc_cl_eta(cl_array):
        return np.dot(cl_array['cell_eta'],np.abs(cl_array['cell_E'])) / sum(np.abs(cl_array['cell_E']))
    def calc_cl_phi(cl_array):
        return np.dot(cl_array['cell_phi'],np.abs(cl_array['cell_E'])) / sum(np.abs(cl_array['cell_E']))

    if target  == 'eta':
        list_pred_cl_etas = [calc_cl_eta(x) for x in list_pred_cl_cells]
        list_tru_cl_etas = [calc_cl_eta(x) for x in list_tru_cl_cells]
        return list_pred_cl_etas, list_tru_cl_etas
    
    if target == 'phi':
        list_pred_cl_phis = [calc_cl_phi(x) for x in list_pred_cl_cells]
        list_tru_cl_phis = [calc_cl_phi(x) for x in list_tru_cl_cells]
        return list_pred_cl_phis, list_tru_cl_phis
    
    if target == 'n_cells':
        list_pred_cl_ns = [len(x) for x in list_pred_cl_cells]
        list_tru_cl_ns = [len(x) for x in list_tru_cl_cells]
        return list_pred_cl_ns, list_tru_cl_ns  





