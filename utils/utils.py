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
    boxes = torch.tensor(boxes)
    suppress = np.zeros(len(boxes))
    for i in range(len(boxes)):
        #needs further investigation
        box_i = boxes[i]

        if (box_i[1] < ymin) or (box_i[3] > ymax):
            #compare to the "un"wrapped box
            modded_box_i = box_i + (-1*torch.sign(box_i[1])) * torch.tensor([0.0, 2*np.pi, 0.0, 2*np.pi])
            overlaps = torchvision.ops.box_iou(modded_box_i.unsqueeze(0), boxes)
            #if overlap too much, take that with the greater confidence
            print('torch max overlaps',torch.max(overlaps),box_i,scores[i],scores[torch.argmax(overlaps)])
            if torch.max(overlaps) > threshold:
                wrapped_guy = torch.argmax(overlaps)
                suppress[i] = max(suppress[i],scores[i]<scores[wrapped_guy])

    boxes = boxes.numpy()
    return boxes[np.where(suppress==0)]


def wrap_check_truth(boxes,ymin,ymax):
    #here we look at truth boxes, remove the (wrapped) "duplicates" 
    #and mitigate those crossing the discontinuity
    #input is a np.ndarray containing the boxes in xyxy coords, after multiplication of extent

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
        #need a check that the corners are inside the true extent:
        #here's where we need to break boxes in 2
        if (box_i[1] < ymin):
            modded_box_i = box_i + np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            conditon1 = np.logical_and.reduce((cells['cell_eta']>box_i[0], cells['cell_eta']<box_i[2],cells['cell_phi']>ymin,cells['cell_phi']<box_i[3]))
            conditon2 = np.logical_and.reduce((cells['cell_eta']>modded_box_i[0], cells['cell_eta']<modded_box_i[2],cells['cell_phi']>modded_box_i[1],cells['cell_phi']<ymax))
            tot_cond = np.logical_and(conditon1,conditon2)
            cells_here = cells[np.where(condition)]
            print(box_i)
            print('!',len(cells_here))

        elif (box_i[3] > ymax):
            modded_box_i = box_i - np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            conditon1 = np.logical_and.reduce((cells['cell_eta']>box_i[0], cells['cell_eta']<box_i[2],cells['cell_phi']>box_i[1],cells['cell_phi']<ymax))
            conditon2 = np.logical_and.reduce((cells['cell_eta']>modded_box_i[0], cells['cell_eta']<modded_box_i[2],cells['cell_phi']>ymin,cells['cell_phi']<modded_box_i[3]))
            tot_cond = np.logical_and(conditon1,conditon2)
            cells_here = cells[np.where(condition)]
            print(box_i)
            print('!',len(cells_here))

        else:
            condition = np.logical_and.reduce((cells['cell_eta']>box_i[0], cells['cell_eta']<box_i[2], cells['cell_phi']>box_i[1], cells['cell_phi']>box_i[3])) #multiple conditions #could use np.all(x,axis)
            cells_here = cells[np.where(condition)]
            print(box_i)
            print(len(cells_here))
    
        list_o_cells.append(cells_here)

    return list_o_cells





