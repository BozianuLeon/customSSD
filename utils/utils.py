import torch
import torchvision
import numpy as np
import scipy
from itertools import compress


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

def phi_mod2pi(phis):
    repeated_phis = np.copy(phis)
    mask = repeated_phis >= 0

    repeated_phis[mask] -= 2*np.pi
    repeated_phis[~mask] += 2*np.pi
    return repeated_phis


#########################################################################################################################################################
# Model Requirements
#########################################################################################################################################################
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



#########################################################################################################################################################
# Making cell images / dataset
#########################################################################################################################################################
def clip_boxes_to_image(boxes, extent):
    #https://detectron2.readthedocs.io/en/latest/_modules/torchvision/ops/boxes.html
    boxes = torchvision.ops.box_convert(boxes,'xywh','xyxy')

    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    xmin,xmax,ymin,ymax = extent

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(xmin, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(xmax, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(ymin, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(ymax, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=xmin, max=xmax)
        boxes_y = boxes_y.clamp(min=ymin, max=ymax)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    clipped_boxes = clipped_boxes.reshape(boxes.shape)
    #ensure that the new clipped boxes satisfy height requirements
    #remember here we're in xyxy coords
    heights = (clipped_boxes[:,3]-clipped_boxes[:,1])
    final_boxes_xyxy = clipped_boxes[heights>0.1]
    final_boxes = torchvision.ops.box_convert(final_boxes_xyxy, 'xyxy','xywh')
    return final_boxes




def union(boxa,boxb):
    x = min(boxa[0], boxb[0])
    y = min(boxa[1], boxb[1])
    w = max(boxa[0]+boxa[2], boxb[0]+boxb[2]) - x
    h = max(boxa[1]+boxa[3], boxb[1]+boxb[3]) - y
    return (x,y,w,h)

def intersection(boxa,boxb):
    x = max(boxa[0], boxb[0])
    y = max(boxa[1], boxb[1])
    w = min(boxa[0]+boxa[2], boxb[0]+boxb[2]) - x
    h = min(boxa[1]+boxa[3], boxb[1]+boxb[3]) - y
    if w<0 or h<0:
        return ()
    return (x,y,w,h)



def merge_rectangles(boxes, max_size=(1.5, 1.5)):

    def contains(rect_a, rect_b):
        x1a, y1a, w1a, h1a = rect_a
        x1b, y1b, w1b, h1b = rect_b
        x2a, y2a = x1a + w1a, y1a + h1a
        x2b, y2b = x1b + w1b, y1b + h1b
        return x1a <= x1b and y1a <= y1b and x2a >= x2b and y2a >= y2b

    new_boxes = list(boxes)  
    merged_boxes = []  
    failed_merge = []  

    while new_boxes:
        ra = new_boxes[0]  # Select the first rectangle for comparison
        new_boxes = new_boxes[1:]  # Remove the first rectangle from the list

        mask = [intersection(ra, rb) != () for rb in new_boxes]
        merged_mask = not any(mask)  # True if ra has no intersections with any remaining rectangles

        if not merged_mask:
            rb_idx = mask.index(True)  # Find the index of the first intersecting rectangle (rb)
            rb = new_boxes[rb_idx]  # Select the intersecting rectangle

            new_boxes = new_boxes[:rb_idx] + new_boxes[rb_idx + 1:]  # Remove rb from the list of rectangles

            merged_box = union(ra, rb)  # Merge ra and rb
            if max_size is None or (merged_box[2] <= max_size[0] and merged_box[3] <= max_size[1]):
                new_boxes.append(merged_box)  # Append the merged rectangle if it meets the size constraint
            else:
                failed_merge.append(ra)  # Add ra and rb to the failed_merge list if they exceed max_size
                failed_merge.append(rb)
        else:
            merged_boxes.append(ra)  # Add ra to merged_boxes if no intersections were found

    # Combine merged_boxes and failed_merge before checking for rectangles contained by larger rectangles
    all_rectangles = np.concatenate((merged_boxes, failed_merge)) if failed_merge else merged_boxes

    # Check for rectangles entirely contained
    final_boxes = []
    for box in all_rectangles:
        if not any(contains(rb, box) for rb in all_rectangles if not np.array_equal(rb, box)):
            final_boxes.append(box)

    return final_boxes





#########################################################################################################################################################
# Interpreting inference
#########################################################################################################################################################

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
        eta_min,phi_min,eta_max,phi_max = box_i
        x_condition = np.logical_and.reduce((cells['cell_eta']>=eta_min, cells['cell_eta']<=eta_max))
        #need a check that the corners are inside the true extent:
        #here's where we need to break boxes in 2

        #box straddles bottom of image
        if (phi_min < ymin) and (phi_max > ymin):
            modded_box_i = box_i + np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            top_out = min(modded_box_i[3],ymax)
            y_condtion1 = np.logical_and.reduce((cells['cell_phi']>=ymin, cells['cell_phi']<=phi_max))
            y_condtion2 = np.logical_and.reduce((cells['cell_phi']>=modded_box_i[1], cells['cell_phi']<=top_out))
            y_cond = np.logical_or(y_condtion1,y_condtion2)

        #box straddles top of image
        elif (phi_max > ymax) and (phi_min < ymax):
            modded_box_i = box_i - np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            bottom_bottom = min(modded_box_i[1],ymin)
            y_condtion1 = np.logical_and.reduce((cells['cell_phi'] >= phi_min, cells['cell_phi'] <= ymax))
            y_condtion2 = np.logical_and.reduce((cells['cell_phi'] >= bottom_bottom, cells['cell_phi'] <= modded_box_i[3]))
            y_cond = np.logical_or(y_condtion1,y_condtion2)

        #box is completely above top
        elif (phi_max < ymin):
            modded_box_i = box_i + np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            y_cond = np.logical_and.reduce((cells['cell_phi']>=modded_box_i[1], cells['cell_phi']>=modded_box_i[3]))


        elif (phi_min > ymax):
            modded_box_i = box_i - np.array([0.0, 2*np.pi, 0.0, 2*np.pi])
            y_cond = np.logical_and.reduce((cells['cell_phi']>=modded_box_i[1], cells['cell_phi']>=modded_box_i[3]))

        else:
            y_cond = np.logical_and.reduce((cells['cell_phi']>=phi_min, cells['cell_phi']<=phi_max)) #multiple conditions #could use np.all(x,axis)
        
        tot_cond = np.logical_and(x_condition,y_cond)
        cells_here = cells[np.where(tot_cond)]
        #guard against boxes containing no cells!
        if not len(cells_here)==0:
            list_o_cells.append(cells_here)
        else:    
            #so that we know where there were no cells!
            placeholder_values = np.array([(-1, -1, -1.0, -1, -1, -1, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)],
                dtype=[('cell_BadCells', '<i4'), ('cell_DetCells', '<i4'), ('cell_E', '<f4'), ('cell_GainCells', '<i4'), ('cell_IdCells', '<u4'), ('cell_QCells', '<i4'), ('cell_Sigma', '<f4'), ('cell_TimeCells', '<f4'), ('cell_eta', '<f4'), ('cell_phi', '<f4'), ('cell_pt', '<f4'), ('cell_xCells', '<f4'), ('cell_yCells', '<f4'), ('cell_zCells', '<f4')])
            list_o_cells.append(placeholder_values)
            print('No cells in box!')

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
    for data in list_pred_cl_cells:
        if sum(data['cell_BadCells']) < 0:
            print(data)
    
    # Check that neither list using placeholder values has an entry with no cells
    #zero_cells_mask tells us that this box contains more than 0 cells
    pred_zero_cells_mask = [sum(data['cell_BadCells']) >= 0 for data in list_pred_cl_cells]
    list_pred_cl_cells = list(compress(list_pred_cl_cells, pred_zero_cells_mask))
    # list_tru_cl_cells = list(compress(list_tru_cl_cells, pred_zero_cells_mask))
    true_zero_cells_mask = [sum(data['cell_BadCells']) >= 0 for data in list_tru_cl_cells]
    # list_pred_cl_cells = list(compress(list_pred_cl_cells, true_zero_cells_mask))
    list_tru_cl_cells = list(compress(list_tru_cl_cells, true_zero_cells_mask))
    
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


#########################################################################################################################################################
# Useful for plotting
#########################################################################################################################################################

def make_image_using_cells(cells,channel=0,padding=True):
    #mirrors make_real_dataset.py, should return a pytorch tensor/numpy array 
    #specify channel number to select a particular slice of the [N,86,94] tensor
    cell_etas = cells['cell_eta']
    cell_phis = cells['cell_phi'] 
    cell_energy = cells['cell_E']
    cell_sigma = cells['cell_Sigma']    
    cell_time = cells['cell_TimeCells']   
    cell_q = cells['cell_QCells']    
    cell_bad = cells['cell_BadCells']    
    cell_Esig =  cell_energy / cell_sigma      

    EM_layers = [65,81,97,113,  #EM barrel
                257,273,289,305, #EM Endcap
                145,161, # IW EM
                2052] #EM FCAL

    HAD_layers = [2,514,1026,1538, #HEC layers
                4100,6148, #FCAL HAD
                65544, 73736,81928, #Tile barrel
                131080,139272,147464, #Tile endcap
                811016,278536,270344] #Tile gap  
    
    EM_indices = np.isin(cells['cell_DetCells'],EM_layers)
    HAD_indices = np.isin(cells['cell_DetCells'],HAD_layers)
    cell_etas_EM = cells['cell_eta'][EM_indices]
    cell_etas_HAD = cells['cell_eta'][HAD_indices]
    cell_phis_EM = cells['cell_phi'][EM_indices]
    cell_phis_HAD = cells['cell_phi'][HAD_indices]
    cell_E_EM = cells['cell_E'][EM_indices]
    cell_E_HAD = cells['cell_E'][HAD_indices]
    cell_Esig_EM = cell_E_EM / cells['cell_Sigma'][EM_indices]
    cell_Esig_HAD = cell_E_HAD / cells['cell_Sigma'][HAD_indices]

    bins_x = np.linspace(min(cell_etas), max(cell_etas), int((max(cell_etas) - min(cell_etas)) / 0.1 + 1))
    bins_y = np.linspace(min(cell_phis), max(cell_phis), int((max(cell_phis) - min(cell_phis)) / ((2*np.pi)/64) + 1))

    
    H_tot, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                        values=abs(cell_Esig),
                                                        bins=(bins_x,bins_y),
                                                        statistic='sum')

    H_em, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas_EM, cell_phis_EM,
                                                values=abs(cell_Esig_EM),
                                                bins=(bins_x,bins_y),
                                                statistic='sum')

    H_had, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas_HAD, cell_phis_HAD,
                                                values=abs(cell_Esig_HAD),
                                                bins=(bins_x,bins_y),
                                                statistic='sum')

    H_max, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                values=cell_Esig,
                                                bins=(bins_x,bins_y),
                                                statistic='max')

    H_mean, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                values=abs(cell_Esig),
                                                bins=(bins_x,bins_y),
                                                statistic='mean')

    H_sigma, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                values=cell_sigma,
                                                bins=(bins_x,bins_y),
                                                statistic='mean')

    H_energy, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                values=cell_energy,
                                                bins=(bins_x,bins_y),
                                                statistic='sum')

    H_time, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                values=cell_time,
                                                bins=(bins_x,bins_y),
                                                statistic='mean')

    H_bad, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                values=cell_bad,
                                                bins=(bins_x,bins_y),
                                                statistic='sum')
    #transpose to correct format/shape
    H_tot = H_tot.T
    H_em = H_em.T
    H_had = H_had.T
    H_max = H_max.T
    H_mean = H_mean.T
    H_sigma = H_sigma.T
    H_energy = H_energy.T
    H_time = H_time.T
    H_bad = H_bad.T
    if padding:
        repeat_frac = 0.5
        repeat_rows = int(H_tot.shape[0]*repeat_frac)
        one_box_height = (yedges[-1]-yedges[0])/H_tot.shape[0]

        # Padding
        H_tot  = np.pad(H_tot, ((repeat_rows,repeat_rows),(0,0)),'wrap')
        H_em   = np.pad(H_em,  ((repeat_rows,repeat_rows),(0,0)),'wrap')
        H_had  = np.pad(H_had, ((repeat_rows,repeat_rows),(0,0)),'wrap')
        H_max  = np.pad(H_max, ((repeat_rows,repeat_rows),(0,0)),'wrap')
        H_mean = np.pad(H_mean,((repeat_rows,repeat_rows),(0,0)),'wrap')
        H_sigma = np.pad(H_sigma,((repeat_rows,repeat_rows),(0,0)),'wrap')
        H_energy = np.pad(H_energy,((repeat_rows,repeat_rows),(0,0)),'wrap')
        H_time = np.pad(H_time,((repeat_rows,repeat_rows),(0,0)),'wrap')
        H_bad = np.pad(H_bad,((repeat_rows,repeat_rows),(0,0)),'wrap')

    #If total cell signficance in a cell exceeds a threshold truncate 
    truncation_threshold = 125
    H_tot = np.where(H_tot < truncation_threshold, H_tot, truncation_threshold)
    H_em = np.where(H_em < truncation_threshold, H_em, truncation_threshold)
    H_had = np.where(H_had < truncation_threshold, H_had, truncation_threshold)

    #Max cell significance in a pixel
    H_max[np.isnan(H_max)] = 0
    H_max = np.where(H_max < 4, 0, H_max) 
    H_max = np.where(H_max > 25, 25, H_max) 
    
    H_mean[np.isnan(H_mean)] = 0
    H_sigma[np.isnan(H_sigma)] = -1
    H_time[np.isnan(H_time)] = 0

    #treat the barrel and forward regions differently
    raw_energy_thresh = 1000
    central_H_energy = np.where(H_energy[:,10:-10]>raw_energy_thresh,H_energy[:,10:-10],0)
    left_edge_H_energy = np.where(H_energy[:,:10]>4*raw_energy_thresh,H_energy[:,:10]/4,0)
    right_edge_H_energy = np.where(H_energy[:,-10:]>4*raw_energy_thresh,H_energy[:,-10:]/4,0)
    H_energy = np.hstack((left_edge_H_energy,central_H_energy,right_edge_H_energy))

    H_layers = np.stack([H_tot,H_em,H_had,H_max,H_mean,H_sigma,H_energy,H_time],axis=0)
    
    return H_layers[channel,:,:]
