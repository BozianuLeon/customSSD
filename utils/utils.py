import torch
import torchvision


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


def find_jaccard_overlap(boxes1,boxes2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    return torchvision.ops.box_iou(boxes1,boxes2)



def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.
    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.
    Input:
      tensor: tensor to be decimated
      list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    Return:
      decimated tensor
    """

    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor




