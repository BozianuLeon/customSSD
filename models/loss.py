import torch
import torch.nn as nn
import torchvision as tv


class Loss(torch.nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Loss, self).__init__()
        self.scale_xy = 1.0
        self.scale_wh = 1.0
        print("self.scale_xy,wh",self.scale_xy,self.scale_wh)

        self.con_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim = 0).to(device),requires_grad=False)
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """

        gxy = self.scale_xy*(loc[:, :2, :] - self.dboxes[:, :2, :])/self.dboxes[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        mask = glabel > 0 
        pos_num = mask.sum(dim=1) 
        vec_gd = self._loc_vec(gloc)
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        
        sl1 = (mask.float()*sl1).sum(dim=1)

        plabel = plabel.squeeze(1)
        plabel = plabel.float()
        glabel = glabel.float()
        con = self.con_loss(plabel, glabel)

        con_neg = con.clone() 
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        closs = (con*((mask + neg_mask).float())).sum(dim=1)

        total_loss = sl1 + closs  
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss*num_mask/pos_num).mean(dim=0)
        
        return ret
    

# Source: https://github.com/hyz-xmaster/VarifocalNet
def varifocal_loss(
    logits,
    labels,
    weight,
    alpha: float=0.75,
    gamma: float=2.0,
    iou_weighted: bool=True,
):
     
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`
 
    Args:
        logits (torch.Tensor): The model predicted logits with shape (N, C), 
        C is the number of classes
        labels (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
    """
    assert logits.size() == labels.size()
    logits_prob = logits.sigmoid()
    labels = labels.type_as(logits)
    if iou_weighted:
        focal_weight = labels * (labels > 0.0).float() + \
            alpha * (logits_prob - labels).abs().pow(gamma) * \
            (labels <= 0.0).float()
 
    else:
        focal_weight = (labels > 0.0).float() + \
            alpha * (logits_prob - labels).abs().pow(gamma) * \
            (labels <= 0.0).float()
 
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction='none') * focal_weight
    loss = loss * weight if weight is not None else loss
    return loss
 
 
class VariFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float=0.75,
        gamma: float=2.0,
        iou_weighted: bool=True,
        reduction: str='mean',
    ):
        # VariFocal Implementation: https://github.com/hyz-xmaster/VarifocalNet/blob/master/mmdet/models/losses/varifocal_loss.py
        super(VariFocalLoss, self).__init__()
        assert reduction in ('mean', 'sum', 'none')
        assert alpha >= 0.0
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.reduction = reduction
 
    def forward(self, logits, labels):
        loss = varifocal_loss(logits, labels, self.alpha, self.gamma, self.iou_weighted)
 
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss


def smooth_l1_loss(input, target, beta=1. / 9, size_average=False):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()





class NewLoss(torch.nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels taken into account, using focal loss with chosen alpha/gamma hyperparams
        See https://arxiv.org/pdf/1708.02002 or https://amaarora.github.io/posts/2020-06-29-FocalLoss.html
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes, scalar=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(NewLoss, self).__init__()
        self.scalar = scalar
        self.device = device

        # REG LOSS
        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.giou_loss = tv.ops.generalized_box_iou_loss
        self.ciou_loss = tv.ops.complete_box_iou_loss
        self.ciou_loss = tv.ops.distance_box_iou_loss
        self.dsl_loss = smooth_l1_loss

        # CLF LOSS
        self.con_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.foc_loss = tv.ops.focal_loss.sigmoid_focal_loss
        self.vfoc_loss = VariFocalLoss(alpha=0.25,gamma=3,reduction='sum')

        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim = 0).to(device),requires_grad=False)

    def xywh2xyxy(self, boxes, i=0):
        """
            Convert boxes xywh to xyxy 
        """
        boxes = boxes.clone()
        boxes[..., i:i + 2] -= boxes[..., i + 2:i + 4] / 2
        boxes[..., i + 2:i + 4] += boxes[..., i:i + 2]
        return boxes


    def _loc_vec(self, loc):
        """
            Parameterise Location Vectors
        """
        gxy = (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, ]
        gwh = torch.log(loc[:, 2:, :]/self.dboxes[:, 2:, :])
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """

        mask = glabel > 0 
        pos_num = mask.sum(dim=1) 
        num_mask = (pos_num > 0).float() # does the image contain any positive anchors
        # print(pos_num,glabel.sum(dim=1))

        # Box Regression Loss (Smooth L1)
        vec_gd = self._loc_vec(gloc)
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        # print(ploc.shape, vec_gd.shape,'Loss shape:',sl1.shape)
        sl1 = (mask.float()*sl1).sum(dim=1)
        # print("SL1 loss:       ", sl1)

        # Box Regression Loss (GIOU)
        gloss = torch.empty(0, device=self.device)
        for batch_id in range(gloc.shape[0]):
            pred = self.xywh2xyxy(ploc[batch_id].permute(1,0))
            tar = self.xywh2xyxy(gloc[batch_id].permute(1,0))
            loss_batch_i = self.giou_loss(pred, tar,reduction='sum')
            # print(loss_batch_i.shape,loss_batch_i)
            gloss = torch.cat((gloss, loss_batch_i.unsqueeze(0)), dim=-1)
        # print(gloss.shape)
        # print("GIOU loss:       ", gloss)

        plabel = plabel.squeeze(1).float()
        glabel = glabel.float()

        # Classification Loss (Focal)
        floss = self.foc_loss(plabel,glabel,alpha=0.25,gamma=3,reduction='sum')
        # normalise by the number of anchors assigned to a ground truth box
        floss = floss / pos_num
        # print("Focal loss:      ", floss)

        # fcloss = torchvision.ops.focal_loss.sigmoid_focal_loss(plabel,glabel,alpha=0.5,gamma=4,reduction='none').sum(dim=1)
        # fcloss = fcloss / pos_num # normalise by the number of anchors assigned to GT boxes
        # total_loss = sl1 + self.scalar*fcloss  

        # Classification Loss (VariFocal)
        vfloss = self.vfoc_loss(plabel,glabel)
        # normalise by the number of anchors assigned to a ground truth box
        vfloss = vfloss / pos_num
        # print("VariFocal loss:      ", vfloss)
        

        # Classification Loss (BCE)
        con = self.con_loss(plabel, glabel)
        con_neg = con.clone() 
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num
        closs = (con*((mask + neg_mask).float())).sum(dim=1)
        # print("BCE loss:      ", closs)

        loss_dict = {
            "SL1"   : (sl1*num_mask/pos_num).mean(dim=0),
            "GIOU"  : (gloss*num_mask/pos_num).mean(dim=0),
            "BCE"   : (closs*num_mask/pos_num).mean(dim=0),
            "FCL"   : (floss*num_mask/pos_num).mean(dim=0),
            "VFCL"  : (vfloss*num_mask/pos_num).mean(dim=0),
        }

        
        return loss_dict
    


