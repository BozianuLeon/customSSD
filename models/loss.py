import torch
import torch.nn as nn


class Loss(torch.nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes, scalar=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Loss, self).__init__()
        self.scale_xy = 1.0/dboxes.scale_xy
        self.scale_wh = 1.0/dboxes.scale_wh
        self.scalar = scalar

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
        # fcloss = torchvision.ops.focal_loss.sigmoid_focal_loss(plabel,glabel,alpha=0.5,gamma=4,reduction='none').sum(dim=1)
        # fcloss = fcloss / pos_num # normalise by the number of anchors assigned to GT boxes
        # total_loss = sl1 + self.scalar*fcloss  

        total_loss = sl1 + closs  
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss*num_mask/pos_num).mean(dim=0)
        
        return ret
    


