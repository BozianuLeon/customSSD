import torch
import torchvision
from data.utils import calc_iou_tensor

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
# MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496
MIN_CELLS_ETA,MAX_CELLS_ETA = -2.5, 2.5
# EXTENT = [-4.82349586, 4.82349586, -6.21738815, 6.21801758] 
EXTENT = (-2.4999826, 2.4999774, -6.217388274177672, 6.2180176992265)


class Encoder(object):
    """
        Inspired by https://github.com/kuangliu/pytorch-src
        Transform between (bboxes, lables) <-> SSD output

        dboxes: default boxes in size 8732 x 4,
            encoder: input ltrb format, output xywh format
            decoder: input xywh format, output ltrb format

        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes

        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
    """

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh
        self.figsize = dboxes.fig_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def encode(self, bboxes_in, labels_in, criteria=0.5):
        bboxes_in = bboxes_in.to(self.dboxes.device)
        labels_in = labels_in.to(self.dboxes.device)

        bboxes_in[:,(0,2)] = bboxes_in[:,(0,2)] / 49 
        bboxes_in[:,(1,3)] = bboxes_in[:,(1,3)] / 125

        ious = calc_iou_tensor(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)        
        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_dbox_ious > criteria

        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]

        # Transform format to xywh format
        x, y, w, h = 0.5*(bboxes_out[:, 0] + bboxes_out[:, 2]), \
                     0.5*(bboxes_out[:, 1] + bboxes_out[:, 3]), \
                     -bboxes_out[:, 0] + bboxes_out[:, 2], \
                     -bboxes_out[:, 1] + bboxes_out[:, 3]

        bboxes_out[:, 0] = x
        bboxes_out[:, 1] = y
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h
        return bboxes_out, labels_out

    def scale_back_batch(self, bboxes_in, scores_in, pt_array_in):
        """
            Do scale and transform from xywh to ltrb
            suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        # Unparameterise p.5 SSD
        # NEED TO UPDATE scale_xy, scale_wh
        bboxes_in[:, :, :2] = self.scale_xy*bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh*bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp()*self.dboxes_xywh[:, :, 2:]

        # Find prediction centres, extract pT from pT array
        centres = bboxes_in[:,:,:2]
        centres_x = (centres[:,:,0]*((EXTENT[1]-EXTENT[0])))+EXTENT[0]
        centres_y = (centres[:,:,1]*((EXTENT[3]-EXTENT[2])))+EXTENT[2]
        
        bins_x = torch.linspace(MIN_CELLS_ETA, MAX_CELLS_ETA, int((MAX_CELLS_ETA - MIN_CELLS_ETA) / 0.1 + 1))
        wrapped_bins_y = torch.linspace(EXTENT[2], EXTENT[3], int((EXTENT[3] - EXTENT[2]) / ((2*torch.pi)/64) + 1))
 
        batch_idxs = torch.arange(pt_array_in.shape[0]).unsqueeze(1).expand(pt_array_in.shape[0], self.nboxes)  # Shape [4, n_dfboxes]
        cx_idxs = torch.bucketize(centres_x, bins_x.to(self.device))
        cy_idxs = torch.bucketize(centres_y, wrapped_bins_y.to(self.device))

        cx_idxs = torch.clamp(cx_idxs, max=pt_array_in.shape[2]-1)
        cy_idxs = torch.clamp(cy_idxs, max=pt_array_in.shape[1]-1)

        pts_in = pt_array_in[batch_idxs, cy_idxs, cx_idxs] #[b,y,x]
        pts_in = pts_in.unsqueeze(-1)


        # for i in range(97):
        #     print(pt_array_in[0,i,:])
        #     print("\n")
        # pt_array_0 = pt_array_in[0].cpu().detach().numpy()
        # f,ax = plt.subplots(1,1,figsize=(20,36))   
        # im = ax.imshow(pt_array_0,cmap='binary_r', origin='lower')
        # f.colorbar(im, ax=ax, orientation='vertical')
        # for i in range(pt_array_0.shape[0]):
        #     for j in range(pt_array_0.shape[1]):
        #         ax.text(j, i, f'{pt_array_0[i, j]:.0f}',va='center',ha='center',fontsize=0.1,color='red')
        # f.savefig(f'pt_array_0.png',dpi=500)
        # plt.close()

        # Transform format to ltrb
        l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                     bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b

        return bboxes_in, torch.nn.functional.softmax(scores_in, dim=0), pts_in

    def decode_batch(self,bboxes_in,scores_in,pt_array_in,iou_thresh,confidence,max_num):#CRITERIA IS NMS CRITERIA AND SO LOWER LESS BOXES (0.45)
        bboxes, probs, pts = self.scale_back_batch(bboxes_in, scores_in, pt_array_in)

        output = []
        for bbox, prob, pt in zip(bboxes.split(1, 0), probs.split(1, 0), pts.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)            
            pt = pt.squeeze(0)            
            output.append(self.decode_single(bbox, prob, pt, iou_thresh, confidence, max_num))

        return output

     # perform non-maximum suppression    
    def decode_single(self, bboxes_in, scores_in, pts_in, iou_thresh, confidence, max_num=200): 
        # first get rid of low confidence boxes
        mask = scores_in > confidence

        bboxes, score, pts = bboxes_in[mask.squeeze(1),:], scores_in[mask], pts_in[mask]
        if score.size(0) == 0:
            raise IndexError("No scores passed chosen threshold")

        # second, if there are more than max_num passing, cut predictions
        score_sorted, score_idx_sorted = score.sort(dim=0)
        score_idx_sorted = score_idx_sorted[-max_num:]
        bboxes_sorted = bboxes[score_idx_sorted, :]
        scores_sorted = score[score_idx_sorted]
        pts_sorted = pts[score_idx_sorted]

        # use torchvision native NMS
        nms_indices = torchvision.ops.nms(bboxes_sorted, scores_sorted, iou_threshold=iou_thresh)

        return bboxes_sorted[nms_indices], torch.ones(len(nms_indices)), scores_sorted[nms_indices], pts_sorted[nms_indices]





if __name__=="__main__":

    # pytorch tensor [5,125,49]
    # figsize = (49,125)
    figsize = (24,63) # (96,125)
    step_x, step_y = 1,1
    from defboxes import DefaultBoxes

    dboxes = DefaultBoxes(figsize, step_x, step_y)
    print('Number of dboxes: ', dboxes.dboxes.shape)




