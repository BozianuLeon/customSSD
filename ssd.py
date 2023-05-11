import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import numpy as np
from math import sqrt


from models.models import VGGBase, AuxiliaryConvolutions, PredictionConvolutions
from utils.utils import cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy, xy_to_cxcy



class SSD(nn.Module):
    def __init__(self, n_classes=2,in_channels=3,device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),pretrained_vgg=True):
        super(SSD, self).__init__()
        self.device=device
        self.n_classes = n_classes
        self.in_channels = in_channels
        
        self.init_conv = nn.Conv2d(in_channels,out_channels=3,kernel_size=1,stride=1).to(device)
        self.base = VGGBase(pretrained=pretrained_vgg).to(device)
        self.aux_convs = AuxiliaryConvolutions().to(device)
        self.pred_convs = PredictionConvolutions(n_classes).to(device)
        
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes (Anchors in RPN terms)
        self.priors_cxcy = self.create_prior_boxes()
        
    def forward(self, image):
        """
        Forward propagation.
        Input:
          image, tensor of dimensions (N, 3, 300, 300)
        Returns:
         locs, class_scores for all prior boxes (regression and classif coutput)
        """
        #We need to ensure the input to VGG has 3 channels, even greyscale images
        image_three_channel = self.init_conv(image)

        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv7_feats = self.base(image_three_channel)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Rescale conv4_3 after L2 norm
        norm = torch.sqrt(conv4_3_feats.pow(2).sum(dim=1, keepdim=True))  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors.to(self.device)  # (N, 512, 38, 38)

        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores

    
    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
        Returns:
          prior boxes in cx,cy,w,h coordinates, a tensor of dimensions (8732, 4)
        """
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap]*sqrt(ratio), obj_scales[fmap]/sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(self.device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes
    
    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        Input:
         predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes; tensor of dimensions (N, 8732, 4)
         predicted_scores: class scores for each of the encoded locations/boxes; tensor of dimensions (N, 8732, n_classes)
         min_score: minimum threshold for a box to be considered a match for a certain class
         max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
         top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'

        Returns:
          detections (boxes, labels, and scores), List [bx,3]
        """

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coords

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732) #NOT USED?

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue

                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)
                
                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = torchvision.ops.box_iou(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)


                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8,device=self.device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                # image_boxes.append(class_decoded_locs[1 - suppress])
                image_boxes.append(class_decoded_locs[torch.logical_not(suppress.bool())])
                image_labels.append(torch.LongTensor((suppress.bool()).sum().item() * [c]).to(self.device))
                image_scores.append(class_scores[torch.logical_not(suppress.bool())])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.device))
                image_labels.append(torch.LongTensor([0]).to(self.device))
                image_scores.append(torch.FloatTensor([0.]).to(self.device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size












class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss as seen in the paper, a loss function for object detection
    is derived from the MultiBox objective.
    This is a weighted sum of:
        (1) a localization/regression loss for the predicted locations of the boxes, 
        (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, 
                 priors_cxcy, 
                 threshold=0.5, 
                 neg_pos_ratio=3, 
                 alpha=1., 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):

        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy #UNINITIALSED PRIORS/ANCHORS
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold =  #IoU threshold
        self.neg_pos_ratio = neg_pos_ratio #unbalanced number of pos/neg default boxes
        self.alpha = alpha #weight of localisation loss

        self.smooth_l1 = nn.SmoothL1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.device = device


    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.
        Input:
         predicted_locs : model output for coord parameterisation w.r.t default/prior boxes; torch.tensor [N,8732,4]
         predicted_scores : model output confidence score for each prior box, prob. of containing object; torch.tensor [N,8732,n_classes]
         boxes : ground truth boxes in boundary coords (xyxy); list [N, ] of torch.tensors
         labels : ground truth object labels, (torch.ones); list [N, ] of torch.tensors
        Returns:
         multibox loss : reduced along batch dimension, scalar
        """

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)


        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float,device=self.device)#.to(self.device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long,device=self.device).to(self.device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = torchvision.ops.box_iou(boxes[i],self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # First, find the prior that has the maximum overlap for each object.
            # Then, assign each object to the corresponding maximum-overlap-prior
            _, prior_for_each_object = overlap.max(dim=1)          
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(self.device)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior] 
            # Set priors whose overlaps with objects are less than the threshold to be background 
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            true_classes[i] = label_for_each_prior
            # Encode the "true" locations (using the center-size coords here) we'll regress to, using paper parameterisation
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)


        # LOCALIZATION LOSS

        # Localization loss is computed only over positive priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar
        # Indexing with torch.uint8 (byte) tensor flattens the tensor along the batch dimension also
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)


        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(self.device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar
            
        return conf_loss + self.alpha * loc_loss




#Matcher
#use hungarian matcher??
#possible need to change the scale of the anchors? So we have more matches
#https://github.com/facebookresearch/detr/blob/main/models/matcher.py
#https://github.com/pytorch/vision/torchvision/models/detection/_utils.py



from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1):
        """
        Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"



    def box_iou(self, boxes1, boxes2):
        area1 = torchvision.ops.boxes.box_area(boxes1)
        area2 = torchvision.ops.boxes.box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union   
        
    def generalized_box_iou(self, boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/
        The boxes should be in [x0, y0, x1, y1] format
        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        #assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        iou, union = self.box_iou(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area


    @torch.no_grad()
    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """ Performs the matching
        Params:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        batch_size = predicted_locs.size(0)
        n_priors = predicted_locs.size(1) #8732
        n_classes = predicted_scores.size(2)  #2

        out_prob = predicted_scores.flatten(0,1).softmax(-1) # [batch_size*n_priors, n_classes]
        out_bbox = predicted_locs.flatten(0,1) # [batch_size*n_priors, 4]
        #print(out_bbox)

        #we should concat the target labels and boxes
        tgt_ids = torch.cat([v for v in labels]).long() # [16]
        tgt_bbox = torch.cat([v for v in boxes]) # [16,4]       

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids] # [batch_size*n_priors, 16]
        #print('1',cost_class.shape)

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(xy_to_cxcy(out_bbox), xy_to_cxcy(tgt_bbox), p=1) # [batch_size*n_priors, 16]
        #print('2',cost_bbox.shape)

        cost_giou = -self.generalized_box_iou(out_bbox,tgt_bbox) # [batch_size*n_priors, 16]
        #print('3',cost_giou.shape)
        #final cost matrix:
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, n_priors, -1).cpu() # [batch_size, n_priors, 16]
        #print('C',C.shape)

        sizes = [len(v) for v in boxes] # [batch_size] n_objects_per_image
        #print(sizes)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        #print(indices)

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]







#testing
if __name__=='__main__':
    from utils.dataset import SSDRealDataset
    from torch.utils.data import DataLoader
    das = SSDRealDataset(annotation_json="/home/users/b/bozianu/work/data/real/anns_2layers.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dal = DataLoader(das, batch_size=2, shuffle=False, collate_fn=das.collate_fn)
    model = SSD(pretrained_vgg=False,in_channels=2)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, alpha=0.9,device=device).to(device)
    #criterion = NewDetectionLoss(priors_cxcy=model.priors_cxcy).to(device)
    print(model.priors_cxcy.shape,type(model.priors_cxcy))
    matcher = HungarianMatcher()
    print(matcher)

    for step, (img, boxes, labels) in enumerate(train_dal):
        img = img.to(device)
        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]
        
        pred_loc, pred_sco = model(img)
        matcher(pred_loc,pred_sco,boxes,labels)



        print('Boxes,',len(boxes),boxes[0].shape,boxes[1].shape)
        loss = criterion(pred_loc, pred_sco, boxes, labels)
        print('Overall loss->',loss)
        if step==0:
            break




