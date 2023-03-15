import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from models.models import VGGBase, AuxiliaryConvolutions, PredictionConvolutions
from utils.utils import cxcywh2xyxy, cxcy_to_gcxgcy, gcxgcy_to_cxcy, xyxy2cxcywh, find_jaccard_overlap



class SSD(nn.Module):
    def __init__(self, n_classes=2,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(SSD,self).__init__()
        self.n_classes = n_classes
        self.device = device

        self.base = VGGBase().to(device)
        self.aux_convs = AuxiliaryConvolutions().to(device)
        self.out_convs = PredictionConvolutions(n_classes).to(device)

        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.generate_prior_boxes()

    
    def forward(self, image):
        """
        Forward prop through all 3 networks, from image to preds.
        Inputs: 
         image: a tensor containing the input images [N,3,300,300]
        Returns:
         8732 locations and class scores (wrt each prior box) for each image [N,8732,4], [N,8732,n_classes]
        """

        #pass through VGG base network - retrieve the low level feature maps
        conv4_3_feats, conv7_feats = self.base(image)

        #rescale conv4_3 after L2 norm
        norm=conv4_3_feats.pow(2).sum(dim=1,keepdim=True).sqrt() #(N,1,38,38)
        conv4_3_feats = conv4_3_feats / norm
        conv4_3_feats = conv4_3_feats * self.rescale_factors

        #pass through auxiliary network
        conv8_2_feats, conv_9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)

        #take all generated feature maps and pass through the output convs
        locs, class_scores = self.out_convs(conv4_3_feats,conv7_feats, conv8_2_feats, conv_9_2_feats, conv10_2_feats, conv11_2_feats) # (N,8732,4), (N, 8732, n_classes)

        return locs, class_scores
    
    def generate_prior_boxes(self):
        """
        Here we produce 8732 prior (default) boxes as seen in the paper with different numbers of boxes produced 
        in different feature maps. 
        Output:
         Prior boxes in cxcywh coords [8732,4]
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
        
        aspect_ratios = {'conv4_3': [1.0, 2.0, 0.5],
                         'conv7': [1.0, 2.0, 3.0, 0.5, .333],
                         'conv8_2': [1.0, 2.0, 3.0, 0.5, .333],
                         'conv9_2': [1.0, 2.0, 3.0, 0.5, .333],
                         'conv10_2': [1.0, 2.0, 0.5],
                         'conv11_2': [1.0, 2.0, 0.5]}
        
        fmaps = list(fmap_dims.keys())

        prior_boxes = []
        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx,
                                            cy,
                                            obj_scales[fmap]*torch.sqrt(torch.tensor(ratio)),
                                            obj_scales[fmap]/torch.sqrt(torch.tensor(ratio))])
                        #for aspect ratio 1 use additional prior, scale is the mean of the scale of current/next feature map
                        if ratio==1.0:
                            try:
                                additional_scale = torch.sqrt(torch.tensor(obj_scales[fmap]*obj_scales[fmaps[k+1]]))
                            except IndexError:
                                additional_scale = 1.0
                            prior_boxes.append([cx,cy,additional_scale,additional_scale])
        
        prior_boxes = torch.tensor(prior_boxes,dtype=float,device=self.device)
        prior_boxes.clamp_(0,1)
        
        return prior_boxes
    

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Take the 8732 locations and class scores to determine where objects (probably) are 
        Perform NMS for each class for boxes above a min threshold

        Inputs:
         predicted_locs: predicted locations of boxes wrt the prior boxes [N,8732,4]
         predicted_scores: class scores for each of the encoded boxes [N,8732,n_classes]
         min_score: min threshold for a box to be considered a match for a class
         max_overlap: maximum overlap permitted between two boxes (NMS)
         top_k: if many boxes remain, keep only top k

        Returns:
         detections (boxes, labels, scores) list of length N (batch size)
        """
        batch_size = predicted_locs.size(0)
        assert predicted_locs.size(0)==predicted_scores.size(0), "Model should output predictions with identical dimension 0"
        
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores,dim=2) # (N,8732,n_classes) #NUMERICAL STABILITY
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        all_image_boxes, all_image_labels, all_image_scores = [], [], [] #final storage lists
        for i in range(batch_size):
            image_boxes, image_labels, image_scores = [], [], [] 

            decoded_locs = cxcywh2xyxy(gcxgcy_to_cxcy(predicted_locs[i],self.priors_cxcy)) # [8732,4]
            max_scores, best_label = predicted_scores[i].max(dim=1) # [8732]

            #for each class
            for c in range(1,self.n_classes):
                class_scores = predicted_scores[i][:,c] #[8732] 
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()

                if n_above_min_score == 0:
                    print('No priors score above min score {}'.format(min_score))
                    continue

                class_scores = class_scores[score_above_min_score] #[n_passing]
                class_decoded_locs = decoded_locs[score_above_min_score] #[n_passing,4]


                class_scores, sort_ind = class_scores.sort(dim=0,descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]
                
                #check overlaps - NMS
                overlap = find_jaccard_overlap(class_decoded_locs,class_decoded_locs) #[n_passing,n_min_score]
                #1 means suprress, 0 means keep
                suppress = torch.zeros((n_above_min_score),dtype=torch.uint8).to(self.device)

                for box in range(class_decoded_locs.size(0)): #consider each of the passing boxes
                    if suppress[box] == 1:
                        continue
                    #supress those whose overlaps with this box > max_overlap
                    #max wretains previously suppressed boxes (logical OR)
                    suppress = torch.max(suppress, overlap[box] > max_overlap)

                    suppress[box] = 0 # don't suppress this box despite its overlap with itself == 1
                
                #only store those passing NMS
                image_boxes.append(class_decoded_locs[1-suppress])
                image_scores.append(class_scores[1-suppress])
                image_labels.append(torch.LongTensor((1-suppress).sum().item()*[c]).to(self.device))
                
            image_boxes = torch.cat(image_boxes, dim=0) #[n_objects,4]
            image_scores = torch.cat(image_scores,dim=0) #[n_objects]
            image_scores = torch.cat(image_labels,dim=0) #[n_objects]
            n_objects = image_scores.size(0)
        

            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0,descending=True)
                image_scores = image_scores[:top_k]
                image_boxes = image_boxes[sort_ind][:top_k]
                image_labels = image_labels[sort_ind][:top_k]
            
            #append to lists of length batch size
            all_image_boxes.append(image_boxes)
            all_image_scores.append(image_scores)
            all_image_labels.append(image_labels)

        return all_image_boxes, all_image_scores, all_image_labels
    



class MultiBoxLoss(nn.Module):
    """
    Implement the loss function for object detction, following paper as a combination
    of localization loss and confidence loss for predicted class scores
    """
    def __init__(self, 
                 priors_cxcy, 
                 threshold=0.5, 
                 neg_pos_ratio=3, 
                 alpha=1.0, 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        super(MultiBoxLoss,self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcywh2xyxy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.device = device

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Inputs:
         predicted_locs: predicted locations of prior boxes [N,8732,4]
         predicted_scores: class scores for each of the encoded locations [N,8732,n_classes]
         boxes: ground truth bounding boxes in xyxy coords list of N tensors
         labels: true object labels, list of N tensors

        Returns:
         Multibox losss, scalar
        """

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size,n_priors,4),dtype=torch.float).to(self.device)
        true_classes = torch.zeros((batch_size,n_priors),dtype=torch.long).to(self.device)
        #print('n_priors',n_priors)
        #print('boxes.size(0)',[b.size(0) for b in boxes])
        #for each image in the batch
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],self.priors_xy)
            #print('overlap\n',overlap.max(dim=0))

            #NOW WE ASSIGN PRIORS A TRUE OBJECT TO AIM FOR, and also ensure all objects have someone aiming for them
            #for each prior box find the object that has the maximmum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)

            #there are situations where objects may not be assigned any prior boxes, we remedy this:
            _, prior_for_each_object = overlap.max(dim=1)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(self.device)
            #artificaially inflate these priors by giving them an overlap > threhsold
            overlap_for_each_prior[prior_for_each_object] = 1.0

            label_for_each_prior = labels[i][object_for_each_prior]
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0

            true_classes[i] = label_for_each_prior
            true_locs[i] = cxcy_to_gcxgcy(xyxy2cxcywh(boxes[i][object_for_each_prior]),self.priors_cxcy) #encode centre object coordinates in the form we regress pred boxes to

        positive_priors = true_classes != 0

        #LOCALIZATION LOSS
        loc_loss = self.smooth_l1(predicted_locs[positive_priors],true_locs[positive_priors])

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)



        #CONFIDENCE LOSS
        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance
        
        # number of positive and hard-negative priors in the image
        n_positives = positive_priors.sum(dim=1)
        #print('n_positives',n_positives)
        n_hard_negatives = self.neg_pos_ratio * n_positives

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  #[N * 8732]
        conf_loss_all = conf_loss_all.view(batch_size, n_priors) #[N,8732]

        #we already know the positive priors = FG!
        conf_loss_pos = conf_loss_all[positive_priors]

        #now for the hard-negative priors
        #sort only negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1,descending=True)
        hardnes_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(self.device)
        hard_negatives = hardnes_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]
        
        #as seen in the paper, averaged over positive priors only although computer over both pos and hard-neg priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()
        return conf_loss + self.alpha * loc_loss




