import numpy as np


import torch
from torchvision import models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision

from PIL import Image
import matplotlib.pyplot as plt

import os
import re
from math import sqrt


from models.models import VGGBase, AuxiliaryConvolutions, PredictionConvolutions
from utils.utils import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy, find_jaccard_overlap


print(os.listdir("/home/users/b/bozianu/work/SSD/SSD/data/input"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #globally define


class SSDDataset(Dataset):
    def __init__(self, file_folder, is_test=False, transform=True):
        self.img_folder_path = '/home/users/b/bozianu/work/SSD/SSD/data/input/Images/'
        self.annotation_folder_path = '/home/users/b/bozianu/work/SSD/SSD/data/input/Annotation/'
        self.file_folder = file_folder
        self.is_test = is_test
        self.transform = transform
        
    def __getitem__(self, idx):
        file = self.file_folder[idx]
        img_path = self.img_folder_path + file
        img = Image.open(img_path)
        img = img.convert('RGB')
        
        if not self.is_test:
            annotation_path = self.annotation_folder_path + file.split('.')[0]
            with open(annotation_path) as f:
                annotation = f.read()

            boxes = self.get_xys(annotation)

            new_boxes = self.resize_boxes(boxes, img)
            if self.transform is not None:
                img = self.resize_image(img)

            return img, new_boxes, torch.ones(len(new_boxes))

        else:
            return img
    
    def __len__(self):
        return len(self.file_folder)
        
    def get_xys(self, annotation):
        xmin = list(map(int,re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', annotation)))
        xmax = list(map(int,re.findall('(?<=<xmax>)[0-9]+?(?=</xmax>)', annotation)))
        ymin = list(map(int,re.findall('(?<=<ymin>)[0-9]+?(?=</ymin>)', annotation)))
        ymax = list(map(int,re.findall('(?<=<ymax>)[0-9]+?(?=</ymax>)', annotation)))
        
        boxes_true = list(map(list, zip(xmin, ymin, xmax,ymax)))
        return torch.tensor(boxes_true)

    def resize_image(self, img, dims=(300,300)):
        tsfm = transforms.Compose([transforms.Resize([300, 300]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        return tsfm(img)
        
    def resize_boxes(self, boxes, img, dims=(300, 300)):
        #rescale boxes so they still cover objects in resized image
        old_dims = torch.FloatTensor([img.width, img.height, img.width, img.height]).unsqueeze(0)
        new_boxes = torch.div(boxes,old_dims)

        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        scaled_boxes = new_boxes * new_dims
        
        return new_boxes
    
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Input:
          batch, the list of imgs, new_boxes, labels from __getitem__
        Returns:
         tensor of images, lists of bounding boxes and labels
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each







all_img_folder = os.listdir('/home/users/b/bozianu/work/SSD/SSD/data/input/Images')

all_img_name = []
for img_folder in all_img_folder:
    img_folder_path = '/home/users/b/bozianu/work/SSD/SSD/data/input/Images/' + img_folder
    all_img_name += list(map(lambda x: img_folder + '/'+ x, os.listdir(img_folder_path)))

print(len(all_img_name), all_img_name[0])

train_ds = SSDDataset(all_img_name[:int(len(all_img_name)*0.1)])
_, bbs, lbs = train_ds[29]
print(bbs)
print(lbs)







class SSD(nn.Module):
    def __init__(self, n_classes=2,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(SSD, self).__init__()
        self.device=device
        self.n_classes = n_classes
        
        self.base = VGGBase().to(device)
        self.aux_convs = AuxiliaryConvolutions().to(device)
        self.pred_convs = PredictionConvolutions(n_classes).to(device)
        
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()
        
    def forward(self, image):
        """
        Forward propagation.
        Input:
          image, tensor of dimensions (N, 3, 300, 300)
        Returns:
         locs, class_scores for all prior boxes (regression and classif coutput)
        """

        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Rescale conv4_3 after L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)

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
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes
    
    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        Input:
         predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
         predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
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

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

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
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

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
                image_labels.append(torch.LongTensor((suppress.bool()).sum().item() * [c]).to(device))
                image_scores.append(class_scores[torch.logical_not(suppress.bool())])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

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
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')


    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """

        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)


        # LOCALIZATION LOSS
        #print('predicted_locs[positive_priors]',predicted_locs[positive_priors])
        #print('true_locs[positive_priors]',true_locs[positive_priors])

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
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
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar
        #print('conf_loss',conf_loss,'\tloc_loss',loc_loss)
        return conf_loss + self.alpha * loc_loss




#######################################################################################################

#######################################################################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 7
LR = 1e-3
BS = 8
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
print_feq = 100

model = SSD().to(device)
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=momentum, weight_decay=weight_decay)



#######################################################################################################



all_img_folder = os.listdir('/home/users/b/bozianu/work/SSD/SSD/data/input/Images')

all_img_name = []
for img_folder in all_img_folder:
    img_folder_path = '/home/users/b/bozianu/work/SSD/SSD/data/input/Images/' + img_folder
    all_img_name += list(map(lambda x: img_folder + '/'+ x, os.listdir(img_folder_path)))

print(len(all_img_name), all_img_name[0])

train_ds = SSDDataset(all_img_name[:int(len(all_img_name)*0.8)])
train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=train_ds.collate_fn)

valid_ds = SSDDataset(all_img_name[int(len(all_img_name)*0.8):int(len(all_img_name)*0.9)])
valid_dl = DataLoader(valid_ds, batch_size=BS, shuffle=True, collate_fn=valid_ds.collate_fn)




#######################################################################################################




from tqdm import tqdm
import time


tr_loss_tot = []
vl_loss_tot = []
for epoch in range(EPOCH):
    model.train()
    train_loss = []
    for step, (img, boxes, labels) in enumerate(train_dl):
        time_1 = time.time()
        img = img.to(device)

        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]
        
        pred_loc, pred_sco = model(img)
        
        loss = criterion(pred_loc, pred_sco, boxes, labels)
        
        # Backward prop.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        if step % print_feq == 0:
            print('epoch:', epoch, 
                  '\tstep:', step+1, '/', len(train_dl) + 1,
                  '\ttrain loss:', '{:.4f}'.format(loss.item()),
                  '\ttime:', '{:.4f}'.format((time.time()-time_1)*print_feq), 's')
    
    model.eval();
    valid_loss = []
    for step, (img, boxes, labels) in enumerate(tqdm(valid_dl)):
        img = img.to(device)
        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]
        pred_loc, pred_sco = model(img)
        loss = criterion(pred_loc, pred_sco, boxes, labels)
        valid_loss.append(loss.item())

    tr_loss_tot.append(np.mean(train_loss))
    vl_loss_tot.append(np.mean(valid_loss))   
    print('epoch:', epoch, '/', EPOCH,
            '\ttrain loss:', '{:.4f}'.format(np.mean(train_loss)),
            '\tvalid loss:', '{:.4f}'.format(np.mean(valid_loss)))





plt.figure()
x_axis = torch.arange(EPOCH)
plt.plot(x_axis,tr_loss_tot,'--',label='training loss')
plt.plot(x_axis,vl_loss_tot,'--',label='val loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Arbitrary Loss')
plt.savefig('losses.png')




torch.save(model.state_dict(), 'SSD_model_{}.pth'.format(EPOCH))








