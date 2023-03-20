print('Starting')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import time
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from random import randint

from models.models import VGGBase, AuxiliaryConvolutions, PredictionConvolutions
from utils.utils import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy, find_jaccard_overlap






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
        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Rescale conv4_3 after L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors.to(device)  # (N, 512, 38, 38)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores

    
    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
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
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
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




print('Imported')




all_img_folder = os.listdir('/home/users/b/bozianu/work/SSD/SSD/data/input/Images')

all_img_name = []
for img_folder in all_img_folder:
    img_folder_path = '/home/users/b/bozianu/work/SSD/SSD/data/input/Images/' + img_folder
    all_img_name += list(map(lambda x: img_folder + '/'+ x, os.listdir(img_folder_path)))




tsfm = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])




model_save_path = "/home/users/b/bozianu/work/SSD/SSD/SSD_model_7.pth"
save_at = "/home/users/b/bozianu/work/SSD/SSD/inference/" + "SSD_model_7/" + time.strftime("%Y%m%d-%H%M%S") + "/"
if not os.path.exists(save_at):
    os.makedirs(save_at)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_images = 15



model = SSD()
model.load_state_dict(torch.load(model_save_path))
model.eval()




for i in range(n_images):
    
    img_number = -1*randint(1,len(all_img_name))
    print('Check here for sanity: ',all_img_name[img_number])
    print('Save loc',len(all_img_name)-img_number)
    origin_img = Image.open('/home/users/b/bozianu/work/SSD/SSD/data/input/Images/' + all_img_name[img_number]).convert('RGB')
    img = tsfm(origin_img)
    img = img.to(device)


    with open('/home/users/b/bozianu/work/SSD/SSD/data/input/Annotation/'+all_img_name[img_number][:-4]) as f:
        reader = f.read()

    xmin = list(map(int,re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', reader)))
    xmax = list(map(int,re.findall('(?<=<xmax>)[0-9]+?(?=</xmax>)', reader)))
    ymin = list(map(int,re.findall('(?<=<ymin>)[0-9]+?(?=</ymin>)', reader)))
    ymax = list(map(int,re.findall('(?<=<ymax>)[0-9]+?(?=</ymax>)', reader)))

    predicted_locs, predicted_scores = model(img.unsqueeze(0))
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=0.2, max_overlap=0.5, top_k=200)
    det_boxes = [box.to(device) for box in det_boxes]
    # print('det_boxes',det_boxes)
    # print('det_labels',det_labels)
    # print('det_scores',det_scores)
    
    origin_dims = torch.FloatTensor([origin_img.width, origin_img.height, origin_img.width, origin_img.height]).unsqueeze(0).to(device)
    det_boxes = [box * origin_dims for box in det_boxes]




    draw = ImageDraw.Draw(origin_img)

    #PREDICTIONS
    pred_boxes = det_boxes[0].tolist()
    print(' # pred_boxes',len(pred_boxes),'!')
    for boxp in pred_boxes:
        draw.rectangle(xy=boxp, outline='red',width=3)

    #TRUTH
    boxes_true = list(map(list, zip(xmin, ymin, xmax,ymax)))
    for boxt in boxes_true:
        draw.rectangle(xy=boxt,outline='limegreen',width=3)

    plt.figure(figsize=(5, 5))
    plt.imshow(origin_img)
    plt.savefig(save_at+'test_infer_img_{}.png'.format(len(all_img_name)-img_number))
    print()

