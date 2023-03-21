import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import re
import numpy as np
from math import sqrt

import json
from pycocotools.coco import COCO
from PIL import Image



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














class SSDCOCODataset(Dataset):
    def __init__(self, root_folder, annotation_json, is_test=False, transform=True):
        self.root = root_folder
        self.annotation_json = annotation_json
        self.is_test = is_test
        self.transform = transform

        self.coco = COCO(annotation_json)
        self.cat_ids = self.coco.getCatIds(['dog','cat','horse','sheep','cow','bird','elephant','zebra','giraffe','bear'])
        id_list = np.hstack([self.coco.getImgIds(catIds=[idx]) for idx in self.cat_ids]) #only include images of animals
        id_list_uniq = list(sorted(set(id_list))) #only include each image once
        #id_list_3 = [i for i in id_list_uniq if self.check_channels(i)] #only include color images
        self.ids = id_list_uniq #id_list_3       
        
        
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_id = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_id)

        path = self.coco.loadImgs([img_id])[0]["file_name"]
        img = Image.open(os.path.join(self.root,path))
        img = img.convert('RGB')
        n_objs = len(anns)
        
        if not self.is_test:
            boxes = self.get_xys(anns, n_objs)

            if self.transform is not None:
                new_boxes = self.resize_boxes(boxes, img)
                img = self.resize_image(img)

            return img, new_boxes, torch.ones(len(new_boxes))

        else:
            return img
    
    def __len__(self):
        return len(self.ids)
        
    def get_xys(self, anns, n_objs):
        boxes_true = []
        for i in range(n_objs):
            xmin = anns[i]["bbox"][0]
            ymin = anns[i]["bbox"][1]
            xmax = xmin + anns[i]["bbox"][2]
            ymax = ymin + anns[i]["bbox"][3]
            boxes_true.append([xmin,ymin,xmax,ymax])

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


























