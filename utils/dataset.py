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














class CustomCOCODataset(Dataset):
    '''
    Preparing COCO images à la h-huang-github.io/tutorials/intermediate/torchvision_tutorial.html
    '''
    def __init__(self,
                 root_folder,
                 annotation_json
    ):

        self.root = root_folder
        self.coco = COCO(annotation_json)
        self.cat_ids = self.coco.getCatIds(['dog','cat','horse','sheep','cow','bird','elephant','zebra','giraffe','bear'])
        id_list = np.hstack([self.coco.getImgIds(catIds=[idx]) for idx in self.cat_ids]) #only include images of animals
        id_list_uniq = list(sorted(set(id_list))) #only include each image once
        id_list_3 = [i for i in id_list_uniq if self.check_channels(i)] #only include color images
        self.ids = id_list_3


    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_id = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_id)

        path = self.coco.loadImgs([img_id])[0]["file_name"]
        img = Image.open(os.path.join(self.root,path))
        n_objs = len(anns)

        # Bounding boxes for all objects in image
        # In coco format bbox = [xmin,ymin,width,height]
        # In pytorch, bbox = [xmin,ymin,xmax,ymax]
        # TODO: No need for the for loop
        boxes = []
        iscrowd = []
        for i in range(n_objs):
            xmin = anns[i]["bbox"][0]
            ymin = anns[i]["bbox"][1]
            xmax = xmin + anns[i]["bbox"][2]
            ymax = ymin + anns[i]["bbox"][3]
            boxes.append([xmin,ymin,xmax,ymax])
            iscrowd.append(anns[i]["iscrowd"])
        
        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd,dtype=torch.float32)
        labels = torch.ones((n_objs,),dtype=torch.int64)
        img_id = torch.tensor([img_id])

        img_tensor, scaled_boxes = self.prepare_image(img,boxes) # ensure all images are same size, and boxes still on objects

        my_annotations = {}
        my_annotations["image_id"] = img_id
        my_annotations["path"] = path
        my_annotations["boxes"] = scaled_boxes
        my_annotations["labels"] = labels
        my_annotations["iscrowd"] = iscrowd

        return img_tensor, my_annotations


    def __len__(self):
        return len(self.ids)
    

    def prepare_image(self,
                      img,
                      boxes,
                      size = (300,300),
    ):
        #resizes and turns image to tensor, ensures boxes still lie on objects

        scaled_boxes = boxes
        scaled_boxes[:,[0,2]] = (size[0]*(boxes[:,[0,2]]/img.size[0]))
        scaled_boxes[:,[1,3]] = (size[1]*(boxes[:,[1,3]]/img.size[1]))

        resize_tens = transforms.Compose([transforms.Resize(size),
                                          transforms.ToTensor(),
                                          transforms.Normalize()])
        scaled_img = resize_tens(img)
        return scaled_img, scaled_boxes


    def check_channels(self, img_id):
        #checks that we have an RGB image
        img_path = self.coco.loadImgs([img_id])[0]["file_name"]
        im = Image.open(os.path.join(self.root, img_path))
        if len(im.mode)==1:
            im.close()
            return False
        elif len(im.mode)==3:
            im.close()
            return True
        else:
            im.close()
            return -1






class CustomToyDataset(Dataset):
    '''
    Preparing toy data dataset, result in same format as COCO dataset
    '''
    def __init__(self,annotation_json):

        with open(annotation_json) as json_file:
            annotations = json.load(json_file)
        self.annotations = annotations
        self.ids = torch.arange(len(self.annotations))

    def __getitem__(self, index):
        annotations_i = self.annotations[str(index)]
        path = annotations_i["image"]["img_path"]
        img = Image.open(path).convert('L')
        img.show()
        print(img.mode)

        n_objs = annotations_i["annotations"]["n_gausses"]

        tensor_boxes = torch.as_tensor(annotations_i["annotations"]["bboxes"])
        boxes = self.xywh2xyxy(tensor_boxes)
        labels = torch.ones((n_objs,),dtype=torch.int32)

        img_tensor, scaled_boxes = self.prepare_image(img,boxes)

        my_annotations = {}
        my_annotations["image_id"] = index
        my_annotations["path"] = path
        my_annotations["boxes"] = scaled_boxes
        my_annotations["labels"] = labels

        return img_tensor, my_annotations
    
    def __len__(self):
        return len(self.ids)

    def xywh2xyxy(self,x):
        x, y, w, h = x.unbind(-1)
        b = [x,y,x+w,y+h]
        return torch.stack(b, dim=-1)
    
    def prepare_image(self,
                      img,
                      boxes,
                      size=(256,256)
    ):
        
        scaled_boxes = boxes
        scaled_boxes[:,[0,2]] = (size[0]*(boxes[:,[0,2]]/img.size[0]))
        scaled_boxes[:,[1,3]] = (size[1]*(boxes[:,[1,3]]/img.size[1]))   

        resize_tens = transforms.Compose([transforms.Resize(size),
                                          transforms.ToTensor()])
        scaled_img = resize_tens(img)
        return scaled_img, scaled_boxes








class CustomCOCODataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle,
                 drop_last=False,
                 num_workers=0
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers

    
    def custom_collate_fn(self, batch):

        # Function to correctly stack images/annotations inside the batch
        # Output: 
        # images: (batch_size, 3, 256, 256)
        # targets: List[Dict[str, Tensor]]

        # boxes: (batch_size, n_obj, 4)
        # labels: (batch_size, n_obj)
        # index: (batch_size)
        # path: (batch_size)

        img_tensor_list = []
        batch_targets_list = []
        for img_tensor, my_anns in batch:
            img_tensor_list.append(img_tensor)
            targets = dict(
                boxes=my_anns['boxes'],
                labels=my_anns['labels'],
                image_index=my_anns["image_id"],
                image_path=my_anns['path']
            )
            batch_targets_list.append(targets)
                
        batch_images = torch.stack(img_tensor_list,dim=0)

        return batch_images, batch_targets_list
    
    
    def loader(self):
        return DataLoader(dataset=self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last,
                          num_workers=self.num_workers,
                          collate_fn=self.custom_collate_fn)








