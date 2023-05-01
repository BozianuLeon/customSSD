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
        id_list_3 = [i for i in id_list_uniq if self.check_channels(i)] #only include color images
        self.ids =  id_list_3       
        
        
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








class SSDToyDataset(Dataset):
    '''
    Preparing toy data dataset, result in same format as COCO dataset
    '''
    def __init__(self,annotation_json,is_test=False,transform=True):

        with open(annotation_json) as json_file:
            annotations = json.load(json_file)
        self.annotations = annotations
        self.ids = torch.arange(len(self.annotations))
        self.is_test = is_test
        self.transform = transform

    def __getitem__(self, index):
        annotations_i = self.annotations[str(index)]
        path = annotations_i["image"]["img_path"]
        #img = cv2.imread(path)#cv2.imread(path,0).T #0 for grayscale, transpose for x-y flip

        img = Image.open(path)
        img = img.convert('RGB')
        n_objs = annotations_i["anns"]["n_gausses"] #len(annotations_i["anns"]["bboxes"])

        if not self.is_test:
            boxes = self.get_xys(annotations_i,n_objs)
            
            if self.transform is True:
                new_boxes = self.resize_boxes(boxes,img)
                img = self.resize_image(img)

            return img, new_boxes, torch.ones(len(new_boxes))
        
        else:
            return img
    
    def __len__(self):
        return len(self.ids)


    def get_xys(self, anns, n_objs):

        boxes_true = []
        for i in range(n_objs):
            xmin = anns["anns"]["bboxes"][i][0]
            ymin = anns["anns"]["bboxes"][i][1]
            xmax = xmin + anns["anns"]["bboxes"][i][2]
            ymax = ymin + anns["anns"]["bboxes"][i][3]
            boxes_true.append([xmin,ymin,xmax,ymax])

        return torch.tensor(boxes_true)


    def resize_image(self, img, dims=(300,300)):
        # tsfm = transforms.Compose([transforms.ToTensor(),
        #                            transforms.Resize([300, 300]),
        #                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        tsfm = transforms.Compose([transforms.Resize([300, 300]),
                                   transforms.ToTensor(),])
                                   #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        return tsfm(img).permute(0,2,1) #permute necessary beacuse its cv2

    def resize_boxes(self, boxes, img, dims=(300, 300)):
        #rescale boxes so they still cover objects in resized image
        #old_dims = torch.FloatTensor([img.shape[0], img.shape[1], img.shape[0], img.shape[1]]).unsqueeze(0)
        old_dims = torch.FloatTensor([img.width, img.height, img.width, img.height]).unsqueeze(0)
        new_boxes = torch.div(boxes,old_dims)

        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        scaled_boxes = new_boxes * new_dims
        
        return new_boxes

    def xywh2xyxy(self,x):
        x, y, w, h = x.unbind(-1)
        b = [x,y,x+w,y+h]
        return torch.stack(b, dim=-1)

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each






class SSDRealDataset(Dataset):
    '''
    Now we have real calorimeter images produced using real_data_production script.
    Conversely to other datasets, the "images" are saved as [2,97,64] pytorch tensors
    '''
    def __init__(self,annotation_json,is_test=False):

        with open(annotation_json) as json_file:
            annotations = json.load(json_file)
        self.annotations = annotations
        self.ids = torch.arange(len(self.annotations))
        self.is_test = is_test
        if self.is_test:
            print('Initialising dataset module in test mode, dataloader ouput in form:\n img, boxes, extent')

    def __getitem__(self, index):
        annotations_i = self.annotations[str(index)]
        path = annotations_i["image"]["img_path"]
        
        img_tensor = torch.load(path)
        img_tensor = img_tensor.type('torch.FloatTensor')
        n_objs = annotations_i["anns"]["n_clusters"] 
        extent = annotations_i["anns"]["extent"]

        
        transfm = transforms.Compose([transforms.Resize([300, 300],interpolation=transforms.InterpolationMode.NEAREST)])
        img = transfm(img_tensor)
        boxes = self.get_xys(annotations_i,n_objs)
        boxes = self.resize_boxes(boxes,img_tensor)

        if not self.is_test:
            return img, boxes, torch.ones(len(boxes)) #for training we need the so-called class labels

        else:
            return img, boxes, torch.FloatTensor(extent) #for inference we know the labels ALL 1, we want the extent for plotting
    
    def __len__(self):
        return len(self.ids)
    
    def get_extent(self):
        #extent is the same for every image!
        annotations_0 = self.annotations[0]
        extent = annotations_0["anns"]["extent"]
        return extent

    def get_xys(self, anns, n_objs):
        boxes_true = []
        for i in range(n_objs):
            xmin = anns["anns"]["bboxes"][i][0]
            ymin = anns["anns"]["bboxes"][i][1]
            xmax = xmin + anns["anns"]["bboxes"][i][2]
            ymax = ymin + anns["anns"]["bboxes"][i][3]
            boxes_true.append([xmin,ymin,xmax,ymax])

        return torch.tensor(boxes_true)

    def resize_boxes(self, boxes, img, dims=(300, 300)):
        #rescale boxes so they still cover objects in resized image
        
        #scaling_factor = torch.FloatTensor([dims[0]/img.shape[2], dims[1]/img.shape[1], dims[0]/img.shape[2], dims[1]/img.shape[1]]).unsqueeze(0)
        scaling_factor = torch.FloatTensor([1/img.shape[2], 1/img.shape[1], 1/img.shape[2], 1/img.shape[1]]).unsqueeze(0)
        scaled_boxes = scaling_factor * boxes
        return scaled_boxes

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each






if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    das = SSDRealDataset('/home/users/b/bozianu/work/data/real/anns_2layers.json')
    print('Images in dataset:',len(das))
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")


    image_tensor, truth_boxes, truth_labels = das[0]
    #examine_one_image(image_tensor,truth_boxes)

    BS = 8
    train_dal = DataLoader(das, batch_size=BS, shuffle=True, collate_fn=das.collate_fn)

    n_batches = 0
    for step, (img, tru_boxes, labels) in enumerate(train_dal):

        img_tensor = img.to(device)
        tru_boxes = [box.detach().to(device) for box in tru_boxes]
        labels = [label.detach().to(device) for label in labels]
        B, C, H, W = img.shape
        print(type(img),img.shape, img_tensor.shape)
        img_shape = torch.FloatTensor([H,W,H,W]).unsqueeze(0)
        tru_boxes = [box * img_shape for box in tru_boxes]

        #revert to original images/box dimensions
        B, C, H, W = img.shape


        for i in range(len(img)):
            image_i = img[i].cpu().squeeze()
            gt_boxes_i = tru_boxes[i]

            f,ax = plt.subplots(1,2)
            ax[0].imshow(image_i[0],cmap='binary_r')
            ax[1].imshow(image_i[1],cmap='binary_r')
            for bbx in gt_boxes_i:
                x,y=float(bbx[0]),float(bbx[1])
                w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
                bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='limegreen',fc='none')
                ax[0].add_patch(bb)
                ax[1].add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='limegreen',fc='none'))

            f.savefig('datasettest.png')
            plt.close()
 
        if step==n_batches:
            break