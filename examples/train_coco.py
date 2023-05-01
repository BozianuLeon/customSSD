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

from ssd import SSD, MultiBoxLoss 
from models.models import VGGBase, AuxiliaryConvolutions, PredictionConvolutions
from utils.utils import xy_to_cxcy, cxcy_to_xy, cxcy_to_gcxgcy, gcxgcy_to_cxcy
from utils.dataset import SSDDataset, SSDCOCODataset




tsfm = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 2
LR = 1e-3
BS = 1#8
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
print_feq = 100

model = SSD()
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=momentum, weight_decay=weight_decay)



das = SSDCOCODataset(
    root_folder="/home/users/b/bozianu/work/data/train2017",
    annotation_json="/home/users/b/bozianu/work/data/annotations/instances_train2017.json",
)
print('Images in dataset:',len(das))

train_size = int(0.5 * len(das))
val_size = int(0.05 * len(das))
test_size = len(das) - train_size - val_size
print('\ttrain / val / test size : ',train_size,'/',val_size,'/',test_size)

torch.random.manual_seed(1)
train_das, val_das, test_das = torch.utils.data.random_split(das,[train_size,val_size,test_size])


train_dal = DataLoader(train_das, batch_size=BS, shuffle=True, collate_fn=das.collate_fn)
val_dal = DataLoader(val_das,batch_size=BS,shuffle=True,collate_fn=das.collate_fn)



from tqdm import tqdm
import time


tr_loss_tot = []
vl_loss_tot = []
for epoch in range(EPOCHS):
    model.train()
    train_loss = []
    for step, (img, boxes, labels) in enumerate(train_dal):
        time_1 = time.perf_counter()
        img = img.to(device)

        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]
        
        pred_loc, pred_sco = model(img)
        
        loss = criterion(pred_loc, pred_sco, boxes, labels)
        if loss > 100:
            print('BAD loss',loss)
            print('\nboxes',len(boxes),boxes)
            print('\nlabels',labels,'\n')
        
        # Backward prop.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        if step % print_feq == 0:
            print('epoch:', epoch, 
                  '\tstep:', step, '/', len(train_dal),
                  '\ttrain loss:', '{:.4f}'.format(loss.item()),
                  '\ttime:', '{:.4f}'.format((time.perf_counter()-time_1)*print_feq), 's')
    print('train_loss',train_loss)
    
    model.eval();
    valid_loss = []
    for step, (img, boxes, labels) in enumerate(val_dal):
        img = img.to(device)
        boxes = [box.to(device) for box in boxes]
        labels = [label.to(device) for label in labels]
        pred_loc, pred_sco = model(img)
        loss = criterion(pred_loc, pred_sco, boxes, labels)
        valid_loss.append(loss.item())
    print('val_loss',valid_loss)

    tr_loss_tot.append(np.mean(train_loss))
    vl_loss_tot.append(np.mean(valid_loss))   
    print('epoch:', epoch, '/', EPOCHS,
            '\ttrain loss:', '{:.4f}'.format(np.mean(train_loss)),
            '\tvalid loss:', '{:.4f}'.format(np.mean(valid_loss)))



torch.save(model.state_dict(), "/home/users/b/bozianu/work/SSD/SSD/models/"+"SSD_model_{}_coco.pth".format(EPOCHS))


plt.figure()
x_axis = torch.arange(EPOCHS)
plt.plot(x_axis,tr_loss_tot,'--',label='training loss')
plt.plot(x_axis,vl_loss_tot,'--',label='val loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Arbitrary Loss Units')
save_at = "/home/users/b/bozianu/work/SSD/SSD/inference/"+"SSD_model_{}_coco/".format(EPOCHS)
if not os.path.exists(save_at):
    os.makedirs(save_at)
plt.savefig(save_at+"traininglosses.png")










