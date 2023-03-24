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
from utils.dataset import SSDDataset

print(os.listdir("/home/users/b/bozianu/work/SSD/SSD/data/input"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #globally define






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





#######################################################################################################

#######################################################################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 7
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


train_ds = SSDDataset(all_img_name[:int(len(all_img_name)*0.8)])
train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=train_ds.collate_fn)

valid_ds = SSDDataset(all_img_name[int(len(all_img_name)*0.8):int(len(all_img_name)*0.9)])
valid_dl = DataLoader(valid_ds, batch_size=BS, shuffle=True, collate_fn=valid_ds.collate_fn)




#######################################################################################################




from tqdm import tqdm
import time


tr_loss_tot = []
vl_loss_tot = []
for epoch in range(EPOCHS):
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
    print('epoch:', epoch, '/', EPOCHS,
            '\ttrain loss:', '{:.4f}'.format(np.mean(train_loss)),
            '\tvalid loss:', '{:.4f}'.format(np.mean(valid_loss)))





plt.figure()
x_axis = torch.arange(EPOCHS)
plt.plot(x_axis,tr_loss_tot,'--',label='training loss')
plt.plot(x_axis,vl_loss_tot,'--',label='val loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Arbitrary Loss Units')
#plt.savefig('losses.png')
plt.savefig("/home/users/b/bozianu/work/SSD/SSD/inference/"+"SSD_model_{}/traininglosses.png".format(EPOCHS))


torch.save(model.state_dict(), "/home/users/b/bozianu/work/SSD/SSD/models/"+"SSD_model_{}.pth".format(EPOCHS))








