print('Starting inference script.')
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import os
import sys
import time

# # caution: path[0] is reserved for script path (or '' in REPL)
# sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')
from ssd import SSD
from utils.dataset import SSDRealDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device used: ',device)
# hyperparameters
BS = 16
NW = 2

# initialise dataset as in train.py, then split into train/val/test
das = SSDRealDataset(
                    #  annotation_json='/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/cluster5GeV_annotations.json',
                    #  image_directory='/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/cell_images_dev/',
                     annotation_json='/home/users/b/bozianu/work/data/pileup50k/anns5GeV2sig.json',
                     image_directory='/home/users/b/bozianu/work/data/pileup50k/imgs5GeV2sig/',
                     is_test=True)
print('Images in dataset:',len(das))

train_size = int(0.78 * len(das))
val_size = int(0.02 * len(das))
test_size = len(das) - train_size - val_size
print('\ttrain / val / test size : ',train_size,'/',val_size,'/',test_size)

torch.random.manual_seed(1)
train_das, val_das, test_das = torch.utils.data.random_split(das,[train_size,val_size,test_size])

train_dal = DataLoader(train_das, batch_size=BS, shuffle=True, collate_fn=das.collate_fn,num_workers=NW,prefetch_factor=NW)
val_dal = DataLoader(val_das,batch_size=BS,shuffle=True,collate_fn=das.collate_fn,num_workers=NW,prefetch_factor=NW)
test_dal = DataLoader(test_das,batch_size=BS,shuffle=False,collate_fn=das.collate_fn,num_workers=NW,prefetch_factor=NW)

##########################################################################################
model_name = "SSD_50k_11e_new"
model = SSD(in_channels=7,device=device,pretrained_vgg=False)
model.load_state_dict(torch.load("/home/users/b/bozianu/work/SSD/SSD/cached_models/{}.pth".format(model_name),map_location=torch.device(device)))
model.eval()

save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/" + model_name + "/" + time.strftime("%Y%m%d-%H") + "/"
if not os.path.exists(save_loc):
    os.makedirs(save_loc)


# this is actually saved in a numpy structured array with the following data type
# event_no: int, h5file: int, img: numpy array?, ground truth boxes: list, predicted_boxes: list, predicted_scores: list, extent
dt = np.dtype([('event_no', 'i4'), ('h5file', 'S2'), ('extent', 'f8', (4)),  #S2 for a string of length exactly 2
               ('t_boxes', 'f4', (250,4)), ('p_boxes', 'f4', (155, 4)), ('p_scores', 'f4', (155))])
Large = np.zeros((len(test_dal)*BS), dtype=dt)               
with torch.inference_mode():
    for step, (batch_imgs, tru_boxes, extents, h5files, h5event_nos) in enumerate(test_dal):
        img_tensor = batch_imgs.to(device)

        predicted_locs, predicted_scores = model(img_tensor)
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=0.35, max_overlap=0.2, top_k=155)
        
        #remove from gpu
        tru_boxes = [box.detach().to("cpu").numpy() for box in tru_boxes]
        extents = [ex.detach().to("cpu").numpy() for ex in extents]
        det_boxes = [box.detach().to("cpu").numpy() for box in det_boxes]   
        det_scores = [score.detach().to("cpu").numpy() for score in det_scores]   

        print(step)
        dataset_idx = step*BS
        Large['event_no'][dataset_idx:dataset_idx+BS] = h5event_nos
        Large['h5file'][dataset_idx:dataset_idx+BS] = h5files
        Large['extent'][dataset_idx:dataset_idx+BS] = extents   

        t_boxes = [np.pad(trub, ((0,250-len(trub)),(0,0)), 'constant', constant_values=(0)) for trub in tru_boxes]
        p_boxes = [np.pad(preb, ((0,155-len(preb)),(0,0)), 'constant', constant_values=(0)) for preb in det_boxes]
        p_scores = [np.pad(pres, ((0,155-len(pres))), 'constant', constant_values=(0)) for pres in det_scores]
    
        Large['t_boxes'][dataset_idx:dataset_idx+BS] = t_boxes   
        Large['p_boxes'][dataset_idx:dataset_idx+BS] = p_boxes   
        Large['p_scores'][dataset_idx:dataset_idx+BS] = p_scores   


print('\n\n')
with open(save_loc+'struc_array.npy', 'wb') as f:
    print('Saving...')
    np.save(f, Large)







#needs a few example images
