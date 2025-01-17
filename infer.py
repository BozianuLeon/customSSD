import torch
import numpy as np

import time
import os
from statistics import mean

import models
import data

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
# MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496
MIN_CELLS_ETA,MAX_CELLS_ETA = -2.5, 2.5


config = {
    "seed"       : 0,
    "device"     : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "NW"         : 2,
    "BS"         : 4,
    "WD"         : 0.01,
    "n_epochs"   : 14,
    "max_num"    : 150,
}
torch.manual_seed(config["seed"])


# dataset = data.CustomDataset(annotation_file="/home/users/b/bozianu/work/data/mu200/anns_central_jets_20GeV.json")
dataset = data.CustomDataset(annotation_file="/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/central_2sig_images/anns_central_jets_ttbar.json")
train_len = int(0.78 * len(dataset))
val_len = int(0.02 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
print('\ttrain / val / test size : ',train_len,'/',val_len,'/',test_len,'\n')

train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=True, drop_last=True, num_workers=config["NW"])
val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=False, drop_last=True, num_workers=config["NW"])
test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=False, drop_last=True, num_workers=config["NW"])


# load trained model
model = models.SSD(backbone_name="uconvnext_central",in_channels=5)
model = model.to(config["device"]) 
model_name = "jetSSD_{}_{}e".format(model.backbone_name,config["n_epochs"])
model_save_path = f"./saved_models/{model_name}.pth"
model.load_state_dict(torch.load(model_save_path, map_location=torch.device(config["device"])))
total_params = sum(p.numel() for p in model.parameters())
print(model.backbone_name,f'!total \t{total_params:,} parameters.\n')

model.eval()

# default prior boxes
dboxes = data.DefaultBoxes(figsize=(24,63),scale=(3.84,4.05),step_x=1,step_y=1) 
print("Generated prior boxes, ",dboxes.dboxes.shape, ", default boxes")

# encoder 
encoder = data.Encoder(dboxes)




save_loc = "./cache/" + model_name + "/" + time.strftime("%Y%m%d-%H") + "/"
if not os.path.exists(save_loc):
    os.makedirs(save_loc)

# let's infer on all events in the test set and store the results in a numpy structured array
# with the following data types:
# event_no: int, h5file: int, img: numpy array?, ground truth boxes: list, predicted_boxes: list, predicted_scores: list, predicted_pt (sumpool): list, extent
beginning = time.perf_counter()
dt = np.dtype([('event_no', 'i4'), ('h5file', 'S2'), ('h5event', 'i4'), ('extent', 'f8', (4)),  #S2 for a string of length exactly 2
                ('t_boxes', 'f4', (250,4)), ('t_pt', 'f4', (250)), 
                ('p_boxes', 'f4', (config["max_num"], 4)), ('p_scores', 'f4', (config["max_num"])), ('p_pt', 'f4', (config["max_num"]))])
BS = config["BS"]
Large = np.zeros((len(test_loader)*BS), dtype=dt)    
with torch.inference_mode():
    for step, (batch_imgs,targets) in enumerate(test_loader):
        img_tensor = batch_imgs.to(config["device"]).float()

        locs,conf,ptmap = model(img_tensor)

        # define NMS scriteria, confidence threshold
        output = encoder.decode_batch(locs, conf, ptmap, 
                                        iou_thresh=0.25, #NMS
                                        confidence=0.3, #conf threshold
                                        max_num=config["max_num"]) #155

        boxes, labels, scores, pts = zip(*output)

        #remove from GPU
        tru_boxes,extents,h5files,h5events,event_nos,tru_pt = [], [], [], [], [], []
        det_boxes, det_scores, det_pts = [], [], []
        for i in range(BS):
            extent_i = targets[i]["extent"].detach().cpu().numpy()
            tru_boxes_ext = targets[i]['boxes'].detach().cpu().numpy()
            tru_boxes_ext[:,(0,2)] = (tru_boxes_ext[:,(0,2)]*((extent_i[1]-extent_i[0])/img_tensor[i].shape[2]))+extent_i[0]
            tru_boxes_ext[:,(1,3)] = (tru_boxes_ext[:,(1,3)]*((extent_i[3]-extent_i[2])/img_tensor[i].shape[1]))+extent_i[2]
            tru_pts = targets[i]['jet_pt']

            tru_boxes.append(tru_boxes_ext)
            tru_pt.append(targets[i]['jet_pt'])
            extents.append(extent_i)
            h5files.append(targets[i]["h5file"])
            h5events.append(targets[i]["h5event"])
            event_nos.append(targets[i]["event_no"])

            # make boxes cover extent
            det_boxes_scr = scores[i].detach().cpu().numpy()
            det_boxes_pts = pts[i].detach().cpu().numpy()
            det_boxes_ext = boxes[i].detach().cpu().numpy()
            det_boxes_ext[:,(0,2)] = (det_boxes_ext[:,(0,2)]*((extent_i[1]-extent_i[0])))+extent_i[0]
            det_boxes_ext[:,(1,3)] = (det_boxes_ext[:,(1,3)]*((extent_i[3]-extent_i[2])))+extent_i[2]

            # remember ALL targets have width/height 0.8
            # mask out boxes that have width and height > 1.3 (== radius >0.65)
            mask_too_big = (det_boxes_ext[:,2] - det_boxes_ext[:,0] < 1.3) & (det_boxes_ext[:,3] - det_boxes_ext[:,1] < 1.3)
            det_boxes_ext = det_boxes_ext[mask_too_big]
            det_boxes_scr = det_boxes_scr[mask_too_big]
            det_boxes_pts = det_boxes_pts[mask_too_big]
            # new!
            # mask out boxes that have  height <0.5 (== radius <0.25)
            mask_too_small = (det_boxes_ext[:,3] - det_boxes_ext[:,1] > 0.5)
            det_boxes_ext = det_boxes_ext[mask_too_small]
            det_boxes_scr = det_boxes_scr[mask_too_small]
            det_boxes_pts = det_boxes_pts[mask_too_small]

            # # new! secondary low score mask
            mask_low_score = det_boxes_scr > 0.95
            det_boxes_ext = det_boxes_ext[mask_low_score]
            det_boxes_pts = det_boxes_pts[mask_low_score]
            det_boxes_scr = det_boxes_scr[mask_low_score]

            det_boxes.append(det_boxes_ext)
            det_scores.append(det_boxes_scr)
            det_pts.append(det_boxes_pts)

            #######
            import matplotlib.pyplot as plt
            import matplotlib
            # det_boxes_ext,det_boxes_scr,det_boxes_pts = wrap_check_NMS3(det_boxes_ext,det_boxes_scr,det_boxes_pts,iou_thresh=0.3)
            # tru_boxes_ext,tru_pts = wrap_check_truth2(torch.tensor(tru_boxes_ext),torch.tensor(targets[i]['jet_pt']),MIN_CELLS_PHI,MAX_CELLS_PHI)
            f,ax = plt.subplots(1,1,figsize=(10,12))   
            img = img_tensor[i].detach().cpu().numpy()
            # img = original_images[i].detach().cpu().numpy()
            ax.imshow(img[0],cmap='binary_r',extent=extent_i,origin='lower')
        
            ax.axhline(y=MIN_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
            ax.axhline(y=MAX_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
    
            for k in range(len(tru_boxes_ext)):
                bbx,pt = tru_boxes_ext[k],tru_pts[k]
                x,y=float(bbx[0]),float(bbx[1])
                w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
                ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1.8,ec='limegreen',fc='none'))
                ax.text(x+0.05,y+h-0.15, f"{pt:.0f}",color='limegreen',fontsize=8)

            for j in range(len(det_boxes_ext)):
                bbx,scr,pt = det_boxes_ext[j],det_boxes_scr[j],det_boxes_pts[j]
                x,y=float(bbx[0]),float(bbx[1])
                w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
                ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1.9,ec='red',fc='none'))
                ax.text(x+w-0.3,y+h-0.15, f"{scr.item():.2f}",color='red',fontsize=8)
                ax.text(x+0.05,y+h/20, f"{pt.item():.0f}",color='red',fontsize=8)

            ax.set(xlabel='$\eta$',ylabel='$\phi$',xlim=(extent_i[0],extent_i[1]),ylim=(extent_i[2],extent_i[3]))
            plt.tight_layout()
            f.savefig(save_loc+f'ex-NMS-{step*BS + i}-ttbar.png',dpi=400)
            plt.close()
            print(step*BS + i)
            print("\t",len(tru_boxes_ext),len(det_boxes_ext),len(det_boxes_pts))
            if (step*BS + i) == 16:
                quit()

        print(step)

        dataset_idx = step*BS
        Large['event_no'][dataset_idx:dataset_idx+BS] = event_nos
        Large['h5file'][dataset_idx:dataset_idx+BS] = h5files
        Large['h5event'][dataset_idx:dataset_idx+BS] = h5events
        Large['extent'][dataset_idx:dataset_idx+BS] = extents  
    
        t_boxes = [np.pad(trub, ((0,250-len(trub)),(0,0)), 'constant', constant_values=(0)) for trub in tru_boxes]
        t_pt = [np.pad(trub, ((0,250-len(trub))), 'constant', constant_values=(0)) for trub in tru_pt]
        p_boxes = [np.pad(preb, ((0,config["max_num"]-len(preb)),(0,0)), 'constant', constant_values=(0)) for preb in det_boxes]
        p_scores = [np.pad(pres, ((0,config["max_num"]-len(pres))), 'constant', constant_values=(0)) for pres in det_scores]
        p_pts = [np.pad(prept, ((0,config["max_num"]-len(prept))), 'constant', constant_values=(0)) for prept in det_pts]
    
        Large['t_boxes'][dataset_idx:dataset_idx+BS] = t_boxes   
        Large['t_pt'][dataset_idx:dataset_idx+BS] = t_pt   
        Large['p_boxes'][dataset_idx:dataset_idx+BS] = p_boxes   
        Large['p_scores'][dataset_idx:dataset_idx+BS] = p_scores 
        Large['p_pt'][dataset_idx:dataset_idx+BS] = p_pts 

end = time.perf_counter()      
print(f"Time taken for entire test set: {(end-beginning)/60:.3f} mins, (or {(end-beginning):.3f}s), average {(end-beginning)/test_len:.4f} per image")

print('\n\n')
with open(save_loc+'struc_array.npy', 'wb') as f:
    print('Saving...')
    np.save(f, Large)
