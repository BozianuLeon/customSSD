import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import time
import os
from statistics import mean
import argparse

import models
import data


parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, required=True, help='Name of backbone model (e.g resnet50)',)
parser.add_argument('-e','--epochs', type=int, required=True, help='Number of training epochs',)
parser.add_argument('-bs','--batch_size', nargs='?', const=1, default=1, type=int, help='Batch size to be used')
parser.add_argument('-nw','--num_workers', nargs='?', const=2, default=2, type=int, help='Number of worker CPUs')
parser.add_argument('-in','--input_file', type=str, required=True, help='Path to annotations file (json file)',)
parser.add_argument('-p','--proc', type=str, required=True, help='Type of process (JZ1,JZ2,JZ3,ttbar)',)
parser.add_argument('--model_dir', type=str, required=True, help='Path to saved models directory',)
parser.add_argument('-out','--output_dir',nargs='?', const='./cache/', default='./cache/', type=str, help='Path to directory containing struc_array.npy',)
args = parser.parse_args()


config = {
    "seed"       : 0,
    "device"     : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "NW"         : args.num_workers,
    "BS"         : 1,
    "n_epochs"   : int(args.epochs),
    "max_num"    : 150,
}

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
# MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496
MIN_CELLS_ETA,MAX_CELLS_ETA = -2.5, 2.5
EXTENT = [-2.4999826, 2.4999774, -6.217388, 6.2180176]
torch.manual_seed(config["seed"])

dataset = data.CustomDataset(annotation_file=args.input_file,truth_info=False) # set truth info to false, won't have it/return it during deployment
time_test_dataset, _ = torch.utils.data.random_split(dataset, [1000, len(dataset)-1000])
print('\tdatatset size for timing test : ',len(time_test_dataset),'\n')

dataloader = torch.utils.data.DataLoader(time_test_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=False, num_workers=config["NW"]) # force batch size to be 1


# load trained model
model = models.SSD(backbone_name=args.backbone,in_channels=5,diamond_mask=True)
model = model.to(config["device"]) 
model_name = "jetSSD_{}_{}e".format(model.backbone_name,config["n_epochs"])
model_save_path = args.model_dir + f"/{model_name}.pth"
model.load_state_dict(torch.load(model_save_path, map_location=torch.device(config["device"])))
total_params = sum(p.numel() for p in model.parameters())
print(model.backbone_name,f'!total \t{total_params:,} parameters.\n')

model.eval()

# default prior boxes
dboxes = data.DefaultBoxes(figsize=(24,63),scale=(3.84,4.05),step_x=1,step_y=1) 
print("Generated prior boxes, ",dboxes.dboxes.shape, ", default boxes")

# encoder 
encoder = data.Encoder(dboxes)


save_loc = args.output_dir + "/" + model_name + "/" + args.proc + "/" + time.strftime("%Y%m%d-%H") + "/"
print("Save location: ", save_loc)
if not os.path.exists(save_loc): os.makedirs(save_loc)
time_per_event, time_only_inference = list(), list()
n_jets_per_event, n_pred_per_event = list(), list()
beginning = time.perf_counter()
with torch.inference_mode():
    for step, (batch_imgs,targets) in enumerate(dataloader):
        start_time = time.perf_counter()
        img_tensor = batch_imgs.to(config["device"]).float() # send data to GPU

        locs,conf,ptmap = model(img_tensor) # run inference

        # define NMS scriteria, confidence threshold
        output = encoder.decode_batch(locs, conf, ptmap, 
                                        iou_thresh=0.25, #NMS
                                        confidence=0.45, #conf threshold
                                        max_num=config["max_num"]) #155
        end_time_only = time.perf_counter()

        boxes, labels, scores, pts = zip(*output)
        for i in range(config["BS"]):
            #remove from GPU (sad!)
            # # tar_boxes_ext = targets[i]['akt_boxes'].detach().cpu().numpy()
            # # tar_boxes_ext[:,(0,2)] = (tar_boxes_ext[:,(0,2)]*((EXTENT[1]-EXTENT[0])/img_tensor[i].shape[2]))+EXTENT[0]
            # # tar_boxes_ext[:,(1,3)] = (tar_boxes_ext[:,(1,3)]*((EXTENT[3]-EXTENT[2])/img_tensor[i].shape[1]))+EXTENT[2]
            tar_pts = targets[i]['jet_pt']
            n_jets_per_event.append(len(tar_pts))

            # make pred boxes cover extent
            det_boxes_scr = scores[i].detach().cpu().numpy()
            det_boxes_pts = pts[i].detach().cpu().numpy()
            det_boxes_ext = boxes[i].detach().cpu().numpy()
            n_pred_per_event.append(len(det_boxes_scr))
            det_boxes_ext[:,(0,2)] = (det_boxes_ext[:,(0,2)]*((EXTENT[1]-EXTENT[0])))+EXTENT[0]
            det_boxes_ext[:,(1,3)] = (det_boxes_ext[:,(1,3)]*((EXTENT[3]-EXTENT[2])))+EXTENT[2]

        end_time = time.perf_counter()
        time_per_event.append(end_time-start_time)
        time_only_inference.append(end_time_only-start_time)

        print(step)

end = time.perf_counter()      
print(f"Time taken for entire test set: {(end-beginning)/60:.3f} mins, (or {(end-beginning):.3f}s), average {(end-beginning)/len(dataloader):.4f} per image")




mean = np.mean(time_per_event)
std_dev = np.std(time_per_event[1:])
median = np.median(time_per_event)
print(f"Mean {mean:.6f} vs median {median:.6f}, std {std_dev:.6f}, {np.std(time_per_event)}")
print("Top 10: ", torch.topk(torch.tensor(time_per_event),10)[0])
print("First 10:", [f"{time_per_event[x]:.4f}" for x in range(10)])
print("Bottom 10: ", -1*torch.topk(-torch.tensor(time_per_event),10)[0])
# bins = np.linspace(0, max(time_per_event), 16) 
bins = np.linspace(np.percentile(time_per_event, 1), np.percentile(time_per_event, 99), 16) 
plt.figure()
plt.hist(time_per_event, bins=bins, edgecolor='black', color='red')
plt.xlabel('Time [s]')
plt.xticks(rotation = 45)
plt.title(f'Inference execution time per event ({args.proc})')
plt.text(0.95, 0.95, f'Mean: {mean:.6f}s\nStd Dev: {std_dev:.7f}\n Median: {median:.6f}s', 
        horizontalalignment='right', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.text(0.95, 0.73, f'Incl. data CPU->GPU->CPU', 
        horizontalalignment='right', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=9)
plt.text(0.95, 0.77, f'Model params {total_params:.0f}', 
        horizontalalignment='right', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=9)
plt.tight_layout()
plt.savefig(save_loc+f"time_per_event.png",dpi=400)


plt.figure()
plt.hist(time_only_inference, bins=bins, edgecolor='black', color='red')
plt.text(0.95, 0.95, f'Mean: {np.mean(time_only_inference):.6f}\nStd Dev: {np.std(time_only_inference):.7f}', 
        horizontalalignment='right', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.xlabel('Time [s]')
plt.title('Inference execution time per event')
plt.savefig(save_loc+f"time_per_event_infer_only.png",dpi=400)


plt.figure()
bins_x = np.linspace(np.percentile(time_per_event, 1), np.percentile(time_per_event, 90), 30) 
bins_y = 30
plt.hist2d(time_per_event, n_jets_per_event, bins=(bins_x,bins_y), cmap='plasma')
plt.xlabel('Time Taken [s]')
plt.xticks(rotation = 45)
plt.ylabel('Number of Jets')
plt.colorbar(label='# Events')
plt.tight_layout()
plt.savefig(save_loc+f"time_vs_jets.png",dpi=400)


plt.figure()
bins_x = np.linspace(np.percentile(time_per_event, 1), np.percentile(time_per_event, 90), 30) 
bins_y = 30
plt.hist2d(time_per_event, n_pred_per_event, bins=(bins_x,bins_y), cmap='plasma')
plt.xlabel('Time Taken [s]')
plt.xticks(rotation = 45)
plt.ylabel('Number of Predictions')
plt.colorbar(label='# Events')
plt.tight_layout()
plt.savefig(save_loc+f"time_vs_preds.png",dpi=400)



