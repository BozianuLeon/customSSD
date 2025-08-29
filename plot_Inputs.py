import torch
import numpy as np

import time
import os
from statistics import mean
import argparse

import models
import data
from metrics.utils import iou_box_matching, dR_box_matching, wrap_check_truth3, wrap_check_NMS3


parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, required=True, help='Name of backbone model (e.g resnet50)',)
parser.add_argument('-e','--epochs', type=int, required=True, help='Number of training epochs',)
parser.add_argument('-bs','--batch_size', nargs='?', const=8, default=8, type=int, help='Batch size to be used')
parser.add_argument('-nw','--num_workers', nargs='?', const=2, default=2, type=int, help='Number of worker CPUs')
parser.add_argument('-in','--input_file', type=str, required=True, help='Path to annotations file (json file)',)
parser.add_argument('-p','--proc', type=str, required=True, help='Type of process (JZ1,JZ2,JZ3,ttbar)',)
parser.add_argument('--model_dir', type=str, required=True, help='Path to saved models directory',)
parser.add_argument('-out','--output_dir',nargs='?', const='./cache/', default='./cache/', type=str, help='Path to directory containing plots, e.g plotting/figs/jetSSD_*/',)
args = parser.parse_args()


config = {
    "seed"       : 0,
    "device"     : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "NW"         : args.num_workers,
    "BS"         : int(args.batch_size),
    "n_epochs"   : int(args.epochs),
    "max_num"    : 150,
}

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
# MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496
MIN_CELLS_ETA,MAX_CELLS_ETA = -2.5, 2.5
torch.manual_seed(config["seed"])

dataset = data.CustomDataset(annotation_file=args.input_file,truth_info=True)
train_len = int(0.01 * len(dataset))
val_len   = int(0.01 * len(dataset))
test_len  = len(dataset) - train_len - val_len
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
print('\ttrain / val / test size : ',train_len,'/',val_len,'/',test_len,'\n')

train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=True, drop_last=True, num_workers=config["NW"])
val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=False, drop_last=True, num_workers=config["NW"])
test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=False, drop_last=True, num_workers=config["NW"])


# load trained model
# model = models.SSD(backbone_name=args.backbone,in_channels=5,diamond_mask=True)
model = models.SSD(backbone_name=args.backbone,in_channels=5)
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



save_loc = args.output_dir + "/" + model_name + "/" + args.proc + "/" + "presentation_plots/"
print("Save location: ", save_loc)
if not os.path.exists(save_loc): os.makedirs(save_loc)

# let's infer on all events in the test set and store the results in a numpy structured array
# with the following data types:
# event_no: int, h5file: int, img: numpy array?, ground truth boxes: list, predicted_boxes: list, predicted_scores: list, predicted_pt (sumpool): list, extent
beginning = time.perf_counter()
dt = np.dtype([('event_no', 'i4'), ('event_weight', 'f4'), ('h5file', 'S2'), ('h5event', 'i4'), ('extent', 'f8', (4)),  #S2 for a string of length exactly 2
                ('tar_boxes', 'f4', (250,4)), ('tar_pt', 'f4', (250)), 
                ('tru_boxes', 'f4', (100,4)), ('tru_pt', 'f4', (100)), 
                ('p_boxes', 'f4', (config["max_num"], 4)), ('p_scores', 'f4', (config["max_num"])), ('p_pt', 'f4', (config["max_num"]))])
BS = config["BS"]
Large = np.zeros((len(test_loader)*BS), dtype=dt)    
with torch.inference_mode():
    for step, (batch_imgs,targets) in enumerate(test_loader):
        # if step in [0]: continue
        img_tensor = batch_imgs.to(config["device"]).float()

        locs,conf,ptmap = model(img_tensor)

        # define NMS scriteria, confidence threshold
        output = encoder.decode_batch(locs, conf, ptmap, 
                                        iou_thresh=0.25, #NMS
                                        confidence=0.45, #conf threshold
                                        max_num=config["max_num"]) #155

        boxes, labels, scores, pts = zip(*output)

        #remove from GPU
        tru_boxes,extents,h5files,h5events,event_nos,tru_pt = [], [], [], [], [], []
        det_boxes, det_scores, det_pts = [], [], []
        for i in range(BS):
            # if i in np.arange(0,11): continue
            extent_i = targets[i]["extent"].detach().cpu().numpy()
            # make target/truth boxes cover extent
            tar_boxes_ext = targets[i]['akt_boxes'].detach().cpu().numpy()
            tar_boxes_ext[:,(0,2)] = (tar_boxes_ext[:,(0,2)]*((extent_i[1]-extent_i[0])/img_tensor[i].shape[2]))+extent_i[0]
            tar_boxes_ext[:,(1,3)] = (tar_boxes_ext[:,(1,3)]*((extent_i[3]-extent_i[2])/img_tensor[i].shape[1]))+extent_i[2]
            tar_pts = targets[i]['akt_jet_pt']

            # tru_boxes_ext = targets[i]['truth_boxes'].detach().cpu().numpy()
            # tru_boxes_ext[:,(0,2)] = (tru_boxes_ext[:,(0,2)]*((extent_i[1]-extent_i[0])/img_tensor[i].shape[2]))+extent_i[0]
            # tru_boxes_ext[:,(1,3)] = (tru_boxes_ext[:,(1,3)]*((extent_i[3]-extent_i[2])/img_tensor[i].shape[1]))+extent_i[2]
            # tru_pts = targets[i]['truth_jet_pt']

            # make pred boxes cover extent
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


            #########################################################################################



            import matplotlib.pyplot as plt
            import matplotlib
            det_boxes_ext,det_boxes_scr,det_boxes_pts = wrap_check_NMS3(det_boxes_ext,det_boxes_scr,det_boxes_pts,iou_thresh=0.3)
            tar_boxes_ext,tar_boxes_pt = wrap_check_truth3(torch.tensor(tar_boxes_ext),torch.tensor(tar_pts),MIN_CELLS_PHI,MAX_CELLS_PHI)

            f,ax = plt.subplots(1,1,figsize=(8,5.5))   
            img = img_tensor[i].detach().cpu().numpy()
            # cmap = plt.get_cmap('magma')
            cmap = plt.get_cmap('binary_r')
            cmap.set_bad(color='black')
            cax = ax.imshow(img[0],cmap=cmap,extent=extent_i,origin='lower')
            # cax = ax.imshow(img[0], cmap=cmap, extent=extent_i, origin='lower', norm=matplotlib.colors.LogNorm())
            cbar = f.colorbar(cax, ax=ax, shrink=0.84)             

            # ax.axhline(y=MIN_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=1.1)
            # ax.axhline(y=MAX_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=1.1)
            # ax.axvline(x=-2.1, color='lemonchiffon', alpha=0.9, linestyle='dashdot',lw=0.7)
            # ax.axvline(x=2.1, color='lemonchiffon', alpha=0.9, linestyle='dashdot',lw=0.7)

            for k in range(len(tar_boxes_ext)):
                bbx,pt = tar_boxes_ext[k],tar_boxes_pt[k]
                x,y=float(bbx[0]),float(bbx[1])
                w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
                ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1.5,ec='limegreen',fc='none'))
                ax.text(x+0.03,y+h-0.21, f"{pt:.0f}",color='limegreen',fontsize=9)

            for j in range(len(det_boxes_ext)):
                bbx,scr,pt = det_boxes_ext[j],det_boxes_scr[j],det_boxes_pts[j]
                x,y=float(bbx[0]),float(bbx[1])
                w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
                ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1.4,ec='red',fc='none'))
                # ax.text(x+w,y+h, f"{scr.item():.2f}",color='white',fontsize=6)
                ax.text(x+0.03,y+0.036, f"{pt.item():.0f}",color='red',fontsize=9)

            # rectangle = matplotlib.patches.Rectangle((-2.1, -np.pi), 4.2, 2*np.pi, 
            #                             linewidth=4, 
            #                             edgecolor='gold', 
            #                             facecolor='none',  
            #                             alpha=0.85)  
            rectangle = matplotlib.patches.Rectangle((-2.1, -6.2), 4.2, 12.4, 
                                        linewidth=3, 
                                        edgecolor='gold', 
                                        facecolor='none',  
                                        alpha=0.85)  
            ax.add_patch(rectangle)
            ax.axvspan(-5, extent_i[0], color='black', alpha=1)
            ax.axvspan(extent_i[1], 15, color='black', alpha=1) # Fill right of the maximum extent

            # ax.set(xlabel='$\eta$',ylabel='$\phi$',xlim=(extent_i[0],extent_i[1]),ylim=(extent_i[2],extent_i[3]))
            # ax.set(xlim=(-4.823496, 4.823496),ylim=(extent_i[2],extent_i[3]))
            ax.set(xlim=(-4.823496, 4.823496),ylim=(-np.pi,np.pi))
            plt.tight_layout()
            f.savefig(save_loc+f'ex-NMS-{i}.png',dpi=400)
            # f.savefig(save_loc+f'ex-NMS-0.png',dpi=400)
            # f.savefig(save_loc+f'ex-NMS-{step*BS + i}.png',dpi=400)
            # f.savefig(save_loc+f'truth-{step*BS + i}.png',dpi=400)
            plt.close()


            # f,ax = plt.subplots(1,1,figsize=(8,5.5))   
            # img = img_tensor[i].detach().cpu().numpy()
            # # cmap = plt.get_cmap('magma')
            # cmap = plt.get_cmap('binary_r')
            # cmap.set_bad(color='black')
            # cax = ax.imshow(img[0],cmap=cmap,extent=extent_i,origin='lower')
            # cbar = f.colorbar(cax, ax=ax, shrink=0.84)             

            # ax.axhline(y=MIN_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=1.1)
            # ax.axhline(y=MAX_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=1.1)

            # for k in range(len(tar_boxes_ext)):
            #     bbx,pt = tar_boxes_ext[k],tar_boxes_pt[k]
            #     x,y=float(bbx[0]),float(bbx[1])
            #     w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
            #     ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1.5,ec='limegreen',fc='none'))
            #     ax.text(x+0.017,y+(7*h/10), f"{pt:.0f}",color='limegreen',fontsize=6)

            # rectangle = matplotlib.patches.Rectangle((-2.1, -6.2), 4.2, 12.4, 
            #                             linewidth=3, 
            #                             edgecolor='gold', 
            #                             facecolor='none',  
            #                             alpha=0.85)  
            # ax.add_patch(rectangle)
            # ax.axvspan(-5, extent_i[0], color='black', alpha=1)
            # ax.axvspan(extent_i[1], 15, color='black', alpha=1) # Fill right of the maximum extent

            # # ax.set(xlabel='$\eta$',ylabel='$\phi$',xlim=(extent_i[0],extent_i[1]),ylim=(extent_i[2],extent_i[3]))
            # ax.set(xlim=(-4.823496, 4.823496),ylim=(extent_i[2],extent_i[3]))
            # # ax.set(xlabel='$\eta$',ylabel='$\phi$',xlim=(extent_i[0],extent_i[1]),ylim=(-np.pi,np.pi))
            # plt.tight_layout()
            # f.savefig(save_loc+f'ex-TARGET-0.png',dpi=400)
            # plt.close()




            # f,ax = plt.subplots(1,1,figsize=(8,5.5))   
            # new_sumpool = model.sumpool.to('cpu')
            # H_output_map = new_sumpool(torch.tensor(img))
            # print(img.shape,H_output_map.shape)
            # cax = ax.imshow(H_output_map[0].detach().numpy(),cmap='binary_r',extent=extent_i,origin='lower')
            # cbar = f.colorbar(cax, ax=ax)  
            # ax.axhline(y=MIN_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
            # ax.axhline(y=MAX_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
            # # for i in range(len(tar_boxes_ext)):
            # #     bbx, pt = tar_boxes_ext[i], pT[i]
            # #     x,y=float(bbx[0]),float(bbx[1])
            # #     w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
            # #     ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1.1,ec='limegreen',fc='none'))

            # rectangle = matplotlib.patches.Rectangle((-2.1, -6.2), 4.2, 12.4, 
            #                         linewidth=4, 
            #                         edgecolor='gold', 
            #                         facecolor='none',  
            #                         alpha=0.85)  
            # ax.add_patch(rectangle)
            # ax.axvspan(-5, extent_i[0], color='black', alpha=1)
            # ax.axvspan(extent_i[1], 15, color='black', alpha=1)

            # # ax.set(xlabel='$\eta$',ylabel='$\phi$',xlim=(extent_i[0],extent_i[1]),ylim=(extent_i[2],extent_i[3]))
            # # ax.set(xlim=(extent_i[0],extent_i[1]),ylim=(extent_i[2],extent_i[3]))
            # ax.set(xlim=(-4.823496, 4.823496),ylim=(extent_i[2],extent_i[3]))
            # f.savefig(save_loc+f'ex-POOL-0.png',dpi=400)
            # # f.savefig(save_loc+f'ex-POOL-{step*BS+i}.png',dpi=400)



            # # do network inputs plot
            # img = img_tensor[i].detach().cpu().numpy()
            # fig = plt.figure(figsize=(16, 9))
            # gs = fig.add_gridspec(2, 24)  # 2 rows, 6 columns grid
            # # First row: 2 plots, each spanning 3 columns to ensure equal size
            # ax1 = fig.add_subplot(gs[0, 4:12])  # First plot in the top row (spanning 3 columns)
            # ax2 = fig.add_subplot(gs[0, 12:20])  # Second plot in the top row (spanning 3 columns)
            # # Second row: 3 plots, each spanning 2 columns to match the top row width
            # ax3 = fig.add_subplot(gs[1, 0:8])  # First plot in the bottom row
            # ax4 = fig.add_subplot(gs[1, 8:16])  # Second plot in the bottom row
            # ax5 = fig.add_subplot(gs[1, 16:24])  # Third plot in the bottom row

            # cax = ax1.imshow(img[0], cmap='jet',extent=extent_i,origin='lower')
            # ax1.axvspan(-5, extent_i[0], color='black', alpha=1)
            # ax1.axvspan(extent_i[1], 10, color='black', alpha=1)
            # rectangle1 = matplotlib.patches.Rectangle((-2.1, -6.2), 4.2, 12.4, linewidth=4, edgecolor='gold', facecolor='none',  alpha=0.85)  
            # ax1.add_patch(rectangle1)
            # ax1.set(xlim=(-4.823496, 4.823496),ylim=(extent_i[2],extent_i[3]))
            # # cbar = fig.colorbar(cax, ax=ax1)  
            # cax = ax2.imshow(img[1], cmap='jet',extent=extent_i,origin='lower')
            # ax2.axvspan(-5, extent_i[0], color='black', alpha=1)
            # ax2.axvspan(extent_i[1], 10, color='black', alpha=1)
            # rectangle2 = matplotlib.patches.Rectangle((-2.1, -6.2), 4.2, 12.4, linewidth=4, edgecolor='gold', facecolor='none',  alpha=0.85)  
            # ax2.add_patch(rectangle2)
            # ax2.set(xlim=(-4.823496, 4.823496),ylim=(extent_i[2],extent_i[3]))
            # # cbar = fig.colorbar(cax, ax=ax2)  
            # cax = ax3.imshow(img[2], cmap='jet',extent=extent_i,origin='lower')
            # ax3.axvspan(-5, extent_i[0], color='black', alpha=1)
            # ax3.axvspan(extent_i[1], 10, color='black', alpha=1)
            # rectangle3 = matplotlib.patches.Rectangle((-2.1, -6.2), 4.2, 12.4, linewidth=4, edgecolor='gold', facecolor='none',  alpha=0.85)  
            # ax3.add_patch(rectangle3)
            # ax3.set(xlim=(-4.823496, 4.823496),ylim=(extent_i[2],extent_i[3]))
            # # cbar = fig.colorbar(cax, ax=ax3,pad=0.01)  
            # cax = ax4.imshow(img[3], cmap='jet',extent=extent_i,origin='lower')
            # ax4.axvspan(-5, extent_i[0], color='black', alpha=1)
            # ax4.axvspan(extent_i[1], 10, color='black', alpha=1)
            # rectangle4 = matplotlib.patches.Rectangle((-2.1, -6.2), 4.2, 12.4, linewidth=4, edgecolor='gold', facecolor='none',  alpha=0.85)  
            # ax4.add_patch(rectangle4)
            # ax4.set(xlim=(-4.823496, 4.823496),ylim=(extent_i[2],extent_i[3]))
            # # cbar = fig.colorbar(cax, ax=ax4)  
            # cax = ax5.imshow(img[4], cmap='jet',extent=extent_i,origin='lower')
            # ax5.axvspan(-5, extent_i[0], color='black', alpha=1)
            # ax5.axvspan(extent_i[1], 10, color='black', alpha=1)
            # rectangle5 = matplotlib.patches.Rectangle((-2.1, -6.2), 4.2, 12.4, linewidth=4, edgecolor='gold', facecolor='none',  alpha=0.85)  
            # ax5.add_patch(rectangle5)
            # ax5.set(xlim=(-4.823496, 4.823496),ylim=(extent_i[2],extent_i[3]))
            # # cbar = fig.colorbar(cax, ax=ax5)  
            # # for a in [ax1,ax2,ax3,ax3,ax4,ax5]:
            # #     a.tick_params(labelbottom=False, labelleft=False)  # Hide the x and y labels
            # #[H_sum_pt,H_max_pt,H_sum_signif,H_max_signif,H_max_noise]
            # # ax1.set_title('$\sum$ cell $p_T$',fontsize=14)
            # # ax1.set(xlabel='$\eta$',ylabel='$\phi$')
            # # ax1.set_xlim((-2.1,2.1))
            # # ax1.xaxis.label.set_size(14)
            # # ax1.yaxis.label.set_size(14)
            # # # ax2.set_title('Max cell $p_T$',fontsize=14)
            # # ax2.set(xlabel='$\eta$',ylabel='$\phi$')
            # # ax2.set_xlim((-2.1,2.1))
            # # ax2.xaxis.label.set_size(14)
            # # ax2.yaxis.label.set_size(14)
            # # # ax3.set_title('$\sum$ cell significance',fontsize=14)
            # # ax3.set(xlabel='$\eta$',ylabel='$\phi$')
            # # ax3.set_xlim((-2.1,2.1))
            # # ax3.xaxis.label.set_size(14)
            # # ax3.yaxis.label.set_size(14)
            # # # ax4.set_title('Max cell significance',fontsize=14)
            # # ax4.set(xlabel='$\eta$',ylabel='$\phi$')
            # # ax4.set_xlim((-2.1,2.1))
            # # ax4.xaxis.label.set_size(14)
            # # ax4.yaxis.label.set_size(14)
            # # # ax5.set_title('Max cell noise',fontsize=14)
            # # ax5.set(xlabel='$\eta$',ylabel='$\phi$')
            # # ax5.set_xlim((-2.1,2.1))
            # # ax5.xaxis.label.set_size(14)
            # # ax5.yaxis.label.set_size(14)
            # # hep.atlas.label(ax=ax5,label='Internal',data=False,lumi=None,loc=1)
            # fig.tight_layout()
            # fig.savefig(save_loc+f'ex-NMS-inp-0.png',dpi=400)
            # # fig.savefig(save_loc+f'ex-NMS-inp-{step*BS+i}.png',dpi=400)
            # plt.close()


            # quit()