import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import matplotlib
import matplotlib.pyplot as plt

import time
from statistics import mean
import argparse

import models
import data


parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, required=True, help='Name of backbone model (e.g resnet50)',)
parser.add_argument('-e','--epochs', type=int, required=True, help='Number of training epochs',)
parser.add_argument('-bs','--batch_size', nargs='?', const=8, default=8, type=int, help='Batch size to be used')
parser.add_argument('-nw','--num_workers', nargs='?', const=2, default=2, type=int, help='Number of worker CPUs')
parser.add_argument('-in','--input_file', type=str, required=True, help='Path to annotations file (json file)',)
args = parser.parse_args()


# https://pytorch.org/docs/stable/notes/randomness.html
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

config = {
    "seed"       : 0,
    "device"     : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "NW"         : args.num_workers,
    "BS"         : args.batch_size,
    "LR"         : 0.01,
    "WD"         : 0.01,
    "wup_epochs" : int(args.epochs/3),
    "n_epochs"   : int(args.epochs),
}
torch.manual_seed(config["seed"])


dataset = data.CustomDataset(annotation_file=args.input_file, rnd_flips=True)
train_len = int(0.78 * len(dataset))
val_len = int(0.02 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
print('\ttrain / val / test size : ',train_len,'/',val_len,'/',test_len,'\n')

train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=True, drop_last=True, num_workers=config["NW"])
val_loader   = torch.utils.data.DataLoader(val_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=False, drop_last=True, num_workers=config["NW"])
test_loader  = torch.utils.data.DataLoader(test_dataset, collate_fn=dataset.collate_fn, batch_size=config["BS"], shuffle=False, drop_last=True, num_workers=config["NW"])


# instantiate model
model = models.SSD(backbone_name=args.backbone,in_channels=5)
model = model.to(config["device"]) 
total_params = sum(p.numel() for p in model.parameters())
print(model.backbone_name, f'\t{total_params:,} total! parameters.\n')
        
# optimizers & learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=config["WD"], amsgrad=True)  
warmup_scheduler = LinearLR(optimizer, start_factor=1./3, end_factor=1.0, total_iters=config["wup_epochs"]) # Linear warmup
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config["n_epochs"] - config["wup_epochs"]) # Cosine Annealing after warmup
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[config["wup_epochs"]])

# default prior boxes
dboxes = data.DefaultBoxes(figsize=(24,63),scale=(3.84, 4.05),step_x=1,step_y=1) 
print("Generated prior boxes, ",dboxes.dboxes.shape, ", default boxes")

# encoder and loss
encoder = data.Encoder(dboxes)
loss = models.Loss(dboxes,scalar=1.0)


print('Starting training...')
for epoch in range(config["n_epochs"]):
    beginning = time.perf_counter()

    model.train()
    running_loss = list()
    for step, (images, target_dict) in enumerate(train_loader):
        # send data to gpu (annoying)
        images = images.to(config["device"],non_blocking=True)

        # forward pass
        plocs, plabel, ptmap = model(images) #plocs.shape(torch.Size([BS, 4, n_dfboxes])) and plabel.shape(torch.Size([BS, 1, n_dfboxes]))
        
        # encode targets/default boxes TODO: move this into encoder?
        gloc, glabel = [], []
        for i in range(config["BS"]):
            true_bboxes = target_dict[i]["boxes"].to(config["device"],non_blocking=True) #torch.Size([x, 4])
            true_labels = target_dict[i]["labels"].to(config["device"],non_blocking=True) #torch.Size([x])

            encoded_gloc, encoded_glabel = encoder.encode(true_bboxes, true_labels) #torch.Size([n_dfboxes, 4]) and torch.Size([n_dfboxes])

            gloc.append(encoded_gloc.to(config["device"],non_blocking=True))
            glabel.append(encoded_glabel.to(config["device"],non_blocking=True)) 

# ###
#         f,ax = plt.subplots()
#         img = images.cpu().detach().numpy()
#         ii = ax.imshow(img[i][0],cmap='binary_r')
#         # ax.hlines([-np.pi,np.pi],-3,3,color='red',ls='dashed')
#         # ax.hlines([-1.9396086193266369,1.940238044375465],-3,3,color='orange',ls='dashed')
#         for bbx in true_bboxes:
#             bbx = bbx.cpu().detach().numpy()
#             bb = matplotlib.patches.Rectangle((bbx[0],bbx[1]),bbx[2]-bbx[0],bbx[3]-bbx[1],lw=1,ec='limegreen',fc='none')
#             ax.add_patch(bb)
#         cbar = f.colorbar(ii,ax=ax)
#         cbar.ax.get_yaxis().labelpad = 10
#         cbar.set_label('cell significance', rotation=90)
#         ax.set(xlabel='eta',ylabel='phi')
#         f.savefig('central-image-example.png')
#         plt.close()
#         quit()
# ###

        gloc = torch.stack(gloc).to(config["device"],non_blocking = True) #torch.Size([BS, n_dfboxes, 4])
        glabel = torch.stack(glabel).to(config["device"],non_blocking = True)#torch.Size([BS, n_dfboxes])
        gloc = gloc.permute(0, 2, 1) #torch.Size([BS, 4, 3024])

        train_loss = loss(plocs, plabel, gloc, glabel)
        running_loss.append(train_loss.item())

        # back prop
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    print(f"\tEpoch {epoch} / {config['n_epochs']}: train loss {mean(running_loss):.4f}, train time {time.perf_counter() - beginning:.2f}s, LR: {optimizer.param_groups[0]['lr']:.4f}")


    # validation step
    model.eval()
    with torch.inference_mode():
        running_val_loss = list()
        for step, (val_images, val_dict) in enumerate(val_loader):
            # send data to gpu (annoying)
            val_images = val_images.to(config["device"],non_blocking=True) 
        
            # forward pass
            plocs, plabel, ptmap = model(val_images) #plocs.shape(torch.Size([BS, 4, n_dfboxes])) and plabel.shape(torch.Size([BS, 1, n_dfboxes]))

            # encode val_targets/default boxes
            gloc, glabel = [], []
            for i in range(config["BS"]):
                true_bboxes = val_dict[i]["boxes"].to(config["device"],non_blocking=True) #torch.Size([x, 4])
                true_labels = val_dict[i]["labels"].to(config["device"],non_blocking=True) #torch.Size([x])

                encoded_gloc, encoded_glabel = encoder.encode(true_bboxes, true_labels) #torch.Size([n_dfboxes, 4]) and torch.Size([n_dfboxes])
                gloc.append(encoded_gloc.to(config["device"],non_blocking=True))
                glabel.append(encoded_glabel.to(config["device"],non_blocking=True)) 

            gloc = torch.stack(gloc).to(config["device"],non_blocking = True) #torch.Size([BS, n_dfboxes, 4])
            glabel = torch.stack(glabel).to(config["device"],non_blocking = True)#torch.Size([BS, n_dfboxes])
            gloc = gloc.permute(0, 2, 1) #torch.Size([BS, 4, n_dfboxes])

            val_loss = loss(plocs, plabel, gloc, glabel) 
            running_val_loss.append(val_loss.item())
        
    print(f"\tEpoch {epoch} / {config['n_epochs']}: valid loss {mean(running_val_loss):.4f}, valid time {time.perf_counter() - beginning:.2f}s")
    
    # update LR scheduler
    scheduler.step()


# save trained model
model_name = "jetSSD_{}_{}e".format(model.backbone_name,config["n_epochs"])
print(f'Saving model now...\t{model_name}')
torch.save(model.state_dict(), "saved_models/{}.pth".format(model_name))



########################################################################################################################



print("Finished training. Now let's look at one event and infer")
print(f"Using default {dboxes.dboxes.shape} boxes and Encoder")
model.eval()
with torch.no_grad():
    for i, (images,targets) in enumerate(test_loader):
        images = images.to(config["device"]).float()
        model = model.to(config["device"])

        locs,conf,ptmap = model(images)
        print("decoding:")
        # define NMS scriteria, confidence threshold
        output = encoder.decode_batch(locs, conf, ptmap,
                                        iou_thresh=0.3, #NMS
                                        confidence=0.6, #conf threshold
                                        max_num=150) 

        boxes, labels, scores, pts = zip(*output)

        # examine event by event
        for j, ((boxes, labels, scores, pts), img) in enumerate(zip(output, images)):
            print(j)
            print(min(scores.cpu()))
            
            # make truth boxes cover extent
            extent_j = targets[j]['extent']
            tru_boxes_ext = targets[j]['boxes'].cpu()
            tru_boxes_ext[:,(0,2)] = (tru_boxes_ext[:,(0,2)]*((extent_j[1]-extent_j[0])/img.shape[2]))+extent_j[0]
            tru_boxes_ext[:,(1,3)] = (tru_boxes_ext[:,(1,3)]*((extent_j[3]-extent_j[2])/img.shape[1]))+extent_j[2]
            
            tru_pt = targets[j]['jet_pt']
            event_number = targets[j]['event_no']
            pred_scores = scores.cpu()
            pred_pts = pts.cpu()
            pred_labels = labels.cpu()

            # make pred boxes cover extent
            det_boxes_ext = boxes.clone().cpu()
            det_boxes_ext[:,(0,2)] = (det_boxes_ext[:,(0,2)]*((extent_j[1]-extent_j[0])))+extent_j[0]
            det_boxes_ext[:,(1,3)] = (det_boxes_ext[:,(1,3)]*((extent_j[3]-extent_j[2])))+extent_j[2]
            print("\tAnti-kt jet pt: ", tru_pt)
            print("\tPred box jet pt: ", pred_pts)


            # plotting
            MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
            MIN_CELLS_ETA,MAX_CELLS_ETA = -2.5, 2.5

            f,ax = plt.subplots(1,2)
            ax[0].imshow(img[0].cpu().numpy(),cmap='binary_r',extent=extent_j,origin='lower')
            ax[1].imshow(img[1].cpu().numpy(),cmap='binary_r',extent=extent_j,origin='lower')
            
            ax[0].axhline(y=MIN_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
            ax[0].axhline(y=MAX_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
            ax[1].axhline(y=MIN_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
            ax[1].axhline(y=MAX_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)

            for bbx in tru_boxes_ext:
                x,y=float(bbx[0]),float(bbx[1])
                w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
                ax[0].add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='limegreen',fc='none'))
                ax[1].add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='limegreen',fc='none'))

            for bbx,scr,pt in zip(det_boxes_ext,pred_scores,pred_pts):
                x,y=float(bbx[0]),float(bbx[1])
                w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
                ax[0].add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1.25,ec='red',fc='none'))
                ax[1].add_patch(matplotlib.patches.Rectangle((x,y),w,h,lw=1.25,ec='red',fc='none'))
                ax[1].text(x+w,y+h, f"{scr.item():.2f}",color='red',fontsize=6)
                ax[1].text(x,y+h/20, f"{pt.item():.0f}",color='red',fontsize=6)

            ax[0].set(xlabel='$\eta$',ylabel='$\phi$')
            ax[1].set(xlabel='$\eta$',ylabel='$\phi$')
            plt.tight_layout()
            model_name = "jetSSD_{}_{}e".format(model.backbone_name,config["n_epochs"])
            f.savefig('{}-{}-{}.png'.format(model_name,j,event_number))
            plt.close()
            
            quit()

