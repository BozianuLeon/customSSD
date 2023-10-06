import numpy as np 
import pandas as pd
import os
import h5py
import sys
import itertools
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')
from utils.utils import wrap_check_NMS, wrap_check_truth, remove_nan, get_cells_from_boxes
from utils.utils import make_image_using_cells





def make_single_event_plot(
    folder_containing_struc_array,
    save_folder,
    idx=0
):
    save_loc = save_folder + "/single_event/{}/".format(idx)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    with open(folder_containing_struc_array+"/struc_array.npy", 'rb') as f:
        a = np.load(f)
        h5f = a[idx]['h5file']
        h5f = h5f.decode('utf-8')
        event_no = a[idx]['event_no']

        preds = a[idx]['p_boxes']
        scores = a[idx]['p_scores']
        truths = a[idx]['t_boxes']
        extent = a[idx]['extent']

    pees = preds[np.where(preds[:,-1] > 0)]
    confs = scores[np.where(scores > 0)]
    tees = truths[np.where(truths[:,-1] > 0)]
    tees[:,(0,2)] = (tees[:,(0,2)]*(extent[1]-extent[0]))+extent[0]
    tees[:,(1,3)] = (tees[:,(1,3)]*(extent[3]-extent[2]))+extent[2]

    pees[:,(0,2)] = (pees[:,(0,2)]*(extent[1]-extent[0]))+extent[0]
    pees[:,(1,3)] = (pees[:,(1,3)]*(extent[3]-extent[2]))+extent[2]
    
    cells_file = "/home/users/b/bozianu/work/data/pileup50k/cells/user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f)
    with h5py.File(cells_file,"r") as f:
        h5group = f["caloCells"]
        event = h5group["1d"][event_no]
        cells = h5group["2d"][event_no]

    clusters_file = "/home/users/b/bozianu/work/data/pileup50k/clusters/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f)
    with h5py.File(clusters_file,"r") as f:
        cl_data = f["caloCells"]
        event_data = cl_data["1d"][event_no]
        cluster_data = cl_data["2d"][event_no]
        cluster_cell_data = cl_data["3d"][event_no]

    jets_file = "/home/users/b/bozianu/work/data/pileup50k/jets/user.cantel.34126190._0000{}.jetD3PD_mc16_JZ4W.r10788.h5".format(h5f)
    with h5py.File(jets_file,"r") as f:
        j_data = f["caloCells"]
        event_data = j_data["1d"][event_no]
        jet_data = j_data["2d"][event_no]
        
    #wrap check
    wc_pred_boxes = wrap_check_NMS(pees,scores,min(cells['cell_phi']),max(cells['cell_phi']),threshold=0.2)
    wc_truth_boxes = wrap_check_truth(tees,min(cells['cell_phi']),max(cells['cell_phi']))
    #maybe dont need
    pees = wc_pred_boxes
    tees = wc_truth_boxes

    #From the ESD file
    new_cluster_data = remove_nan(cluster_data)
    new_jet_data = remove_nan(jet_data)

    #make the image as it would be input to CNN
    H_layer0 = make_image_using_cells(cells,channel=0)


    ###############################################################################################################
    f,ax = plt.subplots(1,1)
    ax.imshow(H_layer0,cmap='binary_r',extent=extent,origin='lower')
    for bbx in tees:
        x,y=float(bbx[0]),float(bbx[1])
        w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
        bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='limegreen',fc='none')
        ax.add_patch(bb)

    for pred_box,pred_score in zip(pees,scores):
        x,y=float(pred_box[0]),float(pred_box[1])
        w,h=float(pred_box[2])-float(pred_box[0]),float(pred_box[3])-float(pred_box[1])  
        bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1.25,ec='red',fc='none')
        ax.add_patch(bb)
    
    ax.set(xlabel='$\eta$',ylabel='$\phi$')
    ax.axhline(y=min(cells['cell_phi']), color='pink', linestyle='--',lw=0.5)
    ax.axhline(y=max(cells['cell_phi']), color='pink', linestyle='--',lw=0.5)
    f.savefig(save_loc+'/cells-img-boxes-{}.png'.format(idx))

    ###############################################################################################################
    H_layer0_nowrap = make_image_using_cells(cells,channel=0,padding=False)
    f,ax = plt.subplots(1,1)

    custom_extent = [min(cells['cell_eta']),max(cells['cell_eta']),min(cells['cell_phi']),max(cells['cell_phi'])]
    ax.imshow(H_layer0_nowrap,cmap='binary_r',extent=custom_extent,origin='lower')
    for bbx in tees:
        x,y=float(bbx[0]),float(bbx[1])
        w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
        bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='limegreen',fc='none')
        ax.add_patch(bb)

    for pred_box,pred_score in zip(pees,scores):
        x,y=float(pred_box[0]),float(pred_box[1])
        w,h=float(pred_box[2])-float(pred_box[0]),float(pred_box[3])-float(pred_box[1])  
        bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1.25,ec='red',fc='none')
        ax.add_patch(bb)
    
    ax.set(xlabel='$\eta$',ylabel='$\phi$')
    ax.axhline(y=min(cells['cell_phi']), color='pink', linestyle='--',lw=0.5)
    ax.axhline(y=max(cells['cell_phi']), color='pink', linestyle='--',lw=0.5)
    f.savefig(save_loc+'/cells-img-boxes-nowrap-{}.png'.format(idx))


    ###############################################################################################################
    f,a = plt.subplots(1,1,figsize=(8,6))
    #esd clusters
    for c in range(len(new_cluster_data)):
        cc = new_cluster_data[c]
        if cc['cl_E_em'] + cc['cl_E_had'] > 2000:
            #a[0].plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.5,color='plum',ms=6)
            a.plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.65,color='plum',ms=3,markeredgecolor='k')
        else:
            # a[0].plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.45,color='thistle',ms=4)
            a.plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.55,color='thistle',ms=3)

    #truth boxes
    for tbx in tees:
        x,y=float(tbx[0]),float(tbx[1])
        w,h=float(tbx[2])-float(tbx[0]),float(tbx[3])-float(tbx[1])  
        bbo = matplotlib.patches.Rectangle((x,y),w,h,lw=2,ec='green',fc='none')
        a.add_patch(bbo)

    #box predictions
    for pbx in pees:
        x,y=float(pbx[0]),float(pbx[1])
        w,h=float(pbx[2])-float(pbx[0]),float(pbx[3])-float(pbx[1])  
        bbo = matplotlib.patches.Rectangle((x,y),w,h,lw=2,ec='red',fc='none')
        a.add_patch(bbo)

    a.axhline(y=min(cells['cell_phi']), color='r', linestyle='--')
    a.axhline(y=max(cells['cell_phi']), color='r', linestyle='--')
    a.grid()
    a.set(xlabel='eta',ylabel='phi',xlim=(min(cells['cell_eta']), max(cells['cell_eta'])),ylim=(min(cells['cell_phi'])-3, max(cells['cell_phi'])+3))

    legend_elements = [matplotlib.patches.Patch(facecolor='w', edgecolor='red',label='Model pred.'),
                    matplotlib.patches.Patch(facecolor='w', edgecolor='green',label='Truth'),
                    matplotlib.lines.Line2D([],[], marker='h', color='plum', label='TopoClusters >2GeV',linestyle='None',markersize=10),
                    matplotlib.lines.Line2D([],[], marker='h', color='plum', label='TopoClusters <2GeV',linestyle='None',markersize=10,markeredgecolor='k'),]
    a.legend(handles=legend_elements, loc='best',frameon=False,bbox_to_anchor=(1, 0.5))
    # plt.subplots_adjust(right=0.7)
    f.tight_layout()
    f.savefig(save_loc+'/boxes-clusters-{}.png'.format(idx))

    return



folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD_50k5_mu_20e/20231005-12/"
save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_plots/SSD_50k5_mu_20e/"
if __name__=="__main__":
    print('Making single event')
    make_single_event_plot(folder_to_look_in,save_at,idx=1)
    print('Completed single event plots\n')










# f,a = plt.subplots(1,2,figsize=(11.5,6))
# #esd jets
# for oj in range(len(new_jet_data)):
#     offjet = new_jet_data[oj]
#     a[1].plot(offjet['AntiKt4EMTopoJets_JetConstitScaleMomentum_eta'],offjet['AntiKt4EMTopoJets_JetConstitScaleMomentum_phi'], markersize=25, marker='*',color='goldenrod',alpha=1.0,mew=1,markeredgecolor='k')

# #fastjet jets
# for j in range(len(ESD_cluster_inc_jets)):
#     jj = ESD_cluster_inc_jets[j]
#     if jj.pt() > 5000:
#         a[1].plot(jj.eta(),transform_angle(jj.phi()), markersize=12, marker='o',alpha=.7,color='dodgerblue',markeredgecolor='k')
#     else:
#         a[1].plot(jj.eta(),transform_angle(jj.phi()), markersize=5, marker='o',alpha=.6,color='dodgerblue')
# print()
# #truth box jets
# for tbj in range(len(truth_box_inc_jets)):
#     jj = truth_box_inc_jets[tbj]
#     a[0].plot(jj.eta(),transform_angle(jj.phi()), ms=15, marker='$T$',color='limegreen',markeredgecolor='k')

# print()
# #truth box jets
# for pbj in range(len(pred_box_inc_jets)):
#     jj = pred_box_inc_jets[pbj]
#     a[0].plot(jj.eta(),transform_angle(jj.phi()), ms=15, marker='$P$',color='mediumvioletred')

# #esd clusters
# for c in range(len(new_cluster_data)):
#     cc = new_cluster_data[c]
#     if cc['cl_E_em'] + cc['cl_E_had'] > 5000:
#         #a[0].plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.5,color='plum',ms=6)
#         a[0].plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.65,color='plum',ms=3,markeredgecolor='k')
#     else:
#         # a[0].plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.45,color='thistle',ms=4)
#         a[0].plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.55,color='thistle',ms=3)

# #truth boxes
# for tbx in tees:
#     x,y=float(tbx[0]),float(tbx[1])
#     w,h=float(tbx[2])-float(tbx[0]),float(tbx[3])-float(tbx[1])  
#     bbo = matplotlib.patches.Rectangle((x,y),w,h,lw=2,ec='green',fc='none')
#     a[0].add_patch(bbo)

# #box predictions
# for pbx in pees:
#     x,y=float(pbx[0]),float(pbx[1])
#     w,h=float(pbx[2])-float(pbx[0]),float(pbx[3])-float(pbx[1])  
#     bbo = matplotlib.patches.Rectangle((x,y),w,h,lw=2,ec='red',fc='none')
#     a[0].add_patch(bbo)

# a[0].axhline(y=min(cells['cell_phi']), color='r', linestyle='--')
# a[0].axhline(y=max(cells['cell_phi']), color='r', linestyle='--')
# a[0].grid()
# a[0].set(xlabel='eta',ylabel='phi',xlim=(min(cells['cell_eta']), max(cells['cell_eta'])),ylim=(min(cells['cell_phi'])-3, max(cells['cell_phi'])+3))
# a[1].axhline(y=min(cells['cell_phi']), color='r', linestyle='--')
# a[1].axhline(y=max(cells['cell_phi']), color='r', linestyle='--')
# a[1].grid()
# a[1].set(xlabel='eta',ylabel='phi',xlim=(min(cells['cell_eta']), max(cells['cell_eta'])),ylim=(min(cells['cell_phi'])-3, max(cells['cell_phi'])+3))

# legend_elements = [matplotlib.patches.Patch(facecolor='w', edgecolor='red',label='Model pred.'),
#                    matplotlib.patches.Patch(facecolor='w', edgecolor='green',label='Truth'),
#                    matplotlib.lines.Line2D([],[], marker='o', color='dodgerblue', label='FJets >5GeV',linestyle='None',markersize=10),
#                    matplotlib.lines.Line2D([],[], marker='o', color='dodgerblue', label='FJets <5GeV',linestyle='None',markersize=10,markeredgecolor='k'),
#                    matplotlib.lines.Line2D([],[], marker='h', color='plum', label='TopoClusters >5GeV',linestyle='None',markersize=10),
#                    matplotlib.lines.Line2D([],[], marker='h', color='plum', label='TopoClusters <5GeV',linestyle='None',markersize=10,markeredgecolor='k'),
#                    matplotlib.lines.Line2D([],[], marker='*', color='goldenrod', label='ESD Jets',linestyle='None',markersize=10),
#                    matplotlib.lines.Line2D([],[], marker='$T$', color='limegreen', label='Truth box FJets',linestyle='None',markersize=10),
#                    matplotlib.lines.Line2D([],[], marker='$P$', color='mediumvioletred', label='Pred box FJets',linestyle='None',markersize=10),]
# a[1].legend(handles=legend_elements, loc='best',frameon=False,bbox_to_anchor=(1, 0.5))
# # plt.subplots_adjust(right=0.7)
# f.tight_layout()
# f.savefig('fast-box-jets{}-2.png'.format(idx))