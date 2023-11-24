import numpy as np 
import pandas as pd
import os
import h5py
import sys
import itertools
import fastjet
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.pyplot as plt
import matplotlib

sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')
from utils.utils import wrap_check_NMS, wrap_check_truth, remove_nan, transform_angle
from utils.utils import make_image_using_cells
from utils.metrics import  event_cluster_estimates

import mplhep as hep
hep.style.use(hep.style.ATLAS)



def make_single_event_plot(
    folder_containing_struc_array,
    save_folder,
    idx=0,
    pdf=False,
):
    save_loc = save_folder + "/single_event/{}/".format(idx)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    image_format = "pdf" if pdf else "png"

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
    tees2 = tees

    pees[:,(0,2)] = (pees[:,(0,2)]*(extent[1]-extent[0]))+extent[0]
    pees[:,(1,3)] = (pees[:,(1,3)]*(extent[3]-extent[2]))+extent[2]
    
    cells_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/cells/user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f)
    with h5py.File(cells_file,"r") as f:
        h5group = f["caloCells"]
        event = h5group["1d"][event_no]
        cells = h5group["2d"][event_no]

    clusters_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/clusters/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f)
    with h5py.File(clusters_file,"r") as f:
        cl_data = f["caloCells"]
        event_data = cl_data["1d"][event_no]
        cluster_data = cl_data["2d"][event_no]
        cluster_cell_data = cl_data["3d"][event_no]
        # mm1 = cluster_data['cl_E_em']+cluster_data['cl_E_had']
        # mm2 = cluster_data['cl_E_EMB1']+cluster_data['cl_E_EMB2']+cluster_data['cl_E_EMB3']+cluster_data['cl_E_EME1']+cluster_data['cl_E_EME2']+cluster_data['cl_E_EME3']+cluster_data['cl_E_FCAL0']+cluster_data['cl_E_FCAL1']+cluster_data['cl_E_FCAL2']+cluster_data['cl_E_HEC0']+cluster_data['cl_E_HEC1']+cluster_data['cl_E_HEC2']+cluster_data['cl_E_HEC3']+cluster_data['cl_E_PreSamplerB']+cluster_data['cl_E_PreSamplerE']+cluster_data['cl_E_TileBar0']+cluster_data['cl_E_TileBar1']+cluster_data['cl_E_TileBar2']+cluster_data['cl_E_TileExt0']+cluster_data['cl_E_TileExt1']+cluster_data['cl_E_TileExt2']+cluster_data['cl_E_TileGap1']+cluster_data['cl_E_TileGap2']+cluster_data['cl_E_TileGap3']

    jets_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/jets/user.cantel.34126190._0000{}.jetD3PD_mc16_JZ4W.r10788.h5".format(h5f)
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

    ###############################################################################################################

    # cells_EMBar = cells[np.isin(cells['cell_DetCells'],[65,81,97,113])]
    # cells_EMEC = cells[np.isin(cells['cell_DetCells'],[257,273,289,305])]
    # cells_EMIW = cells[np.isin(cells['cell_DetCells'],[145,161])]
    # cells_EMFCAL = cells[np.isin(cells['cell_DetCells'],[2052])]

    # cells_HEC = cells[np.isin(cells['cell_DetCells'],[2,514,1026,1538])]
    # cells_HFCAL = cells[np.isin(cells['cell_DetCells'],[4100,6148])]
    # cells_TileBar = cells[np.isin(cells['cell_DetCells'],[65544, 73736,81928,])]
    # cells_TileEC = cells[np.isin(cells['cell_DetCells'],[131080,139272,147464])]
    # cells_TileGap = cells[np.isin(cells['cell_DetCells'],[811016,278536,270344])]

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(cells_EMBar['cell_xCells'],cells_EMBar['cell_zCells'],cells_EMBar['cell_yCells'],s=1.25,marker='.',color='royalblue',alpha=.175,label='EM Bar')
    # ax.scatter(cells_EMEC['cell_xCells'],cells_EMEC['cell_zCells'],cells_EMEC['cell_yCells'],s=1.25,marker='.',color='turquoise',alpha=.175,label='EM EC')
    # ax.scatter(cells_EMIW['cell_xCells'],cells_EMIW['cell_zCells'],cells_EMIW['cell_yCells'],s=1.25,marker='.',color='springgreen',alpha=.65,label='EM IW')
    # ax.scatter(cells_EMFCAL['cell_xCells'],cells_EMFCAL['cell_zCells'],cells_EMFCAL['cell_yCells'],s=1.25,marker='.',color='forestgreen',alpha=.5,label='EM FCAL')
    
    # ax.scatter(cells_HEC['cell_xCells'],cells_HEC['cell_zCells'],cells_HEC['cell_yCells'],s=2.0,marker='o',color='tab:orange',alpha=.5,label='HAD EC')
    # ax.scatter(cells_HFCAL['cell_xCells'],cells_HFCAL['cell_zCells'],cells_HFCAL['cell_yCells'],s=2.0,marker='o',color='yellow',alpha=.5,label='HAD FCAL')
    # ax.scatter(cells_TileBar['cell_xCells'],cells_TileBar['cell_zCells'],cells_TileBar['cell_yCells'],s=2.0,marker='o',color='tomato',alpha=.5,label='HAD Tile Bar')
    # ax.scatter(cells_TileEC['cell_xCells'],cells_TileEC['cell_zCells'],cells_TileEC['cell_yCells'],s=2.0,marker='o',color='red',alpha=.5,label='HAD Tile Bar')
    # ax.scatter(cells_TileGap['cell_xCells'],cells_TileGap['cell_zCells'],cells_TileGap['cell_yCells'],s=2.0,marker='o',color='peru',alpha=.5,label='HAD Tile Gap')

    # ax.set(xlabel='X',ylabel='Z',zlabel='Y')
    # lgnd = ax.legend(loc='lower left',bbox_to_anchor=(1.025,0.35))
    # for i in range(len(lgnd.legend_handles)):
    #     if i<4:
    #         lgnd.legend_handles[i]._sizes = [450]
    #     else:
    #         lgnd.legend_handles[i]._sizes = [150]

    # hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,com="",pad=-0.02)
    # fig.savefig(save_loc+'/calo-cells-{}.{}'.format(idx,image_format),dpi=400,format=image_format,bbox_extra_artists=(lgnd,), bbox_inches='tight')

    ###############################################################################################################

    # data_x = cells['cell_eta']
    # data_y = cells['cell_phi']

    # fig = plt.figure(figsize=(8, 8))
    # a1 = fig.add_subplot(111, projection='3d')
    # bin_width_x,bin_width_y=0.1,0.1
    # bins_x = np.arange(min(data_x), max(data_x) + bin_width_x, bin_width_x)
    # bins_y = np.arange(min(data_y), max(data_y) + bin_width_y, bin_width_y)

    # hist, x_edges, y_edges = np.histogram2d(data_x, data_y, bins=[bins_x, bins_y])

    # x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    # y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    # x, y = np.meshgrid(x_centers, y_centers)
    # x = x.flatten()
    # y = y.flatten()
    # z = hist.flatten()

    # cmap = matplotlib.colormaps.get_cmap('plasma')
    # norm = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
    # colors = cmap(norm(z))
    # sc = matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm)
    # sc.set_array([])
    # cbar = plt.colorbar(sc, ax=a1, pad=0.05, shrink=0.7)
    # # cbar.set_label('Frequency')
    # cbar.set_label(f'Number Calo. Cells')

    # a1.bar3d(x, y, np.zeros_like(z), bin_width_x, bin_width_y, z, shade=True, color=colors)
    # a1.set(ylabel=r'$\eta$',xlabel=r'$\phi$')
    # # hep.atlas.label(ax=a1,label='Work in Progress',data=True,lumi=None,pad=-0.02)
    # fig.savefig(save_loc+'/calo-cells-2d-{}.{}'.format(idx,image_format),dpi=400,format=image_format,bbox_inches="tight")



    ###############################################################################################################
    #make the image as it would be input to CNN
    H_layer0 = make_image_using_cells(cells,channel=0)

    ###############################################################################################################
    f,ax = plt.subplots(1,1,figsize=(10, 7))
    ax.imshow(H_layer0,cmap='binary_r',extent=extent,origin='lower')
    for bbx in tees2:
        x,y=float(bbx[0]),float(bbx[1])
        w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
        bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='limegreen',fc='none')
        ax.add_patch(bb)

    for pred_box,pred_score in zip(pees,scores):
        x,y=float(pred_box[0]),float(pred_box[1])
        w,h=float(pred_box[2])-float(pred_box[0]),float(pred_box[3])-float(pred_box[1])  
        bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1.25,ec='red',fc='none')
        # ax.add_patch(bb)
    
    ax.set(xlabel='$\eta$',ylabel='$\phi$')
    ax.axhline(y=min(cells['cell_phi']), color='pink', linestyle='--',lw=0.5)
    ax.axhline(y=max(cells['cell_phi']), color='pink', linestyle='--',lw=0.5)
    hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=0)
    f.savefig(save_loc+'/cells-img-boxes-{}.{}'.format(idx,image_format),dpi=400,format=image_format,bbox_inches="tight")

    ###############################################################################################################
    
    H_layer0_nowrap = make_image_using_cells(cells,channel=0,padding=False)
    f,ax = plt.subplots(1,1)
    custom_extent = [min(cells['cell_eta']),max(cells['cell_eta']),min(cells['cell_phi']),max(cells['cell_phi'])]
    ax.imshow(H_layer0_nowrap,cmap='binary_r',extent=custom_extent,origin='lower')
    for bbx in tees:
        x,y=float(bbx[0]),float(bbx[1])
        w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
        bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='limegreen',fc='none')
        # ax.add_patch(bb)
    
    for pred_box,pred_score in zip(pees,scores):
        x,y=float(pred_box[0]),float(pred_box[1])
        w,h=float(pred_box[2])-float(pred_box[0]),float(pred_box[3])-float(pred_box[1])  
        bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1.25,ec='red',fc='none')
        # ax.add_patch(bb)
    
    ax.set(xlabel='$\eta$',ylabel='$\phi$')
    ax.axhline(y=min(cells['cell_phi']), color='pink', linestyle='--',lw=0.5)
    ax.axhline(y=max(cells['cell_phi']), color='pink', linestyle='--',lw=0.5)
    hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=0)
    f.savefig(save_loc+'/cells-img-boxes-nowrap-{}.{}'.format(idx,image_format),dpi=400,format=image_format, bbox_inches="tight")


    ###############################################################################################################
    f,a = plt.subplots(1,1,figsize=(7,9))
    #esd clusters
    for c in range(len(new_cluster_data)):
        cc = new_cluster_data[c]
        if cc['cl_E_em'] + cc['cl_E_had'] > 5000:
            #a[0].plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.5,color='plum',ms=6)
            a.plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.65,color='plum',ms=3,markeredgecolor='k')
        else:
            # a[0].plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.45,color='thistle',ms=4)
            a.plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.55,color='thistle',ms=3)

    #truth boxes
    for tbx in tees2:
        x,y=float(tbx[0]),float(tbx[1])
        w,h=float(tbx[2])-float(tbx[0]),float(tbx[3])-float(tbx[1])  
        bbo = matplotlib.patches.Rectangle((x,y),w,h,lw=2,ec='green',fc='none')
        a.add_patch(bbo)

    #box predictions
    for pbx in pees:
        x,y=float(pbx[0]),float(pbx[1])
        w,h=float(pbx[2])-float(pbx[0]),float(pbx[3])-float(pbx[1])  
        bbo = matplotlib.patches.Rectangle((x,y),w,h,lw=2,ec='red',fc='none')
        # a.add_patch(bbo)

    a.axhline(y=min(cells['cell_phi']), color='r', linestyle='--')
    a.axhline(y=max(cells['cell_phi']), color='r', linestyle='--')
    a.grid()
    a.set(xlabel='$\eta$',ylabel='$\phi$',xlim=(min(cells['cell_eta']), max(cells['cell_eta'])),ylim=(min(cells['cell_phi'])-3, max(cells['cell_phi'])+3))

    legend_elements = [
                    # matplotlib.patches.Patch(facecolor='w', edgecolor='red',label='Model pred.'),
                    matplotlib.patches.Patch(facecolor='w', edgecolor='green',label='Truth'),
                    matplotlib.lines.Line2D([],[], marker='h', color='plum', label='TopoClusters >5GeV',linestyle='None',markersize=10),
                    matplotlib.lines.Line2D([],[], marker='h', color='plum', label='TopoClusters <5GeV',linestyle='None',markersize=10,markeredgecolor='k'),]
    a.legend(handles=legend_elements, loc='lower left',bbox_to_anchor=(0.63, 0.74),fontsize="x-small")
    hep.atlas.label(ax=a,label='Work in Progress',data=False,lumi=None,loc=0)
    f.savefig(save_loc+'/boxes-clusters-{}.{}'.format(idx,image_format),dpi=400,format=image_format, bbox_inches="tight")

    ###############################################################################################################

    print('Starting fastjet procedure now...')

    m = 0.0 #topoclusters have 0 mass
    ESD_inputs = []
    for i in range(len(new_cluster_data)):
        cl_px = float(new_cluster_data[i]['cl_pt'] * np.cos(new_cluster_data[i]['cl_phi']))
        cl_py = float(new_cluster_data[i]['cl_pt'] * np.sin(new_cluster_data[i]['cl_phi']))
        cl_pz = float(new_cluster_data[i]['cl_pt'] * np.sinh(new_cluster_data[i]['cl_eta']))
        ESD_inputs.append(fastjet.PseudoJet(cl_px,cl_py,cl_pz,m))


    #From my predictions
    list_p_cl_es, list_t_cl_es = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='energy')
    list_p_cl_etas, list_t_cl_etas = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='eta')
    list_p_cl_phis, list_t_cl_phis = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='phi')
    truth_box_inputs = []
    for j in range(len(list_t_cl_es)):
        truth_box_eta = list_t_cl_etas[j]
        truth_box_phi = list_t_cl_phis[j]
        truth_box_theta = 2*np.arctan(np.exp(-truth_box_eta))
        truth_box_e = list_t_cl_es[j]
        truth_box_inputs.append(fastjet.PseudoJet(truth_box_e*np.sin(truth_box_theta)*np.cos(truth_box_phi),
                                                truth_box_e*np.sin(truth_box_theta)*np.sin(truth_box_phi),
                                                truth_box_e*np.cos(truth_box_theta),
                                                m))

    pred_box_inputs = []                                            
    for j in range(len(list_p_cl_es)):
        pred_box_eta = list_p_cl_etas[j]
        pred_box_phi = list_p_cl_phis[j]
        pred_box_theta = 2*np.arctan(np.exp(-pred_box_eta))
        pred_box_e = list_p_cl_es[j]
        pred_box_inputs.append(fastjet.PseudoJet(pred_box_e*np.sin(pred_box_theta)*np.cos(pred_box_phi),
                                                pred_box_e*np.sin(pred_box_theta)*np.sin(pred_box_phi),
                                                pred_box_e*np.cos(pred_box_theta),
                                                m))


    #######################################################################################
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

    #ESD Clusters:
    ESD_cluster_jets = fastjet.ClusterSequence(ESD_inputs, jetdef)
    ESD_cluster_inc_jets = ESD_cluster_jets.inclusive_jets()

    #Truth box clusters
    truth_box_jets = fastjet.ClusterSequence(truth_box_inputs,jetdef)
    truth_box_inc_jets = truth_box_jets.inclusive_jets()

    #Pred box clusters
    pred_box_jets = fastjet.ClusterSequence(pred_box_inputs,jetdef)
    pred_box_inc_jets = pred_box_jets.inclusive_jets()



    f,a = plt.subplots(1,2,figsize=(14.5,6))
    #esd jets
    for oj in range(len(new_jet_data)):
        offjet = new_jet_data[oj]
        a[1].plot(offjet['AntiKt4EMTopoJets_JetConstitScaleMomentum_eta'],offjet['AntiKt4EMTopoJets_JetConstitScaleMomentum_phi'], markersize=25, marker='*',color='goldenrod',alpha=1.0,mew=1,markeredgecolor='k')

    #fastjet jets
    # for j in range(len(ESD_cluster_inc_jets)):
    #     jj = ESD_cluster_inc_jets[j]
    #     if jj.pt() > 5000:
    #         a[1].plot(jj.eta(),transform_angle(jj.phi()), markersize=12, marker='o',alpha=.7,color='dodgerblue',markeredgecolor='k')
    #     else:
    #         a[1].plot(jj.eta(),transform_angle(jj.phi()), markersize=5, marker='o',alpha=.6,color='dodgerblue')

    #truth box jets
    for tbj in range(len(truth_box_inc_jets)):
        jj = truth_box_inc_jets[tbj]
        a[0].plot(jj.eta(),transform_angle(jj.phi()), ms=15, marker='$T$',color='limegreen',markeredgecolor='k')

    #truth box jets
    for pbj in range(len(pred_box_inc_jets)):
        jj = pred_box_inc_jets[pbj]
        a[0].plot(jj.eta(),transform_angle(jj.phi()), ms=15, marker='$P$',color='mediumvioletred')

    #esd clusters
    for c in range(len(new_cluster_data)):
        cc = new_cluster_data[c]
        if cc['cl_E_em'] + cc['cl_E_had'] > 5000:
            a[1].plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.65,color='plum',ms=3,markeredgecolor='k')
        else:
            a[1].plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.55,color='thistle',ms=3)

    #truth boxes
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

    a[0].axhline(y=min(cells['cell_phi']), color='r', linestyle='--')
    a[0].axhline(y=max(cells['cell_phi']), color='r', linestyle='--')
    # a[0].grid()
    a[0].set(xlabel='eta',ylabel='phi',xlim=(min(cells['cell_eta']), max(cells['cell_eta'])),ylim=(min(cells['cell_phi'])-3, max(cells['cell_phi'])+3))
    a[1].axhline(y=min(cells['cell_phi']), color='r', linestyle='--')
    a[1].axhline(y=max(cells['cell_phi']), color='r', linestyle='--')
    # a[1].grid()
    a[1].set(xlabel='$\eta$',ylabel='$\phi$',xlim=(min(cells['cell_eta']), max(cells['cell_eta'])),ylim=(min(cells['cell_phi'])-3, max(cells['cell_phi'])+3))

    legend_elements = [
                    # matplotlib.patches.Patch(facecolor='w', edgecolor='red',label='Model pred.'),
                    # matplotlib.patches.Patch(facecolor='w', edgecolor='green',label='Truth'),
                    # matplotlib.lines.Line2D([],[], marker='o', color='dodgerblue', label='TopoCl Jets >5GeV',linestyle='None',markersize=10),
                    # matplotlib.lines.Line2D([],[], marker='o', color='dodgerblue', label='TopoCl Jets <5GeV',linestyle='None',markersize=10,markeredgecolor='k'),
                    matplotlib.lines.Line2D([],[], marker='h', color='plum', label='TopoClusters >5GeV',linestyle='None',markersize=10),
                    matplotlib.lines.Line2D([],[], marker='h', color='plum', label='TopoClusters <5GeV',linestyle='None',markersize=10,markeredgecolor='k'),
                    matplotlib.lines.Line2D([],[], marker='*', color='goldenrod', label='ESD Jets',linestyle='None',markersize=10),
                    matplotlib.lines.Line2D([],[], marker='$T$', color='limegreen', label='Truth box Jets',linestyle='None',markersize=10),
                    matplotlib.lines.Line2D([],[], marker='$P$', color='mediumvioletred', label='Pred box Jets',linestyle='None',markersize=10),]
    a[1].legend(handles=legend_elements, loc='best',frameon=False,bbox_to_anchor=(1, 0.5))
    # plt.subplots_adjust(right=0.7)
    f.tight_layout()
    f.savefig(save_loc+'/boxes-clusters-jets-{}.{}'.format(idx,image_format),dpi=400,format=image_format, bbox_inches="tight")





    f2,a = plt.subplots(1,1,figsize=(7.5,6))
    #esd jets
    for oj in range(len(new_jet_data)):
        offjet = new_jet_data[oj]
        a.plot(offjet['AntiKt4EMTopoJets_JetConstitScaleMomentum_eta'],offjet['AntiKt4EMTopoJets_JetConstitScaleMomentum_phi'], markersize=25, marker='*',color='goldenrod',alpha=0.8,mew=1,markeredgecolor='k')

    #esd clusters
    for c in range(len(new_cluster_data)):
        cc = new_cluster_data[c]
        if cc['cl_E_em'] + cc['cl_E_had'] > 5000:
            a.plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.65,color='plum',ms=3,markeredgecolor='k')
        else:
            a.plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.55,color='thistle',ms=3)

    #truth box jets
    for tbj in range(len(truth_box_inc_jets)):
        jj = truth_box_inc_jets[tbj]
        a.plot(jj.eta(),transform_angle(jj.phi()), ms=15, marker='$T$',color='limegreen',markeredgecolor='k')

    #pred box jets
    for pbj in range(len(pred_box_inc_jets)):
        jj = pred_box_inc_jets[pbj]
        a.plot(jj.eta(),transform_angle(jj.phi()), ms=15, marker='$P$',color='mediumvioletred')


    a.axhline(y=min(cells['cell_phi']), color='r', linestyle='--')
    a.axhline(y=max(cells['cell_phi']), color='r', linestyle='--')
    # a.grid()
    a.set(xlabel='$\eta$',ylabel='$\phi$',xlim=(min(cells['cell_eta']), max(cells['cell_eta'])),ylim=(min(cells['cell_phi'])-3, max(cells['cell_phi'])+3))

    legend_elements = [ matplotlib.lines.Line2D([],[], marker='h', color='plum', label='TopoClusters >5GeV',linestyle='None',markersize=10),
                    matplotlib.lines.Line2D([],[], marker='h', color='plum', label='TopoClusters <5GeV',linestyle='None',markersize=10,markeredgecolor='k'),
                    matplotlib.lines.Line2D([],[], marker='*', color='goldenrod', label='ESD Jets',linestyle='None',markersize=10),
                    matplotlib.lines.Line2D([],[], marker='$T$', color='limegreen', label='Truth box FJets',linestyle='None',markersize=10),
                    matplotlib.lines.Line2D([],[], marker='$P$', color='mediumvioletred', label='Pred box FJets',linestyle='None',markersize=10),]
    a.legend(handles=legend_elements, loc='best',frameon=False,bbox_to_anchor=(1, 0.5))
    f2.savefig(save_loc+'/box-jets-jets-{}.{}'.format(idx,image_format),dpi=400,format=image_format, bbox_inches="tight")

    return



folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD1_50k5_mu_15e/20231102-13/"
save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_plots/SSD1_50k5_mu_15e/"
if __name__=="__main__":
    print('Making single event')
    make_single_event_plot(folder_to_look_in,save_at,idx=3,pdf=False)
    make_single_event_plot(folder_to_look_in,save_at,idx=3,pdf=True)
    print('Completed single event plots\n')

