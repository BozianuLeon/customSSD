import numpy as np
import torch
import sys
import os

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


from utils import iou_box_matching, dR_box_matching, wrap_check_truth3, wrap_check_NMS3



MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
# MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496
MIN_CELLS_ETA,MAX_CELLS_ETA = -2.5, 2.5
# EXTENT = [-4.82349586, 4.82349586, -6.21738815, 6.21801758] 
EXTENT = (-2.4999826, 2.4999774, -6.217388274177672, 6.2180176992265)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)

def clip_phi(phi_values):
    return phi_values - 2 * torch.pi * torch.floor((phi_values + torch.pi) / (2 * torch.pi))



results = {
           'evt_weight':         [],
           'jet_evt_weight':     [],

           'tarboxes_pt':         [],
           'tarboxes_eta':        [], 
           'tarboxes_phi':        [], 

           'truboxes_pt':            [],
           'truboxes_eta':           [],
           'truboxes_phi':           [],

           'pboxes_scores':     [], 
           'pboxes_pt':         [], 
           'pboxes_eta':        [], 
           'pboxes_phi':        [], 

           'tarboxes_matched_pt': [], 
           'pboxes_matched_pt': [], 
           'tarboxes_matched_eta':[], 
           'pboxes_matched_eta':[], 
           'tarboxes_matched_phi':[], 
           'pboxes_matched_phi':[], 
           'pboxes_matched_scr':[], 

           'tarboxes_unmatched_pt': [], 
           'pboxes_unmatched_pt': [], 
           'tarboxes_unmatched_eta':[], 
           'pboxes_unmatched_eta':[], 
           'tarboxes_unmatched_phi':[], 
           'pboxes_unmatched_phi':[], 
           'pboxes_unmatched_scr':[], 

           'tarboxes_dRmatched_pt':[],
           'pboxes_dRmatched_pt':[],
           'tarboxes_dRmatched_eta':[],
           'pboxes_dRmatched_eta':[],
           'tarboxes_dRmatched_phi':[],
           'pboxes_dRmatched_phi':[],
           'pboxes_dRmatched_scr':[],

           'tarboxes_dRunmatched_pt': [],
           'pboxes_dRunmatched_pt': [],
           'tarboxes_dRunmatched_eta':[],
           'pboxes_dRunmatched_eta':[],
           'tarboxes_dRunmatched_phi':[],
           'pboxes_dRunmatched_phi':[],
           'pboxes_dRunmatched_scr':[],

            # matching to truth jets:
           'truboxes_dRtruthmatched_pt':[],
           'truboxes_dRtruthmatched_eta':[],
           'truboxes_dRtruthmatched_phi':[],
           'pboxes_dRtruthmatched_pt':[],
           'pboxes_dRtruthmatched_eta':[],
           'pboxes_dRtruthmatched_phi':[],
           'pboxes_dRtruthmatched_scr':[],

           'truboxes_dRtruthunmatched_pt':[],
           'truboxes_dRtruthunmatched_eta':[],
           'truboxes_dRtruthunmatched_phi':[],
           'pboxes_dRtruthunmatched_pt':[],
           'pboxes_dRtruthunmatched_eta':[],
           'pboxes_dRtruthunmatched_phi':[],
           'pboxes_dRtruthunmatched_scr':[],

           # matching TARGETS to truth jets
           'truboxes_dRtruthtarmatched_pt':[],
           'truboxes_dRtruthtarmatched_eta':[],
           'truboxes_dRtruthtarmatched_phi':[],
           'tarboxes_dRtruthtarmatched_pt':[],
           'tarboxes_dRtruthtarmatched_eta':[],
           'tarboxes_dRtruthtarmatched_phi':[],

           'truboxes_dRtruthtarunmatched_pt':[],
           'truboxes_dRtruthtarunmatched_eta':[],
           'truboxes_dRtruthtarunmatched_phi':[],
           'tarboxes_dRtruthtarunmatched_pt':[],
           'tarboxes_dRtruthtarunmatched_eta':[],
           'tarboxes_dRtruthtarunmatched_phi':[],


           'n_targets':           [],
           'n_truth':           [],
           'n_preds':           [],
           'delta_n':           [],
           'delta_n_truth':           [],

           'n_matched_targets':   [],
           'n_unmatched_targets': [],
           'n_matched_preds':   [],
           'n_unmatched_preds': [],

           'n_dRmatched_targets':   [],
           'n_dRunmatched_targets': [],
           'n_dRmatched_preds':   [],
           'n_dRunmatched_preds': [],

           'n_dRtruthmatched_truths': [],
           'n_dRtruthunmatched_truths': [],
           'n_dRtruthmatched_preds': [],
           'n_dRtruthunmatched_preds': [],
}



def calculate_box_metrics(
    folder_containing_struc_array,
    save_folder,
):

    with open(folder_containing_struc_array + "/struc_array.npy", 'rb') as f:
        a = np.load(f)

    for i in range(len(a)):
        # i = 2387
        preds     = a[i]['p_boxes']
        scores    = a[i]['p_scores']
        p_momenta = a[i]['p_pt']
        extent_i  = a[i]['extent']
        akt_boxes = a[i]['tar_boxes']
        akt_pt    = a[i]['tar_pt']
        # trues = a[i]['t_boxes']
        # t_momenta = a[i]['t_pt']
        tru_boxes = a[i]['tru_boxes']
        tru_pt    = a[i]['tru_pt']
        evt_weight = a[i]['event_weight']
        print(i)

        #remove padding, boxes in xyxy coordinates
        pred_mask = ((preds[:, 2] - preds[:, 0]) >= 0.01) & (scores > 0.01) & (p_momenta > 0.01)
        pees = preds[pred_mask]
        scores = scores[pred_mask]
        p_momenta = p_momenta[pred_mask]
        targ_mask = ((akt_boxes[:, 2] - akt_boxes[:, 0]) >= 0.01) & (akt_pt > 1.0) # filter out 0.99 no central jets
        targs = akt_boxes[targ_mask]
        targ_momenta = akt_pt[targ_mask]
        tru_mask = ((tru_boxes[:, 2] - tru_boxes[:, 0]) >= 0.01) & (tru_pt > 1.0) # filter out 0.99 no central jets
        trus = tru_boxes[tru_mask]
        tru_momenta = tru_pt[tru_mask]

        if (len(pees)==0): print("No predicted jets"); continue 
        if (len(targs)==0): print("No target (AKT) jets"); continue 
        if (len(trus)==0): print("No truth jets"); continue 


        pees, scores, p_momenta = wrap_check_NMS3(pees,scores,p_momenta,iou_thresh=0.3)
        targs, targ_momenta = wrap_check_truth3(targs,targ_momenta,MIN_CELLS_PHI,MAX_CELLS_PHI)
        trus, tru_momenta = wrap_check_truth3(trus,tru_momenta,MIN_CELLS_PHI,MAX_CELLS_PHI)

        # centre of the truth/pred boxes
        tarboxes_ceta = (targs[:,2] + targs[:,0])/2
        tarboxes_cphi = (targs[:,3] + targs[:,1])/2
        truboxes_ceta = (trus[:,2] + trus[:,0])/2
        truboxes_cphi = (trus[:,3] + trus[:,1])/2
        pboxes_ceta = (pees[:,2] + pees[:,0])/2
        pboxes_cphi = (pees[:,3] + pees[:,1])/2
        pboxes_cphi = clip_phi(pboxes_cphi) # ensure phi values in [-pi,pi]


        # IoU matching
        tar_box_match_idx, p_box_match_idx = iou_box_matching(targs, pees, iou_thresh=0.5)
        tar_box_match_pt  = targ_momenta[tar_box_match_idx]
        tar_box_match_eta = tarboxes_ceta[tar_box_match_idx]
        tar_box_match_phi = tarboxes_cphi[tar_box_match_idx]
        p_box_match_pt  = p_momenta[p_box_match_idx]
        p_box_match_eta = pboxes_ceta[p_box_match_idx]
        p_box_match_phi = pboxes_cphi[p_box_match_idx]
        p_box_match_scr = scores[p_box_match_idx]

        results['tarboxes_matched_pt'].append(tar_box_match_pt)
        results['pboxes_matched_pt'].append(p_box_match_pt)
        results['tarboxes_matched_eta'].append(tar_box_match_eta)
        results['pboxes_matched_eta'].append(p_box_match_eta)
        results['tarboxes_matched_phi'].append(tar_box_match_phi)
        results['pboxes_matched_phi'].append(p_box_match_phi)
        results['pboxes_matched_scr'].append(p_box_match_scr)

        match_t_mask = torch.zeros(targ_momenta.size(0), dtype=torch.bool)
        match_t_mask[tar_box_match_idx] = True
        match_p_mask = torch.zeros(p_momenta.size(0), dtype=torch.bool)
        match_p_mask[p_box_match_idx] = True
        results['tarboxes_unmatched_pt'].append(targ_momenta[~match_t_mask])
        results['pboxes_unmatched_pt'].append(p_momenta[~match_p_mask])
        results['tarboxes_unmatched_eta'].append(tarboxes_ceta[~match_t_mask])
        results['pboxes_unmatched_eta'].append(pboxes_ceta[~match_p_mask])
        results['tarboxes_unmatched_phi'].append(tarboxes_cphi[~match_t_mask])
        results['pboxes_unmatched_phi'].append(pboxes_cphi[~match_p_mask])
        results['pboxes_unmatched_scr'].append(scores[~match_p_mask])

        # dR matching
        tar_box_dRmatch_idx, p_box_dRmatch_idx = dR_box_matching(tarboxes_ceta, tarboxes_cphi, pboxes_ceta, pboxes_cphi, dR_thresh=0.3)
        tar_box_dRmatch_pt  = targ_momenta[tar_box_dRmatch_idx]
        tar_box_dRmatch_eta = tarboxes_ceta[tar_box_dRmatch_idx]
        tar_box_dRmatch_phi = tarboxes_cphi[tar_box_dRmatch_idx]
        p_box_dRmatch_pt  = p_momenta[p_box_dRmatch_idx]
        p_box_dRmatch_eta = pboxes_ceta[p_box_dRmatch_idx]
        p_box_dRmatch_phi = pboxes_cphi[p_box_dRmatch_idx]
        p_box_dRmatch_scr = scores[p_box_dRmatch_idx]

        results['tarboxes_dRmatched_pt'].append(tar_box_dRmatch_pt)
        results['pboxes_dRmatched_pt'].append(p_box_dRmatch_pt)
        results['tarboxes_dRmatched_eta'].append(tar_box_dRmatch_eta)
        results['pboxes_dRmatched_eta'].append(p_box_dRmatch_eta)
        results['tarboxes_dRmatched_phi'].append(tar_box_dRmatch_phi)
        results['pboxes_dRmatched_phi'].append(p_box_dRmatch_phi)
        results['pboxes_dRmatched_scr'].append(p_box_dRmatch_scr)

        dRmatch_t_mask = torch.zeros(targ_momenta.size(0), dtype=torch.bool)
        dRmatch_t_mask[tar_box_dRmatch_idx] = True
        dRmatch_p_mask = torch.zeros(p_momenta.size(0), dtype=torch.bool)
        dRmatch_p_mask[p_box_dRmatch_idx] = True
        results['tarboxes_dRunmatched_pt'].append(targ_momenta[~dRmatch_t_mask])
        results['pboxes_dRunmatched_pt'].append(p_momenta[~dRmatch_p_mask])
        results['tarboxes_dRunmatched_eta'].append(tarboxes_ceta[~dRmatch_t_mask])
        results['pboxes_dRunmatched_eta'].append(pboxes_ceta[~dRmatch_p_mask])
        results['tarboxes_dRunmatched_phi'].append(tarboxes_cphi[~dRmatch_t_mask])
        results['pboxes_dRunmatched_phi'].append(pboxes_cphi[~dRmatch_p_mask])
        results['pboxes_dRunmatched_scr'].append(scores[~dRmatch_p_mask])

        # dR matching to TRUTH
        tru_box_dRtruthmatch_idx, p_box_dRtruthmatch_idx = dR_box_matching(truboxes_ceta, truboxes_cphi, pboxes_ceta, pboxes_cphi, dR_thresh=0.3)
        tru_box_dRtruthmatch_pt  = tru_momenta[tru_box_dRtruthmatch_idx]
        tru_box_dRtruthmatch_eta = truboxes_ceta[tru_box_dRtruthmatch_idx]
        tru_box_dRtruthmatch_phi = truboxes_cphi[tru_box_dRtruthmatch_idx]
        p_box_dRtruthmatch_pt  = p_momenta[p_box_dRtruthmatch_idx]
        p_box_dRtruthmatch_eta = pboxes_ceta[p_box_dRtruthmatch_idx]
        p_box_dRtruthmatch_phi = pboxes_cphi[p_box_dRtruthmatch_idx]
        p_box_dRtruthmatch_scr = scores[p_box_dRtruthmatch_idx]

        results['truboxes_dRtruthmatched_pt'].append(tru_box_dRtruthmatch_pt)
        results['truboxes_dRtruthmatched_eta'].append(tru_box_dRtruthmatch_eta)
        results['truboxes_dRtruthmatched_phi'].append(tru_box_dRtruthmatch_phi)
        results['pboxes_dRtruthmatched_pt'].append(p_box_dRtruthmatch_pt)
        results['pboxes_dRtruthmatched_eta'].append(p_box_dRtruthmatch_eta)
        results['pboxes_dRtruthmatched_phi'].append(p_box_dRtruthmatch_phi)
        results['pboxes_dRtruthmatched_scr'].append(p_box_dRtruthmatch_scr)

        dRtruthmatch_t_mask = torch.zeros(tru_momenta.size(0), dtype=torch.bool)
        dRtruthmatch_t_mask[tru_box_dRtruthmatch_idx] = True
        dRtruthmatch_p_mask = torch.zeros(p_momenta.size(0), dtype=torch.bool)
        dRtruthmatch_p_mask[p_box_dRtruthmatch_idx] = True
        results['truboxes_dRtruthunmatched_pt'].append(tru_momenta[~dRtruthmatch_t_mask])
        results['truboxes_dRtruthunmatched_eta'].append(truboxes_ceta[~dRtruthmatch_t_mask])
        results['truboxes_dRtruthunmatched_phi'].append(truboxes_cphi[~dRtruthmatch_t_mask])
        results['pboxes_dRtruthunmatched_pt'].append(p_momenta[~dRtruthmatch_p_mask])
        results['pboxes_dRtruthunmatched_eta'].append(pboxes_ceta[~dRtruthmatch_p_mask])
        results['pboxes_dRtruthunmatched_phi'].append(pboxes_cphi[~dRtruthmatch_p_mask])
        results['pboxes_dRtruthunmatched_scr'].append(scores[~dRtruthmatch_p_mask])

        # dR matching TARGET to TRUTH
        tru_box_dRtruthtarmatch_idx, tar_box_dRtruthtarmatch_idx = dR_box_matching(truboxes_ceta, truboxes_cphi, tarboxes_ceta, tarboxes_cphi, dR_thresh=0.3)
        tru_box_dRtruthtarmatch_pt  = tru_momenta[tru_box_dRtruthtarmatch_idx]
        tru_box_dRtruthtarmatch_eta = truboxes_ceta[tru_box_dRtruthtarmatch_idx]
        tru_box_dRtruthtarmatch_phi = truboxes_cphi[tru_box_dRtruthtarmatch_idx]
        tar_box_dRtruthtarmatch_pt  = targ_momenta[tar_box_dRtruthtarmatch_idx]
        tar_box_dRtruthtarmatch_eta = tarboxes_ceta[tar_box_dRtruthtarmatch_idx]
        tar_box_dRtruthtarmatch_phi = tarboxes_cphi[tar_box_dRtruthtarmatch_idx]

        results['truboxes_dRtruthtarmatched_pt'].append(tru_box_dRtruthtarmatch_pt)
        results['truboxes_dRtruthtarmatched_eta'].append(tru_box_dRtruthtarmatch_eta)
        results['truboxes_dRtruthtarmatched_phi'].append(tru_box_dRtruthtarmatch_phi)
        results['tarboxes_dRtruthtarmatched_pt'].append(tar_box_dRtruthtarmatch_pt)
        results['tarboxes_dRtruthtarmatched_eta'].append(tar_box_dRtruthtarmatch_eta)
        results['tarboxes_dRtruthtarmatched_phi'].append(tar_box_dRtruthtarmatch_phi)

        dRtruthtarmatch_tru_mask = torch.zeros(tru_momenta.size(0), dtype=torch.bool)
        dRtruthtarmatch_tru_mask[tru_box_dRtruthtarmatch_idx] = True
        dRtruthtarmatch_tar_mask = torch.zeros(targ_momenta.size(0), dtype=torch.bool)
        dRtruthtarmatch_tar_mask[tar_box_dRtruthtarmatch_idx] = True

        results['truboxes_dRtruthtarunmatched_pt'].append(tru_momenta[~dRtruthtarmatch_tru_mask])
        results['truboxes_dRtruthtarunmatched_eta'].append(truboxes_ceta[~dRtruthtarmatch_tru_mask])
        results['truboxes_dRtruthtarunmatched_phi'].append(truboxes_cphi[~dRtruthtarmatch_tru_mask])
        results['tarboxes_dRtruthtarunmatched_pt'].append(targ_momenta[~dRtruthtarmatch_tar_mask])
        results['tarboxes_dRtruthtarunmatched_eta'].append(tarboxes_ceta[~dRtruthtarmatch_tar_mask])
        results['tarboxes_dRtruthtarunmatched_phi'].append(tarboxes_cphi[~dRtruthtarmatch_tar_mask])

        # # if len(p_box_dRmatch_idx) != len(p_box_match_idx):
        # #     print("IoU matching:",tar_box_match_idx, p_box_match_idx)
        # #     print("dR  matching:",tar_box_dRmatch_idx, p_box_dRmatch_idx)
        # import matplotlib
        # import matplotlib.pyplot as plt
        # f,ax = plt.subplots(1,1,figsize=(10,12))   
        # ax.axhline(y=MIN_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
        # ax.axhline(y=MAX_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)

        # for k in range(len(targs)):
        #     bbx,pt = targs[k],targ_momenta[k]
        #     x,y=float(bbx[0]),float(bbx[1])
        #     w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
        #     ls = '--' if torch.isin(k,tar_box_match_idx) else '-'
        #     ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,ls=ls,lw=1.8,ec='limegreen',fc='none'))
        #     ax.text(x+w-0.3,y+h-0.15, f"{k}",color='black',fontsize=8)
        #     ax.text(x+0.05,y+h-0.15, f"{pt:.0f}",color='limegreen',fontsize=8)

        # for j in range(len(pees)):
        #     bbx,scr,pt = pees[j],scores[j],p_momenta[j]
        #     x,y=float(bbx[0]),float(bbx[1])
        #     w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
        #     ls = '--' if torch.isin(j,p_box_match_idx) else '-'
        #     ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,ls=ls,lw=1.9,ec='red',fc='none'))
        #     ax.text(x+w-0.3,y+h-0.15, f"{j} ; {scr.item():.2f}",color='black',fontsize=8)
        #     ax.text(x+0.05,y+h/20, f"{pt.item():.0f}",color='red',fontsize=8)

        # for l in range(len(trus)):
        #     bbx,pt = trus[l],tru_momenta[l]
        #     x,y=float(bbx[0]),float(bbx[1])
        #     w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
        #     ls = '--' if torch.isin(l,tru_box_match_idx) else '-'
        #     ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,ls=ls,lw=1.8,ec='gold',fc='none'))
        #     ax.text(x+w-0.3,y+h-0.15, f"{i}",color='black',fontsize=8)
        #     ax.text(x+0.05,y+h-0.15, f"{pt:.0f}",color='gold',fontsize=8)
        
        # ax.scatter(tarboxes_ceta[tar_box_dRmatch_idx], tarboxes_cphi[tar_box_dRmatch_idx], alpha=0.6, color='limegreen',s=20,marker='*')
        # ax.scatter(pboxes_ceta[p_box_dRmatch_idx], pboxes_cphi[p_box_dRmatch_idx], alpha=0.6, color='red',s=20,marker='x')

        # ax.set(xlabel='$\eta$',ylabel='$\phi$',xlim=(extent_i[0],extent_i[1]),ylim=(extent_i[2],extent_i[3]))
        # plt.tight_layout()
        # f.savefig(f'ex-NMS-{i}.png',dpi=400)
        # quit()

        results['evt_weight'].append(evt_weight)
        results['jet_evt_weight'].append([evt_weight for i in range(len(tarboxes_ceta))])

        results['pboxes_scores'].append(scores)
        results['pboxes_pt'].append(p_momenta)
        results['pboxes_eta'].append(pboxes_ceta)
        results['pboxes_phi'].append(pboxes_cphi)

        results['tarboxes_pt'].append(targ_momenta)
        results['tarboxes_eta'].append(tarboxes_ceta)
        results['tarboxes_phi'].append(tarboxes_cphi)

        results['truboxes_pt'].append(tru_momenta)
        results['truboxes_eta'].append(truboxes_ceta)
        results['truboxes_phi'].append(truboxes_cphi)

        results['n_targets'].append(len(targs))
        results['n_truth'].append(len(trus))
        results['n_preds'].append(len(pees))
        results['delta_n'].append(len(pees)-len(targs))
        results['delta_n_truth'].append(len(pees)-len(trus))

        results['n_matched_targets'].append(len(tar_box_match_pt))
        results['n_unmatched_targets'].append(len(targs)-len(tar_box_match_pt))
        results['n_matched_preds'].append(len(p_box_match_pt))
        results['n_unmatched_preds'].append(len(pees)-len(p_box_match_pt))

        results['n_dRmatched_targets'].append(len(tar_box_dRmatch_pt))
        results['n_dRunmatched_targets'].append(len(targs)-len(tar_box_dRmatch_pt))
        results['n_dRmatched_preds'].append(len(p_box_dRmatch_pt))
        results['n_dRunmatched_preds'].append(len(pees)-len(p_box_dRmatch_pt))

        results['n_dRtruthmatched_truths'].append(len(tru_box_dRtruthmatch_pt))
        results['n_dRtruthunmatched_truths'].append(len(targs)-len(tru_box_dRtruthmatch_pt))
        results['n_dRtruthmatched_preds'].append(len(p_box_dRtruthmatch_pt))
        results['n_dRtruthunmatched_preds'].append(len(pees)-len(p_box_dRtruthmatch_pt))

    save_loc = save_folder + "/box_metrics/"

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    print('Saving the box metrics in lists...')
    #automate this saving!
    save_object(results['tarboxes_pt'], save_loc+'tarboxes_pt.pkl')
    save_object(results['tarboxes_eta'], save_loc+'tarboxes_eta.pkl')
    save_object(results['tarboxes_phi'], save_loc+'tarboxes_phi.pkl')

    save_object(results['truboxes_pt'], save_loc+'truboxes_pt.pkl')
    save_object(results['truboxes_eta'], save_loc+'truboxes_eta.pkl')
    save_object(results['truboxes_phi'], save_loc+'truboxes_phi.pkl')

    save_object(results['pboxes_scores'], save_loc+'pboxes_scores.pkl')
    save_object(results['pboxes_pt'], save_loc+'pboxes_pt.pkl')
    save_object(results['pboxes_eta'], save_loc+'pboxes_eta.pkl')
    save_object(results['pboxes_phi'], save_loc+'pboxes_phi.pkl')

    save_object(results['tarboxes_matched_pt'], save_loc+'tarboxes_matched_pt.pkl')
    save_object(results['pboxes_matched_pt'], save_loc+'pboxes_matched_pt.pkl')
    save_object(results['tarboxes_matched_eta'], save_loc+'tarboxes_matched_eta.pkl')
    save_object(results['pboxes_matched_eta'], save_loc+'pboxes_matched_eta.pkl')
    save_object(results['tarboxes_matched_phi'], save_loc+'tarboxes_matched_phi.pkl')
    save_object(results['pboxes_matched_phi'], save_loc+'pboxes_matched_phi.pkl')
    save_object(results['pboxes_matched_scr'], save_loc+'pboxes_matched_scr.pkl')
    
    save_object(results['tarboxes_unmatched_pt'], save_loc+'tarboxes_unmatched_pt.pkl')
    save_object(results['pboxes_unmatched_pt'], save_loc+'pboxes_unmatched_pt.pkl')
    save_object(results['tarboxes_unmatched_eta'], save_loc+'tarboxes_unmatched_eta.pkl')
    save_object(results['pboxes_unmatched_eta'], save_loc+'pboxes_unmatched_eta.pkl')
    save_object(results['tarboxes_unmatched_phi'], save_loc+'tarboxes_unmatched_phi.pkl')
    save_object(results['pboxes_unmatched_phi'], save_loc+'pboxes_unmatched_phi.pkl')
    save_object(results['pboxes_unmatched_scr'], save_loc+'pboxes_unmatched_scr.pkl')

    save_object(results['tarboxes_dRmatched_pt'], save_loc+'tarboxes_dRmatched_pt.pkl')
    save_object(results['pboxes_dRmatched_pt'], save_loc+'pboxes_dRmatched_pt.pkl')
    save_object(results['tarboxes_dRmatched_eta'], save_loc+'tarboxes_dRmatched_eta.pkl')
    save_object(results['pboxes_dRmatched_eta'], save_loc+'pboxes_dRmatched_eta.pkl')
    save_object(results['tarboxes_dRmatched_phi'], save_loc+'tarboxes_dRmatched_phi.pkl')
    save_object(results['pboxes_dRmatched_phi'], save_loc+'pboxes_dRmatched_phi.pkl')
    save_object(results['pboxes_dRmatched_scr'], save_loc+'pboxes_dRmatched_scr.pkl')

    save_object(results['tarboxes_dRunmatched_pt'], save_loc+'tarboxes_dRunmatched_pt.pkl')
    save_object(results['pboxes_dRunmatched_pt'], save_loc+'pboxes_dRunmatched_pt.pkl')
    save_object(results['tarboxes_dRunmatched_eta'], save_loc+'tarboxes_dRunmatched_eta.pkl')
    save_object(results['pboxes_dRunmatched_eta'], save_loc+'pboxes_dRunmatched_eta.pkl')
    save_object(results['tarboxes_dRunmatched_phi'], save_loc+'tarboxes_dRunmatched_phi.pkl')
    save_object(results['pboxes_dRunmatched_phi'], save_loc+'pboxes_dRunmatched_phi.pkl')
    save_object(results['pboxes_dRunmatched_scr'], save_loc+'pboxes_dRunmatched_scr.pkl')

    save_object(results['truboxes_dRtruthmatched_pt'], save_loc+'truboxes_dRtruthmatched_pt.pkl')
    save_object(results['truboxes_dRtruthmatched_eta'], save_loc+'truboxes_dRtruthmatched_eta.pkl')
    save_object(results['truboxes_dRtruthmatched_phi'], save_loc+'truboxes_dRtruthmatched_phi.pkl')
    save_object(results['pboxes_dRtruthmatched_pt'], save_loc+'pboxes_dRtruthmatched_pt.pkl')
    save_object(results['pboxes_dRtruthmatched_eta'], save_loc+'pboxes_dRtruthmatched_eta.pkl')
    save_object(results['pboxes_dRtruthmatched_phi'], save_loc+'pboxes_dRtruthmatched_phi.pkl')
    save_object(results['pboxes_dRtruthmatched_scr'], save_loc+'pboxes_dRtruthmatched_scr.pkl')

    save_object(results['truboxes_dRtruthunmatched_pt'], save_loc+'truboxes_dRtruthunmatched_pt.pkl')
    save_object(results['truboxes_dRtruthunmatched_eta'], save_loc+'truboxes_dRtruthunmatched_eta.pkl')
    save_object(results['truboxes_dRtruthunmatched_phi'], save_loc+'truboxes_dRtruthunmatched_phi.pkl')
    save_object(results['pboxes_dRtruthunmatched_pt'], save_loc+'pboxes_dRtruthunmatched_pt.pkl')
    save_object(results['pboxes_dRtruthunmatched_eta'], save_loc+'pboxes_dRtruthunmatched_eta.pkl')#? left out
    save_object(results['pboxes_dRtruthunmatched_phi'], save_loc+'pboxes_dRtruthunmatched_phi.pkl')
    save_object(results['pboxes_dRtruthunmatched_scr'], save_loc+'pboxes_dRtruthunmatched_scr.pkl')


    # targets dR matched to truth
    save_object(results['truboxes_dRtruthtarmatched_pt'], save_loc+'truboxes_dRtruthtarmatched_pt.pkl')
    save_object(results['truboxes_dRtruthtarmatched_eta'], save_loc+'truboxes_dRtruthtarmatched_eta.pkl')
    save_object(results['truboxes_dRtruthtarmatched_phi'], save_loc+'truboxes_dRtruthtarmatched_phi.pkl')
    save_object(results['tarboxes_dRtruthtarmatched_pt'], save_loc+'tarboxes_dRtruthtarmatched_pt.pkl')
    save_object(results['tarboxes_dRtruthtarmatched_eta'], save_loc+'tarboxes_dRtruthtarmatched_eta.pkl')
    save_object(results['tarboxes_dRtruthtarmatched_phi'], save_loc+'tarboxes_dRtruthtarmatched_phi.pkl')
    save_object(results['truboxes_dRtruthtarunmatched_pt'], save_loc+'truboxes_dRtruthtarunmatched_pt.pkl')
    save_object(results['truboxes_dRtruthtarunmatched_eta'], save_loc+'truboxes_dRtruthtarunmatched_eta.pkl')
    save_object(results['truboxes_dRtruthtarunmatched_phi'], save_loc+'truboxes_dRtruthtarunmatched_phi.pkl')
    save_object(results['tarboxes_dRtruthtarunmatched_pt'], save_loc+'tarboxes_dRtruthtarunmatched_pt.pkl')
    save_object(results['tarboxes_dRtruthtarunmatched_eta'], save_loc+'tarboxes_dRtruthtarunmatched_eta.pkl')
    save_object(results['tarboxes_dRtruthtarunmatched_phi'], save_loc+'tarboxes_dRtruthtarunmatched_phi.pkl')


    save_object(results['evt_weight'],save_loc+'evt_weight.pkl')
    save_object(results['jet_evt_weight'],save_loc+'jet_evt_weight.pkl')

    save_object(results['n_targets'],save_loc+'n_targets.pkl')
    save_object(results['n_truth'],save_loc+'n_truth.pkl')
    save_object(results['n_preds'], save_loc+'n_preds.pkl')
    save_object(results['delta_n'], save_loc+'delta_n.pkl')
    save_object(results['delta_n_truth'], save_loc+'delta_n_truth.pkl')
    
    save_object(results['n_matched_targets'],save_loc+'n_matched_targets.pkl')
    save_object(results['n_unmatched_targets'],save_loc+'n_unmatched_targets.pkl')
    save_object(results['n_matched_preds'],save_loc+'n_matched_preds.pkl')
    save_object(results['n_unmatched_preds'],save_loc+'n_unmatched_preds.pkl')

    save_object(results['n_dRmatched_targets'],save_loc+'n_dRmatched_targets.pkl')
    save_object(results['n_dRunmatched_targets'],save_loc+'n_dRunmatched_targets.pkl')
    save_object(results['n_dRmatched_preds'],save_loc+'n_dRmatched_preds.pkl')
    save_object(results['n_dRunmatched_preds'],save_loc+'n_dRunmatched_preds.pkl')

    save_object(results['n_dRtruthmatched_truths'],save_loc+'n_dRtruthmatched_truths.pkl')
    save_object(results['n_dRtruthunmatched_truths'],save_loc+'n_dRtruthunmatched_truths.pkl')
    save_object(results['n_dRtruthmatched_preds'],save_loc+'n_dRtruthmatched_preds.pkl')
    save_object(results['n_dRtruthunmatched_preds'],save_loc+'n_dRtruthunmatched_preds.pkl')

    return



if __name__=="__main__":
    # model_name = "jetSSD_custom_convnext_central_32e"
    model_name = "jetSSD_smallconvnext_central_40e"
    # proc = "JZcomb0_test"
    # date = "20250313-06"
    # date = "20250406-23" # square
    # proc = "ttbar_test"
    proc = "JZ0"
    # date = "20250407-14"
    # date = "20250626-18"
    date = "20251114-09"
    folder_to_look_in = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/{proc}/{date}/"
    save_at = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/{proc}/{date}/"

    print('Making box metrics')
    calculate_box_metrics(folder_to_look_in,save_at)
    print('Completed box metrics\n')



