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
           'tboxes_pt':         [],
           'tboxes_eta':        [], 
           'tboxes_phi':        [], 

           'pboxes_scores':     [], 
           'pboxes_pt':         [], 
           'pboxes_eta':        [], 
           'pboxes_phi':        [], 

           'tboxes_matched_pt': [], 
           'pboxes_matched_pt': [], 
           'tboxes_matched_eta':[], 
           'pboxes_matched_eta':[], 
           'tboxes_matched_phi':[], 
           'pboxes_matched_phi':[], 
           'pboxes_matched_scr':[], 

           'tboxes_unmatched_pt': [], 
           'pboxes_unmatched_pt': [], 
           'tboxes_unmatched_eta':[], 
           'pboxes_unmatched_eta':[], 
           'tboxes_unmatched_phi':[], 
           'pboxes_unmatched_phi':[], 
           'pboxes_unmatched_scr':[], 

           'tboxes_dRmatched_pt':[],
           'pboxes_dRmatched_pt':[],
           'tboxes_dRmatched_eta':[],
           'pboxes_dRmatched_eta':[],
           'tboxes_dRmatched_phi':[],
           'pboxes_dRmatched_phi':[],
           'pboxes_dRmatched_scr':[],

           'tboxes_dRunmatched_pt':[],
           'pboxes_dRunmatched_pt':[],
           'tboxes_dRunmatched_eta':[],
           'pboxes_dRunmatched_eta':[],
           'tboxes_dRunmatched_phi':[],
           'pboxes_dRunmatched_phi':[],
           'pboxes_dRunmatched_scr':[],

           'n_truth':           [],
           'n_preds':           [],
           'delta_n':           [],

           'n_matched_truth':   [],
           'n_unmatched_truth': [],
           'n_matched_preds':   [],
           'n_unmatched_preds': [],

           'n_dRmatched_truth':   [],
           'n_dRunmatched_truth': [],
           'n_dRmatched_preds':   [],
           'n_dRunmatched_preds': [],
}



def calculate_box_metrics(
    folder_containing_struc_array,
    save_folder,
):

    with open(folder_containing_struc_array + "/struc_array.npy", 'rb') as f:
        a = np.load(f)

    for i in range(len(a)):
        # i = 2387
        extent_i = a[i]['extent']
        preds = a[i]['p_boxes']
        scores = a[i]['p_scores']
        p_momenta = a[i]['p_pt']
        trues = a[i]['t_boxes']
        t_momenta = a[i]['t_pt']

        #remove padding, boxes in xyxy coordinates
        pees = preds[(preds[:, 2] - preds[:, 0]) >= 0.01]
        scores = scores[scores > 0.01]
        p_momenta = p_momenta[p_momenta > 0.01]
        tees = trues[(trues[:, 2] - trues[:, 0]) >= 0.01]
        t_momenta = t_momenta[t_momenta > 0.01]

        if (len(pees)==0) or (len(tees)==0):
            print("No predicted jets")
            continue

        pees, scores, p_momenta = wrap_check_NMS3(pees,scores,p_momenta,iou_thresh=0.3)
        tees, t_momenta = wrap_check_truth3(tees,t_momenta,MIN_CELLS_PHI,MAX_CELLS_PHI)
    
        print(i)

        # centre of the truth/pred boxes
        tboxes_ceta = (tees[:,2] + tees[:,0])/2
        pboxes_ceta = (pees[:,2] + pees[:,0])/2
        tboxes_cphi = (tees[:,3] + tees[:,1])/2
        pboxes_cphi = (pees[:,3] + pees[:,1])/2
        pboxes_cphi = clip_phi(pboxes_cphi) # ensure phi values in [-pi,pi]


        # IoU matching
        t_box_match_idx, p_box_match_idx = iou_box_matching(tees, pees, iou_thresh=0.5)
        t_box_match_pt = t_momenta[t_box_match_idx]
        p_box_match_pt = p_momenta[p_box_match_idx]
        t_box_match_eta = tboxes_ceta[t_box_match_idx]
        p_box_match_eta = pboxes_ceta[p_box_match_idx]
        t_box_match_phi = tboxes_cphi[t_box_match_idx]
        p_box_match_phi = pboxes_cphi[p_box_match_idx]
        p_box_match_scr = scores[p_box_match_idx]

        results['tboxes_matched_pt'].append(t_box_match_pt)
        results['pboxes_matched_pt'].append(p_box_match_pt)
        results['tboxes_matched_eta'].append(t_box_match_eta)
        results['pboxes_matched_eta'].append(p_box_match_eta)
        results['tboxes_matched_phi'].append(t_box_match_phi)
        results['pboxes_matched_phi'].append(p_box_match_phi)
        results['pboxes_matched_scr'].append(p_box_match_scr)

        match_t_mask = torch.zeros(t_momenta.size(0), dtype=torch.bool)
        match_t_mask[t_box_match_idx] = True
        match_p_mask = torch.zeros(p_momenta.size(0), dtype=torch.bool)
        match_p_mask[p_box_match_idx] = True
        results['tboxes_unmatched_pt'].append(t_momenta[~match_t_mask])
        results['pboxes_unmatched_pt'].append(p_momenta[~match_p_mask])
        results['tboxes_unmatched_eta'].append(tboxes_ceta[~match_t_mask])
        results['pboxes_unmatched_eta'].append(pboxes_ceta[~match_p_mask])
        results['tboxes_unmatched_phi'].append(tboxes_cphi[~match_t_mask])
        results['pboxes_unmatched_phi'].append(pboxes_cphi[~match_p_mask])
        results['pboxes_unmatched_scr'].append(scores[~match_p_mask])

        # dR matching
        t_box_dRmatch_idx, p_box_dRmatch_idx = dR_box_matching(tboxes_ceta, tboxes_cphi, pboxes_ceta, pboxes_cphi, dR_thresh=0.4)
        t_box_dRmatch_pt = t_momenta[t_box_dRmatch_idx]
        p_box_dRmatch_pt = p_momenta[p_box_dRmatch_idx]
        t_box_dRmatch_eta = tboxes_ceta[t_box_dRmatch_idx]
        p_box_dRmatch_eta = pboxes_ceta[p_box_dRmatch_idx]
        t_box_dRmatch_phi = tboxes_cphi[t_box_dRmatch_idx]
        p_box_dRmatch_phi = pboxes_cphi[p_box_dRmatch_idx]
        p_box_dRmatch_scr = scores[p_box_dRmatch_idx]

        results['tboxes_dRmatched_pt'].append(t_box_dRmatch_pt)
        results['pboxes_dRmatched_pt'].append(p_box_dRmatch_pt)
        results['tboxes_dRmatched_eta'].append(t_box_dRmatch_eta)
        results['pboxes_dRmatched_eta'].append(p_box_dRmatch_eta)
        results['tboxes_dRmatched_phi'].append(t_box_dRmatch_phi)
        results['pboxes_dRmatched_phi'].append(p_box_dRmatch_phi)
        results['pboxes_dRmatched_scr'].append(p_box_dRmatch_scr)

        dRmatch_t_mask = torch.zeros(t_momenta.size(0), dtype=torch.bool)
        dRmatch_t_mask[t_box_dRmatch_idx] = True
        dRmatch_p_mask = torch.zeros(p_momenta.size(0), dtype=torch.bool)
        dRmatch_p_mask[p_box_dRmatch_idx] = True
        results['tboxes_dRunmatched_pt'].append(t_momenta[~dRmatch_t_mask])
        results['pboxes_dRunmatched_pt'].append(p_momenta[~dRmatch_p_mask])
        results['tboxes_dRunmatched_eta'].append(tboxes_ceta[~dRmatch_t_mask])
        results['pboxes_dRunmatched_eta'].append(pboxes_ceta[~dRmatch_p_mask])
        results['tboxes_dRunmatched_phi'].append(tboxes_cphi[~dRmatch_t_mask])
        results['pboxes_dRunmatched_phi'].append(pboxes_cphi[~dRmatch_p_mask])
        results['pboxes_dRunmatched_scr'].append(scores[~dRmatch_p_mask])


        # # if len(p_box_dRmatch_idx) != len(p_box_match_idx):
        # #     print("IoU matching:",t_box_match_idx, p_box_match_idx)
        # #     print("dR  matching:",t_box_dRmatch_idx, p_box_dRmatch_idx)
        # import matplotlib
        # import matplotlib.pyplot as plt
        # f,ax = plt.subplots(1,1,figsize=(10,12))   
        # ax.axhline(y=MIN_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)
        # ax.axhline(y=MAX_CELLS_PHI, color='red', alpha=0.6, linestyle='--',lw=0.7)

        # for k in range(len(tees)):
        #     bbx,pt = tees[k],t_momenta[k]
        #     x,y=float(bbx[0]),float(bbx[1])
        #     w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
        #     ls = '--' if torch.isin(k,t_box_match_idx) else '-'
        #     ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,ls=ls,lw=1.8,ec='limegreen',fc='none'))
        #     ax.text(x+0.05,y+h-0.15, f"{k}, {pt:.0f}",color='limegreen',fontsize=8)

        # for j in range(len(pees)):
        #     bbx,scr,pt = pees[j],scores[j],p_momenta[j]
        #     x,y=float(bbx[0]),float(bbx[1])
        #     w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
        #     ls = '--' if torch.isin(j,p_box_match_idx) else '-'
        #     ax.add_patch(matplotlib.patches.Rectangle((x,y),w,h,ls=ls,lw=1.9,ec='red',fc='none'))
        #     ax.text(x+w-0.3,y+h-0.15, f"{scr.item():.2f}",color='red',fontsize=8)
        #     ax.text(x+0.05,y+h/20, f"{j},{pt.item():.0f}",color='red',fontsize=8)
        
        # ax.scatter(tboxes_ceta[t_box_dRmatch_idx], tboxes_cphi[t_box_dRmatch_idx], alpha=0.6, color='limegreen',s=20,marker='*')
        # ax.scatter(pboxes_ceta[p_box_dRmatch_idx], pboxes_cphi[p_box_dRmatch_idx], alpha=0.6, color='red',s=20,marker='x')

        # ax.set(xlabel='$\eta$',ylabel='$\phi$',xlim=(extent_i[0],extent_i[1]),ylim=(extent_i[2],extent_i[3]))
        # plt.tight_layout()
        # f.savefig(f'ex-NMS-{i}.png',dpi=400)
        # quit()


        results['pboxes_scores'].append(scores)
        results['pboxes_pt'].append(p_momenta)
        results['pboxes_eta'].append(pboxes_ceta)
        results['pboxes_phi'].append(pboxes_cphi)

        results['tboxes_pt'].append(t_momenta)
        results['tboxes_eta'].append(tboxes_ceta)
        results['tboxes_phi'].append(tboxes_cphi)

        results['n_truth'].append(len(tees))
        results['n_preds'].append(len(pees))
        results['delta_n'].append(len(pees)-len(tees))

        results['n_matched_truth'].append(len(t_box_match_pt))
        results['n_unmatched_truth'].append(len(tees)-len(t_box_match_pt))
        results['n_matched_preds'].append(len(p_box_match_pt))
        results['n_unmatched_preds'].append(len(pees)-len(p_box_match_pt))

        results['n_dRmatched_truth'].append(len(t_box_dRmatch_pt))
        results['n_dRunmatched_truth'].append(len(tees)-len(t_box_dRmatch_pt))
        results['n_dRmatched_preds'].append(len(p_box_dRmatch_pt))
        results['n_dRunmatched_preds'].append(len(pees)-len(p_box_dRmatch_pt))

    save_loc = save_folder + "/box_metrics/"

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    print('Saving the box metrics in lists...')
    #automate this saving!
    save_object(results['tboxes_pt'], save_loc+'tboxes_pt.pkl')
    save_object(results['tboxes_eta'], save_loc+'tboxes_eta.pkl')
    save_object(results['tboxes_phi'], save_loc+'tboxes_phi.pkl')

    save_object(results['pboxes_scores'], save_loc+'pboxes_scores.pkl')
    save_object(results['pboxes_pt'], save_loc+'pboxes_pt.pkl')
    save_object(results['pboxes_eta'], save_loc+'pboxes_eta.pkl')
    save_object(results['pboxes_phi'], save_loc+'pboxes_phi.pkl')

    save_object(results['tboxes_matched_pt'], save_loc+'tboxes_matched_pt.pkl')
    save_object(results['pboxes_matched_pt'], save_loc+'pboxes_matched_pt.pkl')
    save_object(results['tboxes_matched_eta'], save_loc+'tboxes_matched_eta.pkl')
    save_object(results['pboxes_matched_eta'], save_loc+'pboxes_matched_eta.pkl')
    save_object(results['tboxes_matched_phi'], save_loc+'tboxes_matched_phi.pkl')
    save_object(results['pboxes_matched_phi'], save_loc+'pboxes_matched_phi.pkl')
    save_object(results['pboxes_matched_scr'], save_loc+'pboxes_matched_scr.pkl')
    
    save_object(results['tboxes_unmatched_pt'], save_loc+'tboxes_unmatched_pt.pkl')
    save_object(results['pboxes_unmatched_pt'], save_loc+'pboxes_unmatched_pt.pkl')
    save_object(results['tboxes_unmatched_eta'], save_loc+'tboxes_unmatched_eta.pkl')
    save_object(results['pboxes_unmatched_eta'], save_loc+'pboxes_unmatched_eta.pkl')
    save_object(results['tboxes_unmatched_phi'], save_loc+'tboxes_unmatched_phi.pkl')
    save_object(results['pboxes_unmatched_phi'], save_loc+'pboxes_unmatched_phi.pkl')
    save_object(results['pboxes_unmatched_scr'], save_loc+'pboxes_unmatched_scr.pkl')

    save_object(results['tboxes_dRmatched_pt'], save_loc+'tboxes_dRmatched_pt.pkl')
    save_object(results['pboxes_dRmatched_pt'], save_loc+'pboxes_dRmatched_pt.pkl')
    save_object(results['tboxes_dRmatched_eta'], save_loc+'tboxes_dRmatched_eta.pkl')
    save_object(results['pboxes_dRmatched_eta'], save_loc+'pboxes_dRmatched_eta.pkl')
    save_object(results['tboxes_dRmatched_phi'], save_loc+'tboxes_dRmatched_phi.pkl')
    save_object(results['pboxes_dRmatched_phi'], save_loc+'pboxes_dRmatched_phi.pkl')
    save_object(results['pboxes_dRmatched_scr'], save_loc+'pboxes_dRmatched_scr.pkl')

    save_object(results['tboxes_dRunmatched_pt'], save_loc+'tboxes_dRunmatched_pt.pkl')
    save_object(results['pboxes_dRunmatched_pt'], save_loc+'pboxes_dRunmatched_pt.pkl')
    save_object(results['tboxes_dRunmatched_eta'], save_loc+'tboxes_dRunmatched_eta.pkl')
    save_object(results['pboxes_dRunmatched_eta'], save_loc+'pboxes_dRunmatched_eta.pkl')
    save_object(results['tboxes_dRunmatched_phi'], save_loc+'tboxes_dRunmatched_phi.pkl')
    save_object(results['pboxes_dRunmatched_phi'], save_loc+'pboxes_dRunmatched_phi.pkl')
    save_object(results['pboxes_dRunmatched_scr'], save_loc+'pboxes_dRunmatched_scr.pkl')

    save_object(results['n_truth'],save_loc+'n_truth.pkl')
    save_object(results['n_preds'], save_loc+'n_preds.pkl')
    save_object(results['delta_n'], save_loc+'delta_n.pkl')
    
    save_object(results['n_matched_truth'],save_loc+'n_matched_truth.pkl')
    save_object(results['n_unmatched_truth'],save_loc+'n_unmatched_truth.pkl')
    save_object(results['n_matched_preds'],save_loc+'n_matched_preds.pkl')
    save_object(results['n_unmatched_preds'],save_loc+'n_unmatched_preds.pkl')

    save_object(results['n_dRmatched_truth'],save_loc+'n_dRmatched_truth.pkl')
    save_object(results['n_dRunmatched_truth'],save_loc+'n_dRunmatched_truth.pkl')
    save_object(results['n_dRmatched_preds'],save_loc+'n_dRmatched_preds.pkl')
    save_object(results['n_dRunmatched_preds'],save_loc+'n_dRunmatched_preds.pkl')

    return



if __name__=="__main__":
    model_name = "jetSSD_di_uconvnext_central_11e"
    proc = "JZ4"
    date = "20250211-13"
    folder_to_look_in = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/{proc}/{date}/"
    save_at = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/{proc}/{date}/"

    print('Making box metrics')
    calculate_box_metrics(folder_to_look_in,save_at)
    print('Completed box metrics\n')



