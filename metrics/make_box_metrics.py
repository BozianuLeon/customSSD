import numpy as np
import torch
import sys
import os

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


from utils import target_box_matching, target_box_match_pt, wrap_check_truth3, wrap_check_NMS3



MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
# MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496
MIN_CELLS_ETA,MAX_CELLS_ETA = -2.5, 2.5
# EXTENT = [-4.82349586, 4.82349586, -6.21738815, 6.21801758] 
EXTENT = (-2.4999826, 2.4999774, -6.217388274177672, 6.2180176992265)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)


results = {
           'tboxes_pt':         [],
           'tboxes_matched':    [],
           'tboxes_eta':        [], 
           'tboxes_phi':        [], 

           'pboxes_matched':    [],
           'pboxes_scores':     [], 
           'pboxes_pt':         [], 
           'pboxes_eta':        [], 
           'pboxes_phi':        [], 

           'tboxes_matched_pt': [], 
           'pboxes_matched_pt': [], 

           'n_truth':           [],
           'n_preds':           [],
           'delta_n':           [],
           'n_matched_truth':   [],
           'n_unmatched_truth': [],
           'n_matched_preds':   [],
           'n_unmatched_preds': [],
}



def calculate_box_metrics(
    folder_containing_struc_array,
    save_folder,
):

    with open(folder_containing_struc_array + "/struc_array.npy", 'rb') as f:
        a = np.load(f)

    for i in range(len(a)):
        # i=2368
        extent_i = a[i]['extent']
        preds = a[i]['p_boxes']
        scores = a[i]['p_scores']
        p_momenta = a[i]['p_pt']
        trues = a[i]['t_boxes']
        t_momenta = a[i]['t_pt']

        #remove padding
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
        # #could filter by score again here i.e pees[scores>0.85]
        # centre of the truth/pred boxes
        tboxes_ceta = (tees[:,2] + tees[:,0])/2
        pboxes_ceta = (pees[:,2] + pees[:,0])/2
        tboxes_cphi = (tees[:,3] + tees[:,1])/2
        pboxes_cphi = (pees[:,3] + pees[:,1])/2


        t_box_match_pt, p_box_match_pt = target_box_match_pt(tees, t_momenta, pees, p_momenta, iou_thresh=0.5)
        # get matched pred and gt boxes pt in order 
        results['tboxes_matched_pt'].append(t_box_match_pt)
        results['pboxes_matched_pt'].append(p_box_match_pt)

        t_box_match, p_box_match = target_box_matching(tees, pees, iou_thresh=0.5)
        # print(len(t_box_match_pt)-sum(t_box_match).item())

        # check if tbox tb is "matched" to any box in pees
        results['tboxes_matched'].append(t_box_match.to(dtype=torch.int).tolist())
        # check if pbox pb is "matched" to any box in tees
        results['pboxes_matched'].append(p_box_match.to(dtype=torch.int).tolist())

        tboxes_ceta = (tees[:,2] + tees[:,0])/2
        tboxes_cphi = (tees[:,3] + tees[:,1])/2
        pboxes_ceta = (pees[:,2] + pees[:,0])/2
        pboxes_cphi = (pees[:,3] + pees[:,1])/2

        results['pboxes_scores'].append(scores)
        results['pboxes_pt'].append(p_momenta)
        results['pboxes_eta'].append(pboxes_ceta)
        results['pboxes_phi'].append(pboxes_cphi)

        results['tboxes_pt'].append(t_momenta)
        results['tboxes_eta'].append(tboxes_ceta)
        results['tboxes_phi'].append(tboxes_cphi)

        #store the results
        results['n_truth'].append(len(tees))
        results['n_preds'].append(len(pees))
        results['delta_n'].append(len(pees)-len(tees))
        results['n_matched_truth'].append(sum(t_box_match))
        results['n_unmatched_truth'].append(len(t_box_match[t_box_match==0]))
        results['n_matched_preds'].append(sum(p_box_match))
        results['n_unmatched_preds'].append(len(p_box_match[p_box_match==0]))

    save_loc = save_folder + "/box_metrics/"

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    print('Saving the box metrics in lists...')
    #automate this saving!
    save_object(results['tboxes_pt'], save_loc+'tboxes_pt.pkl')
    save_object(results['tboxes_eta'], save_loc+'tboxes_eta.pkl')
    save_object(results['tboxes_phi'], save_loc+'tboxes_phi.pkl')
    save_object(results['tboxes_matched'], save_loc+'tboxes_matched.pkl')

    save_object(results['pboxes_matched'], save_loc+'pboxes_matched.pkl')
    save_object(results['pboxes_scores'], save_loc+'pboxes_scores.pkl')
    save_object(results['pboxes_pt'], save_loc+'pboxes_pt.pkl')
    save_object(results['pboxes_eta'], save_loc+'pboxes_eta.pkl')
    save_object(results['pboxes_phi'], save_loc+'pboxes_phi.pkl')

    save_object(results['tboxes_matched_pt'], save_loc+'tboxes_matched_pt.pkl')
    save_object(results['pboxes_matched_pt'], save_loc+'pboxes_matched_pt.pkl')

    save_object(results['n_truth'],save_loc+'n_truth.pkl')
    save_object(results['n_preds'], save_loc+'n_preds.pkl')
    save_object(results['delta_n'], save_loc+'delta_n.pkl')
    
    save_object(results['n_matched_truth'],save_loc+'n_matched_truth.pkl')
    save_object(results['n_unmatched_truth'],save_loc+'n_unmatched_truth.pkl')
    save_object(results['n_matched_preds'],save_loc+'n_matched_preds.pkl')
    save_object(results['n_unmatched_preds'],save_loc+'n_unmatched_preds.pkl')

    return



if __name__=="__main__":
    model_name = "jetSSD_smallconvnext_central_32e"
    folder_to_look_in = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/JZ4/20250124-13/"
    save_at = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/JZ4/20250124-13/"

    print('Making box metrics')
    calculate_box_metrics(folder_to_look_in,save_at)
    print('Completed box metrics\n')








