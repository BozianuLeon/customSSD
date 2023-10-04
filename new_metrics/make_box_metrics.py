import numpy as np
import torch
import sys
import time

import h5py
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')
# from utils.metrics import delta_n, n_unmatched_truth, n_unmatched_preds, centre_diffs, hw_diffs, area_covered
from utils.metrics import delta_n, n_matched_preds, n_unmatched_preds, n_matched_truth, n_unmatched_truth, percentage_area_covered_by_boxes
from utils.utils import wrap_check_NMS, wrap_check_truth, remove_nan
MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496

# [-4.82349586  4.82349586 -6.21738815  6.21801758]





results = {'n_truth':[],
           'n_preds':[],
           'delta_n':[],
           'n_matched_truth':[],
           'n_unmatched_truth':[],
           'n_matched_preds':[],
           'n_unmatched_preds':[],
           'percentage_total_area_covered_truth':[],
           'percentage_total_area_covered_preds':[],
           'percentage_truth_area_covered':[],   
}





def calculate_box_metrics(
    folder_containing_struc_array,
    save_folder,
):

    with open(folder_containing_struc_array + "/struc_array.npy", 'rb') as f:
        a = np.load(f)


    for i in range(len(a)):
        extent_i = a[i]['extent']
        preds = a[i]['p_boxes']
        trues = a[i]['t_boxes']
        scores = a[i]['p_scores']
        pees = preds[np.where(preds[:,0] > 0)]
        tees = trues[np.where(trues[:,0] > 0)]

        pees = torch.tensor(pees)
        tees = torch.tensor(tees)
        # wc_pred_boxes = wrap_check_NMS(pees,scores,MIN_CELLS_PHI,MAX_CELLS_PHI,threshold=0.2)
        # wc_truth_boxes = wrap_check_truth(tees,MIN_CELLS_PHI,MAX_CELLS_PHI)

        #make boxes cover extent
        tees[:,(0,2)] = (tees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
        tees[:,(1,3)] = (tees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]

        pees[:,(0,2)] = (pees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
        pees[:,(1,3)] = (pees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]
  
        #store the results
        results['n_truth'].append(len(tees))
        results['n_preds'].append(len(pees))
        results['delta_n'].append(delta_n(tees,pees))
        results['n_matched_truth'].append(n_matched_truth(tees,pees))
        results['n_unmatched_truth'].append(n_unmatched_truth(tees,pees))
        results['n_matched_preds'].append(n_matched_preds(tees,pees))
        results['n_unmatched_preds'].append(n_unmatched_preds(tees,pees))
        results['percentage_total_area_covered_truth'].append(percentage_area_covered_by_boxes(tees,extent_i))
        results['percentage_total_area_covered_preds'].append(percentage_area_covered_by_boxes(pees,extent_i))

    return






