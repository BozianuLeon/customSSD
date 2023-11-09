import numpy as np
import torch
import sys
import os
import time

import h5py
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
# caution: path[0] is reserved for script path 
sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')
from utils.utils import wrap_check_NMS, wrap_check_truth, remove_nan
from utils.metrics import grab_cells_from_boxes, extract_physics_variables
from utils.metrics import event_cluster_estimates

from utils.metrics import n_clusters_per_box, clusters_in_box_E_diff, number_cluster_in_tboxes

from utils.metrics import RetrieveCellIdsFromBox, RetrieveCellIdsFromCluster
from utils.metrics import get_physics_dictionary


MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)


eval_results = {
                'n_clusters':           [],
                'topocl_energies':      [],
                'topocl_etas':          [],
                'topocl_phis':          [],
                'topocl_n_cells':       [],
                       
                'n_tboxes':             [],
                'num_tboxes':           [],
                'tboxes_energies':      [],
                'tboxes_eta':           [],
                'tboxes_phi':           [],
                'tboxes_eT':            [],
                'tboxes_n_cells':       [],
                'tboxes_noise':         [],
                'tboxes_significance':  [],
                'tboxes_neg_frac':      [],
                'tboxes_max_frac':      [],
               
                'n_pboxes':             [],
                'num_pboxes':           [],
                'pboxes_energies':      [],
                'pboxes_eta':           [],
                'pboxes_phi':           [],
                'pboxes_eT':            [],
                'pboxes_n_cells':       [],
                'pboxes_noise':         [],
                'pboxes_significance':  [],
                'pboxes_neg_frac':      [],
                'pboxes_max_frac':      [],


}


#TODO: Make this work for matched/unmatched boxes and quickly!


def calculate_phys_metrics2(
    folder_containing_struc_array,
    save_folder,
):

    with open(folder_containing_struc_array + "/struc_array.npy", 'rb') as f:
        a = np.load(f)

    for i in range(len(a)):
        start = time.perf_counter()
        extent_i = a[i]['extent']
        preds = a[i]['p_boxes']
        trues = a[i]['t_boxes']
        scores = a[i]['p_scores']

        pees = preds[np.where(preds[:,0] > 0)]
        tees = trues[np.where(trues[:,0] > 0)]

        #make boxes cover extent
        tees[:,(0,2)] = (tees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
        tees[:,(1,3)] = (tees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]
        pees[:,(0,2)] = (pees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
        pees[:,(1,3)] = (pees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]

        #wrap check boxes here
        pees = wrap_check_NMS(pees,scores,MIN_CELLS_PHI,MAX_CELLS_PHI,threshold=0.2)
        tees = wrap_check_truth(tees,MIN_CELLS_PHI,MAX_CELLS_PHI)
        print(i)
        
        #get the cells
        h5f = a[i]['h5file']
        try:
            h5f = h5f.decode('utf-8')
        except:
            h5f = h5f
        event_no = a[i]['event_no']

        #load cells from h5
        cells_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/cells/user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        with h5py.File(cells_file,"r") as f:
            h5group = f["caloCells"]
            cells = h5group["2d"][event_no]

        clusters_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/clusters/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        with h5py.File(clusters_file,"r") as f:
            cl_data = f["caloCells"] 
            event_data = cl_data["1d"][event_no]
            cluster_data = cl_data["2d"][event_no]
            raw_E_mask = (cluster_data['cl_E_em']+cluster_data['cl_E_had']) > 5000 #5GeV cut
            cluster_data = cluster_data[raw_E_mask]
            # cluster_cell_data = cl_data["3d"][event_no]
            # cluster_cell_data = cluster_cell_data[raw_E_mask]

        eval_results['n_clusters'].append(len(cluster_data['cl_eta'].tolist()))
        eval_results['topocl_energies'].append((cluster_data['cl_E_em']+cluster_data['cl_E_had']).tolist())
        eval_results['topocl_etas'].append(cluster_data['cl_eta'].tolist())
        eval_results['topocl_phis'].append(cluster_data['cl_phi'].tolist())
        eval_results['topocl_n_cells'].append(cluster_data['cl_cell_n'].tolist())

        l_pred_cells = RetrieveCellIdsFromBox(cells,pees)
        l_true_cells = RetrieveCellIdsFromBox(cells,tees)

        tb_phys_dict = get_physics_dictionary(l_true_cells,cells)
        pb_phys_dict = get_physics_dictionary(l_pred_cells,cells)

        eval_results['n_tboxes'].append(len(tees))
        eval_results['num_tboxes'].append(len(tb_phys_dict['energy']))
        eval_results['tboxes_energies'].append(tb_phys_dict['energy'])
        eval_results['tboxes_eta'].append(tb_phys_dict['eta'])
        eval_results['tboxes_phi'].append(tb_phys_dict['phi'])
        eval_results['tboxes_eT'].append(tb_phys_dict['eT'])
        eval_results['tboxes_n_cells'].append(tb_phys_dict['n_cells'])
        eval_results['tboxes_noise'].append(tb_phys_dict['noise'])
        eval_results['tboxes_significance'].append(tb_phys_dict['significance'])
        eval_results['tboxes_neg_frac'].append(tb_phys_dict['neg_frac'])
        eval_results['tboxes_max_frac'].append(tb_phys_dict['max_frac'])

        eval_results['n_pboxes'].append(len(pees))
        eval_results['num_pboxes'].append(len(pb_phys_dict['energy']))
        eval_results['pboxes_energies'].append(pb_phys_dict['energy'])
        eval_results['pboxes_eta'].append(pb_phys_dict['eta'])
        eval_results['pboxes_phi'].append(pb_phys_dict['phi'])
        eval_results['pboxes_eT'].append(pb_phys_dict['eT'])
        eval_results['pboxes_n_cells'].append(pb_phys_dict['n_cells'])
        eval_results['pboxes_noise'].append(pb_phys_dict['noise'])
        eval_results['pboxes_significance'].append(pb_phys_dict['significance'])
        eval_results['pboxes_neg_frac'].append(pb_phys_dict['neg_frac'])
        eval_results['pboxes_max_frac'].append(pb_phys_dict['max_frac']) 
        
        # print('time: ', time.perf_counter()-start,' (s)')


    save_loc = save_folder + "/new_phys_metrics/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    print('Saving the box metrics in lists...')
    for key, value in eval_results.items():
        filename = f"{key}.pkl"
        save_object(value, save_loc+filename)

    return





if __name__=="__main__":
    folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD1_50k5_mu_15e/20231102-13/"
    save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD1_50k5_mu_15e/"

    print('Making truth box eval metrics')
    calculate_phys_metrics2(folder_to_look_in,save_at)
    print('Completed truth box eval metrics\n')














