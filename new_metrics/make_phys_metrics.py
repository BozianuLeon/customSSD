import numpy as np
import torch
import sys
import os

import h5py
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')
from utils.metrics import event_cluster_estimates
from utils.utils import wrap_check_NMS, wrap_check_truth, remove_nan
MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)


cluster_level_results = {'n_clusters':[],
                         'cluster_energies':[],
                         'cluster_etas':[],
                         'cluster_phis':[],
                         'cluster_n_cells':[],
                         'n_tboxes':[],
                         'num_tboxes':[],
                         'tbox_energies':[],
                         'tbox_eT':[],
                         'tbox_etas':[],
                         'tbox_phis':[],
                         'tbox_n_cells':[],
                         'n_pboxes':[],
                         'num_pboxes':[],
                         'pbox_energies':[],
                         'pbox_eT':[],
                         'pbox_etas':[],
                         'pbox_phis':[],
                         'pbox_n_cells':[],

                         'tbox_match_energies':[],
                         'tbox_match_eT':[],
                         'tbox_match_etas':[],
                         'tbox_match_phis':[],
                         'tbox_match_n_cells':[],
                         'pbox_match_energies':[],
                         'pbox_match_eT':[],
                         'pbox_match_etas':[],
                         'pbox_match_phis':[],
                         'pbox_match_n_cells':[],

                         'tbox_unmatch_energies':[],
                         'tbox_unmatch_eT':[],
                         'tbox_unmatch_etas':[],
                         'tbox_unmatch_phis':[],
                         'tbox_unmatch_n_cells':[],
                         'pbox_unmatch_energies':[],
                         'pbox_unmatch_eT':[],
                         'pbox_unmatch_etas':[],
                         'pbox_unmatch_phis':[],
                         'pbox_unmatch_n_cells':[],
}




def calculate_phys_metrics(
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

        tees[:,(0,2)] = (tees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
        tees[:,(1,3)] = (tees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]
        pees[:,(0,2)] = (pees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
        pees[:,(1,3)] = (pees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]

        #get the cells
        h5f = a[i]['h5file']
        try:
            h5f = h5f.decode('utf-8')
        except:
            h5f = h5f
        event_no = a[i]['event_no']


        #load cells from h5
        cells_file = "/home/users/b/bozianu/work/data/pileup50k/cells/user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        with h5py.File(cells_file,"r") as f:
            h5group = f["caloCells"]
            cells = h5group["2d"][event_no]

        clusters_file = "/home/users/b/bozianu/work/data/pileup50k/clusters/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        with h5py.File(clusters_file,"r") as f:
            cl_data = f["caloCells"] 
            event_data = cl_data["1d"][event_no]
            cluster_data = cl_data["2d"][event_no]
            cluster_data = remove_nan(cluster_data)
            cluster_data = cluster_data[cluster_data['cl_E_em']+cluster_data['cl_E_had']>5000]


        cluster_level_results['n_clusters'].append(len(cluster_data['cl_eta'].tolist()))
        cluster_level_results['cluster_energies'].append((cluster_data['cl_E_em']+cluster_data['cl_E_had']).tolist())
        cluster_level_results['cluster_etas'].append(cluster_data['cl_eta'].tolist())
        cluster_level_results['cluster_phis'].append(cluster_data['cl_phi'].tolist())
        cluster_level_results['cluster_n_cells'].append(cluster_data['cl_cell_n'].tolist())

        #total
        list_p_cl_es_tot, list_t_cl_es_tot = event_cluster_estimates(pees,scores,tees,cells,mode='total',target='energy')
        list_p_cl_eT_tot, list_t_cl_eT_tot = event_cluster_estimates(pees,scores,tees,cells,mode='total',target='eT')
        list_p_cl_etas_tot, list_t_cl_etas_tot = event_cluster_estimates(pees,scores,tees,cells,mode='total',target='eta')
        list_p_cl_phis_tot, list_t_cl_phis_tot = event_cluster_estimates(pees,scores,tees,cells,mode='total',target='phi') 
        list_p_cl_ns_tot, list_t_cl_ns_tot = event_cluster_estimates(pees,scores,tees,cells,mode='total',target='n_cells') 

        #matched
        list_p_cl_es, list_t_cl_es = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='energy')
        list_p_cl_eT, list_t_cl_eT = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='eT')
        list_p_cl_etas, list_t_cl_etas = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='eta')
        list_p_cl_phis, list_t_cl_phis = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='phi')
        list_p_cl_ns, list_t_cl_ns = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='n_cells')

        #unmatched
        list_p_cl_es_unm, list_t_cl_es_unm = event_cluster_estimates(pees,scores,tees,cells,mode='unmatch',target='energy')
        list_p_cl_eT_unm, list_t_cl_eT_unm = event_cluster_estimates(pees,scores,tees,cells,mode='unmatch',target='eT')
        list_p_cl_etas_unm, list_t_cl_etas_unm = event_cluster_estimates(pees,scores,tees,cells,mode='unmatch',target='eta')
        list_p_cl_phis_unm, list_t_cl_phis_unm = event_cluster_estimates(pees,scores,tees,cells,mode='unmatch',target='phi') 
        list_p_cl_ns_unm, list_t_cl_ns_unm = event_cluster_estimates(pees,scores,tees,cells,mode='unmatch',target='n_cells') 
        
        cluster_level_results['n_tboxes'].append(len(tees))
        cluster_level_results['num_tboxes'].append(len(list_t_cl_es_tot))
        cluster_level_results['tbox_energies'].append(list_t_cl_es_tot)
        cluster_level_results['tbox_eT'].append(list_t_cl_eT_tot)
        cluster_level_results['tbox_etas'].append(list_t_cl_etas_tot)
        cluster_level_results['tbox_phis'].append(list_t_cl_phis_tot)
        cluster_level_results['tbox_n_cells'].append(list_t_cl_ns_tot)

        cluster_level_results['tbox_match_energies'].append(list_t_cl_es)
        cluster_level_results['tbox_match_eT'].append(list_t_cl_eT)
        cluster_level_results['tbox_match_etas'].append(list_t_cl_etas)
        cluster_level_results['tbox_match_phis'].append(list_t_cl_phis)
        cluster_level_results['tbox_match_n_cells'].append(list_t_cl_ns)

        cluster_level_results['tbox_unmatch_energies'].append(list_t_cl_es_unm)
        cluster_level_results['tbox_unmatch_eT'].append(list_t_cl_eT_unm)
        cluster_level_results['tbox_unmatch_etas'].append(list_t_cl_etas_unm)
        cluster_level_results['tbox_unmatch_phis'].append(list_t_cl_phis_unm)
        cluster_level_results['tbox_unmatch_n_cells'].append(list_t_cl_ns_unm)
        
        cluster_level_results['n_pboxes'].append(len(pees))
        cluster_level_results['num_pboxes'].append(len(list_p_cl_es_tot))
        cluster_level_results['pbox_energies'].append(list_p_cl_es_tot)
        cluster_level_results['pbox_eT'].append(list_p_cl_eT_tot)
        cluster_level_results['pbox_etas'].append(list_p_cl_etas_tot)
        cluster_level_results['pbox_phis'].append(list_p_cl_phis_tot)
        cluster_level_results['pbox_n_cells'].append(list_p_cl_ns_tot)

        cluster_level_results['pbox_match_energies'].append(list_p_cl_es)
        cluster_level_results['pbox_match_eT'].append(list_p_cl_eT)
        cluster_level_results['pbox_match_etas'].append(list_p_cl_etas)
        cluster_level_results['pbox_match_phis'].append(list_p_cl_phis)
        cluster_level_results['pbox_match_n_cells'].append(list_p_cl_ns)

        cluster_level_results['pbox_unmatch_energies'].append(list_p_cl_es_unm)
        cluster_level_results['pbox_unmatch_eT'].append(list_p_cl_eT_unm)
        cluster_level_results['pbox_unmatch_etas'].append(list_p_cl_etas_unm)
        cluster_level_results['pbox_unmatch_phis'].append(list_p_cl_phis_unm)
        cluster_level_results['pbox_unmatch_n_cells'].append(list_p_cl_ns_unm)


    save_loc = save_folder + "/phys_metrics/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    print('Saving the box metrics in lists...')
    #automate this saving!
    save_object(cluster_level_results['n_clusters'],save_loc+'n_clusters.pkl')
    save_object(cluster_level_results['cluster_energies'], save_loc+'cluster_energies.pkl')
    save_object(cluster_level_results['cluster_etas'], save_loc+'cluster_etas.pkl')
    save_object(cluster_level_results['cluster_phis'],save_loc+'cluster_phis.pkl')
    save_object(cluster_level_results['cluster_n_cells'],save_loc+'cluster_n_cells.pkl')
    
    save_object(cluster_level_results['n_tboxes'],save_loc+'n_tboxes.pkl')
    save_object(cluster_level_results['num_tboxes'],save_loc+'num_tboxes.pkl')
    save_object(cluster_level_results['tbox_energies'],save_loc+'tbox_energies.pkl')
    save_object(cluster_level_results['tbox_eT'],save_loc+'tbox_eT.pkl')
    save_object(cluster_level_results['tbox_etas'],save_loc+'tbox_etas.pkl')
    save_object(cluster_level_results['tbox_phis'],save_loc+'tbox_phis.pkl')
    save_object(cluster_level_results['tbox_n_cells'],save_loc+'tbox_n_cells.pkl')
    
    save_object(cluster_level_results['n_pboxes'],save_loc+'n_pboxes.pkl')
    save_object(cluster_level_results['num_pboxes'],save_loc+'num_pboxes.pkl')
    save_object(cluster_level_results['pbox_energies'],save_loc+'pbox_energies.pkl')
    save_object(cluster_level_results['pbox_eT'],save_loc+'pbox_eT.pkl')
    save_object(cluster_level_results['pbox_etas'],save_loc+'pbox_etas.pkl')
    save_object(cluster_level_results['pbox_phis'],save_loc+'pbox_phis.pkl')
    save_object(cluster_level_results['pbox_n_cells'],save_loc+'pbox_n_cells.pkl')

    save_object(cluster_level_results['tbox_match_energies'],save_loc+'tbox_match_energies.pkl')
    save_object(cluster_level_results['tbox_match_eT'],save_loc+'tbox_match_eT.pkl')
    save_object(cluster_level_results['tbox_match_etas'],save_loc+'tbox_match_etas.pkl')
    save_object(cluster_level_results['tbox_match_phis'],save_loc+'tbox_match_phis.pkl')
    save_object(cluster_level_results['tbox_match_n_cells'],save_loc+'tbox_match_n_cells.pkl')
    save_object(cluster_level_results['pbox_match_energies'],save_loc+'pbox_match_energies.pkl')
    save_object(cluster_level_results['pbox_match_eT'],save_loc+'pbox_match_eT.pkl')
    save_object(cluster_level_results['pbox_match_etas'],save_loc+'pbox_match_etas.pkl')
    save_object(cluster_level_results['pbox_match_phis'],save_loc+'pbox_match_phis.pkl')
    save_object(cluster_level_results['pbox_match_n_cells'],save_loc+'pbox_match_n_cells.pkl')

    save_object(cluster_level_results['tbox_unmatch_energies'],save_loc+'tbox_unmatch_energies.pkl')
    save_object(cluster_level_results['tbox_unmatch_eT'],save_loc+'tbox_unmatch_eT.pkl')
    save_object(cluster_level_results['tbox_unmatch_etas'],save_loc+'tbox_unmatch_etas.pkl')
    save_object(cluster_level_results['tbox_unmatch_phis'],save_loc+'tbox_unmatch_phis.pkl')
    save_object(cluster_level_results['tbox_unmatch_n_cells'],save_loc+'tbox_unmatch_n_cells.pkl')
    save_object(cluster_level_results['pbox_unmatch_energies'],save_loc+'pbox_unmatch_energies.pkl')
    save_object(cluster_level_results['pbox_unmatch_eT'],save_loc+'pbox_unmatch_eT.pkl')
    save_object(cluster_level_results['pbox_unmatch_etas'],save_loc+'pbox_unmatch_etas.pkl')
    save_object(cluster_level_results['pbox_unmatch_phis'],save_loc+'pbox_unmatch_phis.pkl')
    save_object(cluster_level_results['pbox_unmatch_n_cells'],save_loc+'pbox_unmatch_n_cells.pkl')


    return






if __name__=="__main__":

    folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD_50k5_mu_20e/20231005-12/"
    save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD_50k5_mu_20e/"

    print('Making phys metrics')
    calculate_phys_metrics(folder_to_look_in,save_at)
    print('Completed [phys] metrics\n')














