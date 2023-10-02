print('Starting physics metrics')
import numpy as np
from scipy import stats
import matplotlib
from matplotlib import pyplot as plt
from itertools import compress
import torch
import torchvision
import os
import sys

import h5py
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')
from utils.metrics import delta_n, n_unmatched_truth, n_unmatched_preds, centre_diffs, hw_diffs, area_covered
from utils.utils import wrap_check_NMS, wrap_check_truth, remove_nan, get_cells_from_boxes



########################################################################################################
#load inference from .npy
# save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp3_SSD_model_15_real/20230526-05/"
save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD_model_25_large_mu/20230908-12/"
path_to_structured_array = save_loc + "/struc_array.npy"

with open(path_to_structured_array, 'rb') as f:
    a = np.load(f)

print('Length of inference array',len(a))



def event_cluster_estimates(pred_boxes, scores, truth_boxes, cells, mode='match',target='energy'):
    #arguments are:
    #pred_boxes, model output, no augmentation/wrap checking [n_preds,4] NO WRAP CHECK
    #truth_boxes, from .npy file, [n_objs, 4]
    #mode, tells us whether we should look at matched predictions, unmatched, or all (None) ['match','unmatch']
    #target, cluster statistic to check ['energy','eta','phi']

    wc_pred_boxes = wrap_check_NMS(pred_boxes,scores,min(cells['cell_phi']),max(cells['cell_phi']),threshold=0.2)
    wc_truth_boxes = wrap_check_truth(truth_boxes,min(cells['cell_phi']),max(cells['cell_phi']))

    if mode=='match':
        iou_mat = torchvision.ops.boxes.box_iou(torch.tensor(wc_truth_boxes),torch.tensor(wc_pred_boxes))
        matched_vals, matches = iou_mat.max(dim=0)
        wc_truth_boxes = wc_truth_boxes[matches[np.nonzero(matched_vals)]].reshape(-1,4)
        wc_pred_boxes = wc_pred_boxes[np.nonzero(matched_vals)].reshape(-1,4)
    elif mode=='unmatch':
        iou_mat = torchvision.ops.boxes.box_iou(torch.tensor(wc_truth_boxes),torch.tensor(wc_pred_boxes))
        matched_vals, matches = iou_mat.max(dim=0)
        unmatched_idxs = np.where(matched_vals==0)
        wc_pred_boxes = wc_pred_boxes[unmatched_idxs].reshape(-1,4)
        wc_truth_boxes = np.delete(wc_truth_boxes,matches[np.nonzero(matched_vals)],axis=0)

    list_pred_cl_cells = get_cells_from_boxes(wc_pred_boxes,cells)
    list_tru_cl_cells = get_cells_from_boxes(wc_truth_boxes,cells)
    for data in list_pred_cl_cells:
        if sum(data['cell_BadCells']) < 0:
            print(data)
    
    # Check that neither list using placeholder values has an entry with no cells
    pred_zero_cells_mask = [sum(data['cell_BadCells']) >= 0 for data in list_pred_cl_cells]
    list_pred_cl_cells = list(compress(list_pred_cl_cells, pred_zero_cells_mask))
    list_tru_cl_cells = list(compress(list_tru_cl_cells, pred_zero_cells_mask))
    true_zero_cells_mask = [sum(data['cell_BadCells']) >= 0 for data in list_tru_cl_cells]
    list_pred_cl_cells = list(compress(list_pred_cl_cells, true_zero_cells_mask))
    list_tru_cl_cells = list(compress(list_tru_cl_cells, true_zero_cells_mask))
    
    if target == 'energy':
        list_pred_cl_energies = [sum(x['cell_E']) for x in list_pred_cl_cells]
        list_tru_cl_energies = [sum(x['cell_E']) for x in list_tru_cl_cells]
        return list_pred_cl_energies, list_tru_cl_energies

    def calc_cl_eta(cl_array):
        return np.dot(cl_array['cell_eta'],np.abs(cl_array['cell_E'])) / sum(np.abs(cl_array['cell_E']))
    def calc_cl_phi(cl_array):
        return np.dot(cl_array['cell_phi'],np.abs(cl_array['cell_E'])) / sum(np.abs(cl_array['cell_E']))

    if target  == 'eta':
        list_pred_cl_etas = [calc_cl_eta(x) for x in list_pred_cl_cells]
        list_tru_cl_etas = [calc_cl_eta(x) for x in list_tru_cl_cells]
        return list_pred_cl_etas, list_tru_cl_etas
    
    if target == 'phi':
        list_pred_cl_phis = [calc_cl_phi(x) for x in list_pred_cl_cells]
        list_tru_cl_phis = [calc_cl_phi(x) for x in list_tru_cl_cells]
        return list_pred_cl_phis, list_tru_cl_phis
    
    if target == 'n_cells':
        list_pred_cl_ns = [len(x) for x in list_pred_cl_cells]
        list_tru_cl_ns = [len(x) for x in list_tru_cl_cells]
        return list_pred_cl_ns, list_tru_cl_ns  






total_clus_energy, total_clus_eta, total_clus_phi, total_clus_n = list(), list(), list(), list()

total_match_energy_ratios, total_match_eta_diff, total_match_phi_diff, total_match_n_diff = list(), list(), list(), list()
total_match_pred_energy, total_match_pred_eta, total_match_pred_phi, total_match_pred_n = list(), list(), list(), list()
total_unmatch_pred_energy, total_unmatch_pred_eta, total_unmatch_pred_phi, total_unmatch_pred_n = list(), list(), list(), list()
total_pred_energy, total_pred_eta, total_pred_phi, total_pred_n = list(), list(), list(), list()

total_match_tru_energy, total_match_tru_eta, total_match_tru_phi, total_match_tru_n = list(), list(), list(), list()
total_unmatch_tru_energy, total_unmatch_tru_eta, total_unmatch_tru_phi, total_unmatch_tru_n = list(), list(), list(), list()
total_tru_energy, total_tru_eta, total_tru_phi, total_tru_n = list(), list(), list(), list()
for i in range(len(a)):
    extent_i = a[i]['extent']
    preds = a[i]['p_boxes']
    trues = a[i]['t_boxes']
    scores = a[i]['p_scores']
    #get region predictions and truths
    pees = preds[np.where(preds[:,0] > 0)]
    tees = trues[np.where(trues[:,0] > 0)]
    pees = torch.tensor(pees)
    tees = torch.tensor(tees)

    #make boxes cover extent
    tees[:,(0,2)] = (tees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
    tees[:,(1,3)] = (tees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]
    pees[:,(0,2)] = (pees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
    pees[:,(1,3)] = (pees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]

    #get the cells
    h5f = a[i]['h5file']
    event_no = a[i]['event_no']
    # print(h5f,type(h5f),str(h5f),h5f.decode('utf-8'))

    #load cells from h5
    # file = "/home/users/b/bozianu/work/data/real/cells/user.cantel.33075755._00000{}.calocellD3PD_mc16_JZW4.r14423.h5".format(h5f)
    file = "/home/users/b/bozianu/work/data/pileup50k/cells/user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
    with h5py.File(file,"r") as f:
        h5group = f["caloCells"]
        cells = h5group["2d"][event_no]

    # clusters_file = "/home/users/b/bozianu/work/data/real/topo/user.cantel.33075755._00000{}.topoclusterD3PD_mc16_JZW4.r14423.h5".format(h5f)
    clusters_file = "/home/users/b/bozianu/work/data/pileup50k/clusters/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f.decode('utf-8'))
    with h5py.File(clusters_file,"r") as f:
        cl_data = f["caloCells"] 
        event_data = cl_data["1d"][event_no]
        cluster_data = cl_data["2d"][event_no]
        cluster_data = remove_nan(cluster_data)
        cluster_data = cluster_data[cluster_data['cl_E_em']+cluster_data['cl_E_had']>5000]

    total_clus_energy.append((cluster_data['cl_E_em']+cluster_data['cl_E_had']).tolist())
    total_clus_eta.append(cluster_data['cl_eta'].tolist())
    total_clus_phi.append(cluster_data['cl_phi'].tolist())
    total_clus_n.append(cluster_data['cl_cell_n'].tolist())

    print(i)
    #matched
    list_p_cl_es, list_t_cl_es = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='energy')
    list_p_cl_etas, list_t_cl_etas = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='eta')
    list_p_cl_phis, list_t_cl_phis = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='phi')
    list_p_cl_ns, list_t_cl_ns = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='n_cells')


    #matches
    print(len(list_t_cl_es),len(list_p_cl_es))
    print(np.array(list_t_cl_es).shape,np.array(list_p_cl_es).shape)
    total_match_energy_ratios.append(np.array(list_p_cl_es) / np.array(list_t_cl_es))
    if np.any(np.array(list_p_cl_es) / np.array(list_t_cl_es) > 50):
        print('GREATER THAN FIFTY:')
        print('h5f',h5f,'event_no',event_no)

    total_match_eta_diff.append(np.array(list_p_cl_etas) - np.array(list_t_cl_etas))
    if np.any(np.array(list_p_cl_etas) - np.array(list_t_cl_etas) > 1.6):
        print('ETA difference GREATER THAN 1.6:')
        print('h5f',h5f,'event_no',event_no)
    total_match_phi_diff.append(np.array(list_p_cl_phis) - np.array(list_t_cl_phis))
    if np.any(np.array(list_p_cl_phis) - np.array(list_t_cl_phis) < -1.9):
        print('PHI Difference LESS THAN -1.9:')
        print('h5f',h5f,'event_no',event_no)
    total_match_n_diff.append(np.array(list_p_cl_ns) - np.array(list_t_cl_ns))
    total_match_pred_energy.append(list_p_cl_es)
    total_match_tru_energy.append(list_t_cl_es)
    total_match_pred_eta.append(list_p_cl_etas)
    total_match_tru_eta.append(list_t_cl_etas)
    total_match_pred_phi.append(list_p_cl_phis)
    if np.any(np.array(list_p_cl_phis)  < -6):
        print('PHI Difference LESS THAN -1.9:')
        print('h5f',h5f,'event_no',event_no)
        print(list_p_cl_phis)
        quit()
    total_match_tru_phi.append(list_t_cl_phis)
    total_match_pred_n.append(list_p_cl_ns)
    total_match_tru_n.append(list_t_cl_ns)
    

    #unmatched
    list_p_cl_es2, list_t_cl_es2 = event_cluster_estimates(pees,scores,tees,cells,mode='unmatch',target='energy')
    list_p_cl_etas2, list_t_cl_etas2 = event_cluster_estimates(pees,scores,tees,cells,mode='unmatch',target='eta')
    list_p_cl_phis2, list_t_cl_phis2 = event_cluster_estimates(pees,scores,tees,cells,mode='unmatch',target='phi') 
    list_p_cl_ns2, list_t_cl_ns2 = event_cluster_estimates(pees,scores,tees,cells,mode='unmatch',target='n_cells') 
    
    total_unmatch_pred_energy.append(list_p_cl_es2)
    total_unmatch_tru_energy.append(list_t_cl_es2)
    total_unmatch_pred_eta.append(list_p_cl_etas2)
    total_unmatch_tru_eta.append(list_t_cl_etas2)
    total_unmatch_pred_phi.append(list_p_cl_phis2)
    total_unmatch_tru_phi.append(list_t_cl_phis2)
    total_unmatch_pred_n.append(list_p_cl_ns2)
    total_unmatch_tru_n.append(list_t_cl_ns2)

    #total
    list_p_cl_es3, list_t_cl_es3 = event_cluster_estimates(pees,scores,tees,cells,mode='total',target='energy')
    list_p_cl_etas3, list_t_cl_etas3 = event_cluster_estimates(pees,scores,tees,cells,mode='total',target='eta')
    list_p_cl_phis3, list_t_cl_phis3 = event_cluster_estimates(pees,scores,tees,cells,mode='total',target='phi') 
    list_p_cl_ns3, list_t_cl_ns3 = event_cluster_estimates(pees,scores,tees,cells,mode='total',target='n_cells') 

    total_pred_energy.append(list_p_cl_es3)
    total_tru_energy.append(list_t_cl_es3)
    total_pred_eta.append(list_p_cl_etas3)
    total_tru_eta.append(list_t_cl_etas3)
    total_pred_phi.append(list_p_cl_phis3)
    total_tru_phi.append(list_t_cl_phis3)
    total_pred_n.append(list_t_cl_ns3)
    total_tru_n.append(list_t_cl_ns3)



def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)




print('Saving the physics metrics in lists...')
#ground truth clusters
save_object(total_clus_energy,save_loc+'total_clus_energy.pkl')
save_object(total_clus_eta, save_loc+'total_clus_eta.pkl')
save_object(total_clus_phi, save_loc+'total_clus_phi.pkl')
save_object(total_clus_n, save_loc+'total_clus_n.pkl')

#matched
save_object(total_match_energy_ratios,save_loc+'total_match_energy_ratios.pkl')
save_object(total_match_eta_diff, save_loc+'total_match_eta_diff.pkl')
save_object(total_match_phi_diff, save_loc+'total_match_phi_diff.pkl')
save_object(total_match_n_diff, save_loc+'total_match_n_diff.pkl')

save_object(total_match_pred_energy,save_loc+'total_match_pred_energy.pkl')
save_object(total_match_tru_energy,save_loc+'total_match_tru_energy.pkl')
save_object(total_match_pred_eta,save_loc+'total_match_pred_eta.pkl')
save_object(total_match_tru_eta,save_loc+'total_match_tru_eta.pkl')
save_object(total_match_pred_phi,save_loc+'total_match_pred_phi.pkl')
save_object(total_match_tru_phi,save_loc+'total_match_tru_phi.pkl')
save_object(total_match_pred_n,save_loc+'total_match_pred_n.pkl')
save_object(total_match_tru_n,save_loc+'total_match_tru_n.pkl')

#unmatched
save_object(total_unmatch_pred_energy,save_loc+'total_unmatch_pred_energy.pkl')
save_object(total_unmatch_tru_energy,save_loc+'total_unmatch_tru_energy.pkl')
save_object(total_unmatch_pred_eta,save_loc+'total_unmatch_pred_eta.pkl')
save_object(total_unmatch_tru_eta,save_loc+'total_unmatch_tru_eta.pkl')
save_object(total_unmatch_pred_phi,save_loc+'total_unmatch_pred_phi.pkl')
save_object(total_unmatch_tru_phi,save_loc+'total_unmatch_tru_phi.pkl')
save_object(total_unmatch_pred_n,save_loc+'total_unmatch_pred_n.pkl')
save_object(total_unmatch_tru_n,save_loc+'total_unmatch_tru_n.pkl')

#total
save_object(total_pred_energy,save_loc+'total_pred_energy.pkl')
save_object(total_tru_energy,save_loc+'total_tru_energy.pkl')
save_object(total_pred_eta,save_loc+'total_pred_eta.pkl')
save_object(total_tru_eta,save_loc+'total_tru_eta.pkl')
save_object(total_pred_phi,save_loc+'total_pred_phi.pkl')
save_object(total_tru_phi,save_loc+'total_tru_phi.pkl')
save_object(total_pred_n,save_loc+'total_pred_n.pkl')
save_object(total_tru_n,save_loc+'total_tru_n.pkl')



