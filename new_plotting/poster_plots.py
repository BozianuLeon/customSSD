import numpy as np
import torch
import torchvision
import sys
import os
import time
import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep
hep.style.use(hep.style.ATLAS)

import h5py
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
# caution: path[0] is reserved for script path 
sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')

from utils.utils import matched_boxes, unmatched_boxes, wrap_check_NMS, wrap_check_truth, remove_nan
from utils.metrics import RetrieveCellIdsFromBox, RetrieveCellIdsFromCluster
from utils.metrics import get_physics_dictionary


MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)


def CalculateMatchFromBox(box_in_question,boxes_array,desired_cells):
    if desired_cells is None:  
        print('It happened! None -> np.nan')  
        return np.nan
    else:
        box_in_question = torch.tensor(box_in_question).unsqueeze(0)
        boxes_tensor = torch.tensor(boxes_array)
        iou_mat = torchvision.ops.boxes.box_iou(box_in_question, boxes_tensor)
        matched_vals, matches = iou_mat.max(dim=1)
    return int(matched_vals>0.05)

def CalculateMatchesFromBoxes(boxes_array1,boxes_array2):
    #calculate the number of predicted boxes lie on top of a GT box
    boxes_tensor1 = torch.tensor(boxes_array1)
    boxes_tensor2 = torch.tensor(boxes_array2)
    iou_mat = torchvision.ops.boxes.box_iou(boxes_tensor1, boxes_tensor2)
    matched_vals, matches = iou_mat.max(dim=1)

    return np.array(matched_vals>0.05,dtype=np.float32)
    # matched_q = np.zeros_like(matched_vals)
    # matched_q[matched_vals>0.4]=1.0
    # return matched_q


necessary_results = {
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
    'tboxes_energies_2sig': [],
    'tboxes_eT_2sig':       [], 
    'tboxes_n_cells_2sig':  [],  
    'tboxes_matched': [],
    
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
    'pboxes_energies_2sig': [],
    'pboxes_eT_2sig':       [], 
    'pboxes_n_cells_2sig':  [],  
    'pboxes_matched': [],

}

def make_poster_plot_data(
    folder_containing_struc_array,
    save_folder,
):

    with open(folder_containing_struc_array + "/struc_array.npy", 'rb') as f:
        a = np.load(f)

    for i in range(len(a)):
        start = time.perf_counter()
        print(i)
        extent_i = a[i]['extent']
        preds = a[i]['p_boxes']
        trues = a[i]['t_boxes']
        scores = a[i]['p_scores']

        pees = preds[np.where(preds[:,0] > 0)]
        tees = trues[np.where(trues[:,0] > 0)]

        # f,ax = plt.subplots(1,1)
        # for bbx in tees:
        #     x,y=300*float(bbx[0]),300*float(bbx[1])
        #     w,h=300*float(bbx[2])-300*float(bbx[0]),300*float(bbx[3])-300*float(bbx[1])  
        #     bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='limegreen',fc='none')
        #     ax.add_patch(bb)
        # for bbx in pees:
        #     x,y=300*float(bbx[0]),300*float(bbx[1])
        #     w,h=300*float(bbx[2])-300*float(bbx[0]),300*float(bbx[3])-300*float(bbx[1])  
        #     bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='red',fc='none')
        #     ax.add_patch(bb)
        # ax.set(xlim=(0,300),ylim=(0,300))
        # f.savefig('inference2.png')
        # plt.close()
        # print(pees)
        # print(extent_i)
        # quit()

        #make boxes cover extent
        tees[:,(0,2)] = (tees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
        tees[:,(1,3)] = (tees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]
        pees[:,(0,2)] = (pees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
        pees[:,(1,3)] = (pees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]

        #wrap check boxes here
        pees = wrap_check_NMS(pees,scores,MIN_CELLS_PHI,MAX_CELLS_PHI,threshold=0.2)
        tees = wrap_check_truth(tees,MIN_CELLS_PHI,MAX_CELLS_PHI)
        
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

        # clusters_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/clusters/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        # with h5py.File(clusters_file,"r") as f:
        #     cl_data = f["caloCells"] 
        #     event_data = cl_data["1d"][event_no]
        #     cluster_data = cl_data["2d"][event_no]
        #     raw_E_mask = (cluster_data['cl_E_em']+cluster_data['cl_E_had']) > 5000 #5GeV cut
        #     cluster_data = cluster_data[raw_E_mask]
        #     # cluster_cell_data = cl_data["3d"][event_no]
        #     # cluster_cell_data = cluster_cell_data[raw_E_mask]


        l_pred_cells = RetrieveCellIdsFromBox(cells,pees)
        l_true_cells = RetrieveCellIdsFromBox(cells,tees)

        tb_phys_dict = get_physics_dictionary(l_true_cells,cells)
        pb_phys_dict = get_physics_dictionary(l_pred_cells,cells)

        are_pboxes_matched = [CalculateMatchFromBox(pb,tees,des_cells) for pb,des_cells in zip(pees,l_pred_cells)]
        are_tboxes_matched = [CalculateMatchFromBox(tb,pees,des_cells) for tb,des_cells in zip(tees,l_true_cells)]

        # check if tbox tb is "matched" to any box in pees
        necessary_results['tboxes_matched'].append(are_tboxes_matched)
        # check if pbox pb is "matched" to any box in tees
        necessary_results['pboxes_matched'].append(are_pboxes_matched)


        necessary_results['n_tboxes'].append(len(tees))
        necessary_results['num_tboxes'].append(len(tb_phys_dict['energy']))
        necessary_results['tboxes_energies'].append(tb_phys_dict['energy'])
        necessary_results['tboxes_eta'].append(tb_phys_dict['eta'])
        necessary_results['tboxes_phi'].append(tb_phys_dict['phi'])
        necessary_results['tboxes_eT'].append(tb_phys_dict['eT'])
        necessary_results['tboxes_n_cells'].append(tb_phys_dict['n_cells'])
        necessary_results['tboxes_noise'].append(tb_phys_dict['noise'])
        necessary_results['tboxes_significance'].append(tb_phys_dict['significance'])
        necessary_results['tboxes_neg_frac'].append(tb_phys_dict['neg_frac'])
        necessary_results['tboxes_max_frac'].append(tb_phys_dict['max_frac'])

        necessary_results['tboxes_energies_2sig'].append(tb_phys_dict['energy2sig'])
        necessary_results['tboxes_eT_2sig'].append(tb_phys_dict['eT2sig'])
        necessary_results['tboxes_n_cells_2sig'].append(tb_phys_dict['n_cells2sig'])

        necessary_results['n_pboxes'].append(len(pees))
        necessary_results['num_pboxes'].append(len(pb_phys_dict['energy']))
        necessary_results['pboxes_energies'].append(pb_phys_dict['energy'])
        necessary_results['pboxes_eta'].append(pb_phys_dict['eta'])
        necessary_results['pboxes_phi'].append(pb_phys_dict['phi'])
        necessary_results['pboxes_eT'].append(pb_phys_dict['eT'])
        necessary_results['pboxes_n_cells'].append(pb_phys_dict['n_cells'])
        necessary_results['pboxes_noise'].append(pb_phys_dict['noise'])
        necessary_results['pboxes_significance'].append(pb_phys_dict['significance'])
        necessary_results['pboxes_neg_frac'].append(pb_phys_dict['neg_frac'])
        necessary_results['pboxes_max_frac'].append(pb_phys_dict['max_frac']) 

        necessary_results['pboxes_energies_2sig'].append(pb_phys_dict['energy2sig'])
        necessary_results['pboxes_eT_2sig'].append(pb_phys_dict['eT2sig'])
        necessary_results['pboxes_n_cells_2sig'].append(pb_phys_dict['n_cells2sig'])

    save_loc = save_folder + f"/new_phys_metrics/poster25/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    print('Saving the box metrics in lists...')
    for key, value in necessary_results.items():
        filename = f"{key}.pkl"
        save_object(value, save_loc+filename)

    return None

if __name__=="__main__":
    folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD1_50k5_mu_15e/20231102-13/"
    save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD1_50k5_mu_15e/"

    print('Making poster plot info')
    make_poster_plot_data(folder_to_look_in,save_at)
    print('Completed poster plot info\n')







