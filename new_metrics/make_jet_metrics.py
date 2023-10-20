import numpy as np
import torch
import sys
import os

import h5py
import awkward as ak
import fastjet
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')
from utils.utils import wrap_check_NMS, wrap_check_truth, remove_nan, transform_angle
from utils.utils import event_cluster_estimates, get_cells_from_boxes
MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)


jet_level_results = {'n_esdjets':[],
                    'esdjet_energies':[],
                    'esdjet_etas':[],
                    'esdjet_phis':[],
                    'esdjet_pts':[],

                    'n_fjets':[],
                    'fjet_energies':[],
                    'fjet_etas':[],
                    'fjet_phis':[],
                    'fjet_pts':[],

                    'n_tboxjets':[],
                    'tboxjet_energies':[],
                    'tboxjet_etas':[],
                    'tboxjet_phis':[],
                    'tboxjet_pts':[],

                    'n_pboxjets':[],
                    'pboxjet_energies':[],
                    'pboxjet_etas':[],
                    'pboxjet_phis':[],
                    'pboxjet_pts':[],    
}



def calculate_jet_metrics(
    folder_containing_struc_array,
    save_folder,
):

    with open(folder_containing_struc_array + "/struc_array.npy", 'rb') as f:
        a = np.load(f)
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)


    for i in range(len(a)):
        # print(i)
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
        # cells_file = "/home/users/b/bozianu/work/data/pileup50k/cells/user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        cells_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/cells/user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        with h5py.File(cells_file,"r") as f:
            h5group = f["caloCells"]
            cells = h5group["2d"][event_no]

        # clusters_file = "/home/users/b/bozianu/work/data/pileup50k/clusters/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        clusters_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/clusters/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        with h5py.File(clusters_file,"r") as f:
            cl_data = f["caloCells"] 
            event_data = cl_data["1d"][event_no]
            cluster_data = cl_data["2d"][event_no]
            cluster_data = remove_nan(cluster_data)
            cluster_data = cluster_data[cluster_data['cl_E_em']+cluster_data['cl_E_had']>5000]

        # jets_file = "/home/users/b/bozianu/work/data/pileup50k/jets/user.cantel.34126190._0000{}.jetD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        jets_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/jets/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        with h5py.File(jets_file,"r") as f:
            j_data = f["caloCells"]
            jet_data = j_data["2d"][event_no]
            jet_data = remove_nan(jet_data)
        

        #all TC's (greater than 5GeV)
        ESD_inputs = []
        m = 0.0 #topoclusters have 0 mass
        for i in range(len(cluster_data)):
            cl_px = float(cluster_data[i]['cl_pt'] * np.cos(cluster_data[i]['cl_phi']))
            cl_py = float(cluster_data[i]['cl_pt'] * np.sin(cluster_data[i]['cl_phi']))
            cl_pz = float(cluster_data[i]['cl_pt'] * np.sinh(cluster_data[i]['cl_eta']))
            ESD_inputs.append(fastjet.PseudoJet(cl_px,cl_py,cl_pz,m))

        #model prediction and truth box jets
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
        for k in range(len(list_p_cl_es)):
            pred_box_eta = list_p_cl_etas[k]
            pred_box_phi = list_p_cl_phis[k]
            pred_box_theta = 2*np.arctan(np.exp(-pred_box_eta))
            pred_box_e = list_p_cl_es[k]
            pred_box_inputs.append(fastjet.PseudoJet(pred_box_e*np.sin(pred_box_theta)*np.cos(pred_box_phi),
                                                    pred_box_e*np.sin(pred_box_theta)*np.sin(pred_box_phi),
                                                    pred_box_e*np.cos(pred_box_theta),
                                                    m))

        #esd jets
        batch_energy_esdjets, batch_pt_esdjets, batch_eta_esdjets, batch_phi_esdjets, batch_m_esdjets = list(),list(),list(),list(), list()
        for oj in range(len(jet_data)):
            offjet = jet_data[oj]
            batch_energy_esdjets.append(offjet['AntiKt4EMTopoJets_E'])
            batch_pt_esdjets.append(offjet['AntiKt4EMTopoJets_JetConstitScaleMomentum_pt'])
            batch_eta_esdjets.append(offjet['AntiKt4EMTopoJets_JetConstitScaleMomentum_eta'])
            batch_phi_esdjets.append(offjet['AntiKt4EMTopoJets_JetConstitScaleMomentum_phi'])

        #store results
        jet_level_results['n_esdjets'].append(len(jet_data))
        jet_level_results['esdjet_energies'].append(batch_energy_esdjets)
        jet_level_results['esdjet_pts'].append(batch_pt_esdjets)
        jet_level_results['esdjet_etas'].append(batch_eta_esdjets)
        jet_level_results['esdjet_phis'].append(batch_phi_esdjets)


        #ALL TopoClusters:
        ESD_cluster_jets = fastjet.ClusterSequence(ESD_inputs, jetdef)
        ESD_cluster_inc_jets = ESD_cluster_jets.inclusive_jets()
        batch_energy_fjets, batch_pt_fjets, batch_eta_fjets, batch_phi_fjets, batch_m_fjets = list(),list(),list(),list(),list()
        for i in range(len(ESD_cluster_inc_jets)):
            ii = ESD_cluster_inc_jets[i]
            batch_energy_fjets.append(ii.E())
            batch_pt_fjets.append(ii.pt())
            batch_eta_fjets.append(ii.eta())
            batch_phi_fjets.append(ii.phi())
        
        jet_level_results['n_fjets'].append(len(ESD_cluster_inc_jets))
        jet_level_results['fjet_energies'].append(batch_energy_fjets)
        jet_level_results['fjet_pts'].append(batch_pt_fjets)
        jet_level_results['fjet_etas'].append(batch_eta_fjets)
        jet_level_results['fjet_phis'].append(batch_phi_fjets)

        #Truth box clusters
        truth_box_jets = fastjet.ClusterSequence(truth_box_inputs,jetdef)
        truth_box_inc_jets = truth_box_jets.inclusive_jets()
        batch_energy_tboxjets, batch_pt_tboxjets, batch_eta_tboxjets, batch_phi_tboxjets,batch_m_tboxjets = list(),list(),list(),list(),list()
        for j in range(len(truth_box_inc_jets)):
            jj = truth_box_inc_jets[j]
            batch_energy_tboxjets.append(jj.E())
            batch_pt_tboxjets.append(jj.pt())
            batch_eta_tboxjets.append(jj.eta())
            batch_phi_tboxjets.append(jj.phi())

        jet_level_results['n_tboxjets'].append(len(truth_box_inc_jets))
        jet_level_results['tboxjet_energies'].append(batch_energy_tboxjets)
        jet_level_results['tboxjet_pts'].append(batch_pt_tboxjets)
        jet_level_results['tboxjet_etas'].append(batch_eta_tboxjets)
        jet_level_results['tboxjet_phis'].append(batch_phi_tboxjets)

        #Pred box clusters
        pred_box_jets = fastjet.ClusterSequence(pred_box_inputs,jetdef)
        pred_box_inc_jets = pred_box_jets.inclusive_jets()
        batch_energy_pboxjets, batch_pt_pboxjets, batch_eta_pboxjets, batch_phi_pboxjets,batch_m_pboxjets = list(),list(),list(),list(),list()
        for k in range(len(pred_box_inc_jets)):
            kk = pred_box_inc_jets[k]
            batch_energy_pboxjets.append(kk.E())
            batch_pt_pboxjets.append(kk.pt())
            batch_eta_pboxjets.append(kk.eta())
            batch_phi_pboxjets.append(kk.phi())
            batch_m_pboxjets.append(kk.m())        

        jet_level_results['n_pboxjets'].append(len(pred_box_inc_jets))
        jet_level_results['pboxjet_energies'].append(batch_energy_pboxjets)
        jet_level_results['pboxjet_pts'].append(batch_pt_pboxjets)
        jet_level_results['pboxjet_etas'].append(batch_eta_pboxjets)
        jet_level_results['pboxjet_phis'].append(batch_phi_pboxjets)


    save_loc = save_folder + "/jet_metrics/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    print('Saving the jet metrics in lists...')
    #automate this saving!
    save_object(jet_level_results['n_esdjets'],save_loc+'n_esdjets.pkl')
    save_object(jet_level_results['esdjet_energies'],save_loc+'esdjet_energies.pkl')
    save_object(jet_level_results['esdjet_etas'],save_loc+'esdjet_etas.pkl')
    save_object(jet_level_results['esdjet_phis'],save_loc+'esdjet_phis.pkl')
    save_object(jet_level_results['esdjet_pts'],save_loc+'esdjet_pts.pkl')
     
    save_object(jet_level_results['n_fjets'],save_loc+'n_fjets.pkl')
    save_object(jet_level_results['fjet_energies'],save_loc+'fjet_energies.pkl')
    save_object(jet_level_results['fjet_etas'],save_loc+'fjet_etas.pkl')
    save_object(jet_level_results['fjet_phis'],save_loc+'fjet_phis.pkl')
    save_object(jet_level_results['fjet_pts'],save_loc+'fjet_pts.pkl')

    save_object(jet_level_results['n_tboxjets'],save_loc+'n_tboxjets.pkl')
    save_object(jet_level_results['tboxjet_energies'],save_loc+'tboxjet_energies.pkl')
    save_object(jet_level_results['tboxjet_etas'],save_loc+'tboxjet_etas.pkl')
    save_object(jet_level_results['tboxjet_phis'],save_loc+'tboxjet_phis.pkl')
    save_object(jet_level_results['tboxjet_pts'],save_loc+'tboxjet_pts.pkl')
    
    save_object(jet_level_results['n_pboxjets'],save_loc+'n_pboxjets.pkl')
    save_object(jet_level_results['pboxjet_energies'],save_loc+'pboxjet_energies.pkl')
    save_object(jet_level_results['pboxjet_etas'],save_loc+'pboxjet_etas.pkl')
    save_object(jet_level_results['pboxjet_phis'],save_loc+'pboxjet_phis.pkl')
    save_object(jet_level_results['pboxjet_pts'],save_loc+'pboxjet_pts.pkl')
    

