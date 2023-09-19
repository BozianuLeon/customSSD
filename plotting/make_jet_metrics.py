import numpy as np
import torch
import awkward as ak
import matplotlib
from matplotlib import pyplot as plt
import h5py
import sys
import fastjet #!
import pickle

sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')
from utils.utils import transform_angle, remove_nan, get_cells_from_boxes, event_cluster_estimates



########################################################################################################
#load inference from .npy
save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD_model_20_real_PU/20230831-05/"
path_to_structured_array = save_loc + "struc_array.npy"

with open(path_to_structured_array, 'rb') as f:
    a = np.load(f)

print('Length of inference array',len(a))
jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)




total_energy_esdjets, total_pt_esdjets, total_eta_esdjets, total_phi_esdjets, total_n_esdjets, total_m_esdjets = list(), list(), list(), list(), list(), list()
total_energy_fjets, total_pt_fjets, total_eta_fjets, total_phi_fjets, total_n_fjets, total_m_fjets = list(), list(), list(), list(), list(), list()
total_energy_tboxjets, total_pt_tboxjets, total_eta_tboxjets, total_phi_tboxjets, total_n_tboxjets, total_m_tboxjets = list(), list(), list(), list(), list(), list()
total_energy_pboxjets, total_pt_pboxjets, total_eta_pboxjets, total_phi_pboxjets, total_n_pboxjets, total_m_pboxjets = list(), list(), list(), list(), list(), list()

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

    #load cells from h5
    #get the cells
    h5f = a[i]['h5file']
    h5f = h5f.decode('utf-8')
    print(h5f)
    event_no = a[i]['event_no']
    file = "/home/users/b/bozianu/work/data/pileup-JZ4/cells/user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f)
    with h5py.File(file,"r") as f:
        h5group = f["caloCells"]
        cells = h5group["2d"][event_no]

    clusters_file = "/home/users/b/bozianu/work/data/pileup-JZ4/clusters/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f)
    with h5py.File(clusters_file,"r") as f:
        cl_data = f["caloCells"]
        cluster_data = cl_data["2d"][event_no]
        cluster_cell_data = cl_data["3d"][event_no]

    jets_file = "/home/users/b/bozianu/work/data/pileup-JZ4/jets/user.cantel.34126190._0000{}.jetD3PD_mc16_JZ4W.r10788.h5".format(h5f)
    with h5py.File(jets_file,"r") as f:
        j_data = f["caloCells"]
        jet_data = j_data["2d"][event_no]

    print(i)
    #From the ESD file
    new_cluster_data = remove_nan(cluster_data)
    new_jet_data = remove_nan(jet_data)


    #all clusters
    ESD_inputs = []
    m = 0.0 #topoclusters have 0 mass
    for i in range(len(new_cluster_data)):
        cl_px = float(new_cluster_data[i]['cl_pt'] * np.cos(new_cluster_data[i]['cl_phi']))
        cl_py = float(new_cluster_data[i]['cl_pt'] * np.sin(new_cluster_data[i]['cl_phi']))
        cl_pz = float(new_cluster_data[i]['cl_pt'] * np.sinh(new_cluster_data[i]['cl_eta']))
        ESD_inputs.append(fastjet.PseudoJet(cl_px,cl_py,cl_pz,m))



    #From my predictions
    list_p_cl_es, list_t_cl_es = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='energy')
    list_p_cl_etas, list_t_cl_etas = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='eta')
    list_p_cl_phis, list_t_cl_phis = event_cluster_estimates(pees,scores,tees,cells,mode='match',target='phi')
    m = 0.0
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
    for oj in range(len(new_jet_data)):
        offjet = new_jet_data[oj]
        batch_energy_esdjets.append(offjet['AntiKt4EMTopoJets_E'])
        batch_pt_esdjets.append(offjet['AntiKt4EMTopoJets_JetConstitScaleMomentum_pt'])
        batch_eta_esdjets.append(offjet['AntiKt4EMTopoJets_JetConstitScaleMomentum_eta'])
        batch_phi_esdjets.append(offjet['AntiKt4EMTopoJets_JetConstitScaleMomentum_phi'])
        batch_m_esdjets.append(offjet['AntiKt4EMTopoJets_JetConstitScaleMomentum_m'])
    total_energy_esdjets.append(batch_energy_esdjets)
    total_pt_esdjets.append(batch_pt_esdjets)
    total_eta_esdjets.append(batch_eta_esdjets)
    total_phi_esdjets.append(batch_phi_esdjets)
    total_m_esdjets.append(batch_m_esdjets)
    total_n_esdjets.append(len(new_jet_data))

    #ESD Clusters:
    ESD_cluster_jets = fastjet.ClusterSequence(ESD_inputs, jetdef)
    ESD_cluster_inc_jets = ESD_cluster_jets.inclusive_jets()
    batch_energy_fjets, batch_pt_fjets, batch_eta_fjets, batch_phi_fjets, batch_m_fjets = list(),list(),list(),list(),list()
    for i in range(len(ESD_cluster_inc_jets)):
        ii = ESD_cluster_inc_jets[i]
        batch_energy_fjets.append(ii.E())
        batch_pt_fjets.append(ii.pt())
        batch_eta_fjets.append(ii.eta())
        batch_phi_fjets.append(ii.phi())
        batch_m_fjets.append(ii.m())
    total_energy_fjets.append(batch_energy_fjets)
    total_pt_fjets.append(batch_pt_fjets)
    total_eta_fjets.append(batch_eta_fjets)
    total_phi_fjets.append(batch_phi_fjets)
    total_m_fjets.append(batch_m_fjets)
    total_n_fjets.append(len(ESD_cluster_inc_jets))

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
        batch_m_tboxjets.append(jj.m())
    total_energy_tboxjets.append(batch_energy_tboxjets)
    total_pt_tboxjets.append(batch_pt_tboxjets)
    total_eta_tboxjets.append(batch_eta_tboxjets)
    total_phi_tboxjets.append(batch_phi_tboxjets)
    total_m_tboxjets.append(batch_m_tboxjets)
    total_n_tboxjets.append(len(truth_box_inc_jets))

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

    total_energy_pboxjets.append(batch_energy_pboxjets)
    total_pt_pboxjets.append(batch_pt_pboxjets)
    total_eta_pboxjets.append(batch_eta_pboxjets)
    total_phi_pboxjets.append(batch_phi_pboxjets)
    total_m_pboxjets.append(batch_m_pboxjets)
    total_n_pboxjets.append(len(pred_box_inc_jets))





def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)


save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp3_SSD_model_15_real/20230526-05/"
print('Saving the physics metrics in lists...')
#esd jets list
save_object(total_energy_esdjets,save_loc+'total_energy_esdjets.pkl')
save_object(total_pt_esdjets,save_loc+'total_pt_esdjets.pkl')
save_object(total_eta_esdjets,save_loc+'total_eta_esdjets.pkl')
save_object(total_phi_esdjets,save_loc+'total_phi_esdjets.pkl')
save_object(total_n_esdjets,save_loc+'total_n_esdjets.pkl')
save_object(total_m_esdjets,save_loc+'total_m_esdjets.pkl')
#esd clusters list
save_object(total_energy_fjets,save_loc+'total_energy_fjets.pkl')
save_object(total_pt_fjets,save_loc+'total_pt_fjets.pkl')
save_object(total_eta_fjets,save_loc+'total_eta_fjets.pkl')
save_object(total_phi_fjets,save_loc+'total_phi_fjets.pkl')
save_object(total_n_fjets,save_loc+'total_n_fjets.pkl')
save_object(total_m_fjets,save_loc+'total_m_fjets.pkl')
#truth boxes list
save_object(total_energy_tboxjets,save_loc+'total_energy_tboxjets.pkl')
save_object(total_pt_tboxjets,save_loc+'total_pt_tboxjets.pkl')
save_object(total_eta_tboxjets,save_loc+'total_eta_tboxjets.pkl')
save_object(total_phi_tboxjets,save_loc+'total_phi_tboxjets.pkl')
save_object(total_n_tboxjets,save_loc+'total_n_tboxjets.pkl')
save_object(total_m_tboxjets,save_loc+'total_m_tboxjets.pkl')
#pred boxes lists
save_object(total_energy_pboxjets,save_loc+'total_energy_pboxjets.pkl')
save_object(total_pt_pboxjets,save_loc+'total_pt_pboxjets.pkl')
save_object(total_eta_pboxjets,save_loc+'total_eta_pboxjets.pkl')
save_object(total_phi_pboxjets,save_loc+'total_phi_pboxjets.pkl')
save_object(total_n_pboxjets,save_loc+'total_n_pboxjets.pkl')
save_object(total_m_pboxjets,save_loc+'total_m_pboxjets.pkl')

