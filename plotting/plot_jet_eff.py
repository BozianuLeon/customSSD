import numpy as np 
import pandas as pd
import os

import itertools
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib
plt.style.use(hep.style.ATLAS)

def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)

def transform_angle(angle):
    #maps angle to [-pi,pi]
    angle %= 2 * np.pi  # Map angle to [0, 2π]
    if angle >= np.pi:
        angle -= 2 * np.pi  # Map angle to [-π, π]
    return angle


def leading_jet_pt(list_of_jet_energies_in_event):
    return max(list_of_jet_energies_in_event)

def nth_leading_jet_pt(list_of_jet_energies_in_event,n):
    return sorted(list_of_jet_energies_in_event,reverse=True)[n-1]

    
model_name = "comp3_SSD_model_15_real"
save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_plots/" + model_name + "/"
if not os.path.exists(save_loc):
   os.makedirs(save_loc)

file_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp3_SSD_model_15_real/20230526-05/"

file_names = [['total_energy_esdjets','total_energy_fjets','total_energy_tboxjets','total_energy_pboxjets'],
              ['total_pt_esdjets','total_pt_fjets','total_pt_tboxjets','total_pt_pboxjets'],
              ['total_eta_esdjets','total_eta_fjets','total_eta_tboxjets','total_eta_pboxjets'],
              ['total_phi_esdjets','total_phi_fjets','total_phi_tboxjets','total_phi_pboxjets']]


#removing low pt FJets
pt_fjets = load_object(file_to_look_in + '/' + 'total_pt_fjets' + '.pkl')
highpt_indices = np.argwhere(np.concatenate(pt_fjets)>5000)





esdj_pt = load_object(file_to_look_in + '/total_pt_esdjets.pkl')
fjj_pt = load_object(file_to_look_in + '/total_pt_fjets.pkl')
esdj_lead_pt = [leading_jet_pt(x) for x in esdj_pt]
fjj_lead_pt = [leading_jet_pt(y) for y in fjj_pt]
trigger_decision = np.argwhere(np.array(esdj_lead_pt)>400_000).T[0]
print(trigger_decision)


f, axs = plt.subplots(2,1, figsize=(7, 10),sharex=True)
n, bins, _ = axs[0].hist(esdj_lead_pt, bins=50,histtype='step',label='ESD')
axs[0].hist(fjj_lead_pt, bins=bins, histtype='step',label='FJet')
bin_centers = (bins[:-1] + bins[1:]) / 2
axs[1].hist(np.array(esdj_lead_pt)[trigger_decision],bins=bins,histtype='step')
axs[1].hist(np.array(fjj_lead_pt)[trigger_decision],bins=bins,histtype='step')
axs[1].set(xlabel='Leading jet pT',ylabel='Frequency')
f.savefig('eff1.png')




