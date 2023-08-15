import numpy as np 
import pandas as pd
import os

import itertools
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.pyplot as plt
import matplotlib

def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)

def transform_angle(angle):
    #maps angle to [-pi,pi]
    angle %= 2 * np.pi  # Map angle to [0, 2π]
    if angle >= np.pi:
        angle -= 2 * np.pi  # Map angle to [-π, π]
    return angle


    

# save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/comp3_SSD_model_15_real/"
save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/comp3_SSD_model_15_real/jets/"
if not os.path.exists(save_loc):
   os.makedirs(save_loc)

file_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp3_SSD_model_15_real/20230526-05/"

total_match_energy_ratios = np.concatenate(load_object(file_to_look_in + "/total_match_energy_ratios.pkl"))

file_names = [['total_energy_esdjets','total_energy_fjets','total_energy_tboxjets','total_energy_pboxjets'],
              ['total_pt_esdjets','total_pt_fjets','total_pt_tboxjets','total_pt_pboxjets'],
              ['total_eta_esdjets','total_eta_fjets','total_eta_tboxjets','total_eta_pboxjets'],
              ['total_phi_esdjets','total_phi_fjets','total_phi_tboxjets','total_phi_pboxjets'],
              ['total_m_esdjets','total_m_fjets','total_m_tboxjets','total_m_pboxjets']]

xlabs = ['Jet E', 'Jet $p_{T}$', '$\eta$', '$\phi$', 'Mass']


#now we want to look at only events with >4 jets 
n_esdjets = load_object(file_to_look_in + '/' + 'total_n_esdjets' + '.pkl')
n_fjets = load_object(file_to_look_in + '/' + 'total_n_fjets' + '.pkl')
n_tboxjets = load_object(file_to_look_in + '/' + 'total_n_tboxjets' + '.pkl')
n_pboxjets = load_object(file_to_look_in + '/' + 'total_n_pboxjets' + '.pkl')

# high_n_indices = np.argwhere(n_esdjets>=4)
# high_n_indices2 = np.argwhere(n_fjets>=4)
# high_n_indices3 = np.argwhere(n_tboxjets>=4)
# high_n_indices4 = np.argwhere(n_pboxjets>=4)

high_n_indices = [np.argwhere(n_esdjets>=4),np.argwhere(n_fjets>=4),np.argwhere(n_tboxjets>=4),np.argwhere(n_pboxjets>=4)]
#removing low pt FJets
pt_fjets = load_object(file_to_look_in + '/' + 'total_pt_fjets' + '.pkl')
highpt_indices = np.argwhere(np.concatenate(pt_fjets)>5000)



for i in range(len(file_names)):
    labels = ['ESD Jets', 'FJets>5GeV', 'True Box', 'Pred Box']
    f,ax = plt.subplots()
    for j in range(len(file_names[i])):
        events_list = load_object(file_to_look_in + '/' + file_names[i][j] + '.pkl')
        print(len(events_list))
        events_list = events_list[high_n_indices[j]]
        print(len(events_list))
        total_array = np.concatenate(events_list)
        if file_names[i][j] in ['total_energy_fjets','total_pt_fjets','total_eta_fjets','total_phi_fjets','total_n_fjets','total_m_fjets']:
            total_array = total_array[highpt_indices]
        if file_names[i][j] in ['total_phi_fjets','total_phi_tboxjets','total_phi_pboxjets']:
            total_array = transform_angle(total_array)

        ax.hist(total_array,bins=100,label="{} ({} Jets)".format(labels[j],len(total_array)),histtype='step',density=True)


    ax.legend()
    ax.grid(color="0.95")
    ax.set(xlabel=xlabs[i],ylabel='Freq.')
    f.savefig(save_loc + file_names[i][0] + '.png')
    plt.close()




