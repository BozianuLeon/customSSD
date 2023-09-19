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

save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD_model_20_real_PU/"
if not os.path.exists(save_loc):
   os.makedirs(save_loc)
file_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD_model_20_real_PU/20230831-05/"
# file_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD_model_15_real_PU/20230821-12/"

total_match_energy_ratios = np.concatenate(load_object(file_to_look_in + "total_match_energy_ratios.pkl"))
file_names = ['total_match_energy_ratios',
              'total_match_eta_diff',
              'total_match_phi_diff',
              'total_match_n_diff',
              ['total_match_pred_energy','total_match_tru_energy'],
              ['total_match_pred_eta','total_match_tru_eta'],
              ['total_match_pred_phi','total_match_tru_phi'],
              ['total_match_pred_n','total_match_tru_n'],
              ['total_unmatch_pred_energy','total_unmatch_tru_energy'],
              ['total_unmatch_pred_eta','total_unmatch_tru_eta'],
              ['total_unmatch_pred_phi','total_unmatch_tru_phi'],
              ['total_unmatch_pred_n','total_unmatch_tru_n']]

titles = ['Matched','Matched','Matched','Matched','Matched','Matched','Matched','Matched','Unmatched','Unmatched','Unmatched','Unmatched']
xlabs = ['$E_{pred} / E_{truth}$', '$\eta_{pred} - \eta_{truth}$', '$\phi_{pred} - \phi_{truth}$', '$\Delta$ n_cells',
         'Energy', 'Eta', 'Phi', 'N_cells', 'Energy', 'Eta', 'Phi', 'N_cells']


for i in range(len(file_names)):
    f,ax = plt.subplots()
    if isinstance(file_names[i],list):
        labels = ['pred','truth']
        for j in range(len(file_names[i])):
            events_list = load_object(file_to_look_in + '/' + file_names[i][j] + '.pkl')
            total_list = np.concatenate(events_list)
            try:
                ax.hist(total_list,bins=bins_same,label="{} ({} boxes)".format(labels[j],len(total_list)),histtype='step')
            except NameError:
                _, bins_same, _ = ax.hist(total_list,bins=50,label="{} ({} boxes)".format(labels[j],len(total_list)),histtype='step')
        del bins_same
        
        ax.legend()
        ax.grid(color="0.95")
        ax.set(xlabel=xlabs[i],ylabel='Freq.',title='{} {} Boxes'.format(len(total_list),titles[i]))
        f.savefig(save_loc + file_names[i][0] + '.png')
        plt.close()
    else:
        print(i,file_names[i])
        events_list = load_object(file_to_look_in + '/' + file_names[i] + '.pkl')
        total_list = np.concatenate(events_list)
        print(min(total_list),max(total_list))
        if file_names[i]=='total_match_energy_ratios':
            binsE = np.linspace(0,10,num=50)
            ax.hist(total_list,bins=binsE,histtype='step')
        # elif file_names[i]=='total_match_phi_diff':
        #     binsP = np.linspace(-2,2,num=50)
        #     ax.hist(total_list,bins=binsP,histtype='step')       
        else:
            ax.hist(total_list,bins=50,histtype='step')
        ax.grid(color="0.95")
        ax.set(xlabel=xlabs[i],ylabel='Freq.',title='{} {} Boxes'.format(len(total_list),titles[i]))
        f.savefig(save_loc + file_names[i] + '.png')
        plt.close()







