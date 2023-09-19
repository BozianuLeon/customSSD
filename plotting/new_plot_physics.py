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



def get_ratio(numer,denom):
    result = np.zeros_like(numer, dtype=float)
    non_zero_indices = denom != 0
    # Perform element-wise division, handling zeros in the denominator
    result[non_zero_indices] = numer[non_zero_indices] / denom[non_zero_indices]
    return result


def remove_nan(array):
    #find the indices where there are not nan values
    good_indices = np.where(array==array) 
    return array[good_indices]

def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)

# save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/comp3_SSD_model_15_real/clusters/"
save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD_model_20_real_PU/clusters/"
if not os.path.exists(save_loc):
   os.makedirs(save_loc)
# file_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp3_SSD_model_15_real/20230526-05/"
file_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD_model_20_real_PU/20230831-05/"

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




#Average number of "fake" boxes
unmatch_pred_e = load_object(file_to_look_in + 'total_unmatch_pred_energy.pkl')
n_unmatched_per_event = [len(sublist1)  for sublist1 in unmatch_pred_e]
average_n = sum(n_unmatched_per_event) / len(n_unmatched_per_event)
print(len(n_unmatched_per_event))
print(average_n)

unmatch_true_e = load_object(file_to_look_in + 'total_unmatch_tru_energy.pkl')
n_unmatched_per_event = [len(sublist1)  for sublist1 in unmatch_true_e]
average_n = sum(n_unmatched_per_event) / len(n_unmatched_per_event)
print(len(n_unmatched_per_event))
print(average_n)

true_cluster_numbers = load_object(file_to_look_in + 'total_clus_energy.pkl')


#####################################################################################################################################
#Plot 1, the energies of our boxes and the true clusters
#total:
total_clus_energies = np.concatenate(load_object(file_to_look_in + 'total_clus_energy.pkl'))
total_pred_energies = np.concatenate(load_object(file_to_look_in + 'total_pred_energy.pkl'))
total_tru_energies = np.concatenate(load_object(file_to_look_in + 'total_tru_energy.pkl'))

f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
n_clus, bins, _ = ax[0].hist(total_clus_energies/1000,bins=50,histtype='step',color='tab:blue',label='True Clusters >1GeV ({})'.format(len(total_clus_energies)))
n_pbox, _, _ = ax[0].hist(total_pred_energies/1000,bins=bins,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_energies)))
n_tbox, _, _ = ax[0].hist(total_tru_energies/1000,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_energies)))
ax[0].grid()
ax[0].set(ylabel='Freq.',title='Cluster/Box Energies',yscale='log')
ax[0].legend()

ratios_pbox = get_ratio(n_pbox,n_clus)
ratios_tbox = get_ratio(n_tbox,n_clus)
bin_centers = (bins[:-1] + bins[1:]) / 2
# ax[1].plot(bin_centers, ratios_tbox, label='TBox Jets',marker='o',color='green',markersize=3.5)
# ax[1].plot(bin_centers, ratios_pbox, label='PBox Jets',marker='x',color='red',markersize=3.5)
ax[1].scatter(bin_centers, ratios_tbox, label='TBox Jets',marker='_',color='green',s=50)
ax[1].scatter(bin_centers, ratios_pbox, label='PBox Jets',marker='_',color='red',s=50)
ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
ax[1].grid()
# ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000)))
f.subplots_adjust(hspace=0)
f.savefig(save_loc + 'total_cluster_boxes_energy.png')
plt.close()



#Matched
total_match_pred_energy = np.concatenate(load_object(file_to_look_in + 'total_match_pred_energy.pkl'))
total_match_tru_energy = np.concatenate(load_object(file_to_look_in + 'total_match_tru_energy.pkl'))
print(len(total_match_tru_energy),len(total_match_tru_energy))

f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
# n_clus, bins, _ = ax[0].hist(total_clus_energies/1000,bins=50,histtype='step',label='True Clusters >5GeV ({})'.format(len(total_clus_energies)))
n_pbox, bins, _ = ax[0].hist(total_match_pred_energy/1000,bins=50,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_match_pred_energy)))
n_tbox, _, _ = ax[0].hist(total_match_tru_energy/1000,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_match_tru_energy)))
ax[0].grid()
ax[0].set(ylabel='Freq.',title='Matched Box Energies',yscale='log')
ax[0].legend()

ratios_pbox = get_ratio(n_pbox,n_tbox)
bin_centers = (bins[:-1] + bins[1:]) / 2
ax[1].scatter(bin_centers, ratios_pbox, label='PBox Jets',marker='_',color='red',s=50)
ax[1].axhline(1,ls='--',color='green',alpha=0.5)
ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
ax[1].grid()
f.subplots_adjust(hspace=0)
f.savefig(save_loc + 'total_match_boxes_energy.png')
plt.close()




#Unmatched
total_unmatch_pred_energy = np.concatenate(load_object(file_to_look_in + 'total_unmatch_pred_energy.pkl'))
total_unmatch_tru_energy = np.concatenate(load_object(file_to_look_in + 'total_unmatch_tru_energy.pkl'))

f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
# n_clus, bins, _ = ax[0].hist(total_clus_energies/1000,bins=50,histtype='step',label='True Clusters >5GeV ({})'.format(len(total_clus_energies)))
n_pbox, bins, _ = ax[0].hist(total_unmatch_pred_energy/1000,bins=50,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_unmatch_pred_energy)))
n_tbox, _, _ = ax[0].hist(total_unmatch_tru_energy/1000,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_unmatch_tru_energy)))
ax[0].grid()
ax[0].set(ylabel='Freq.',title='Unmatched Box Energies',yscale='log')
ax[0].legend()

ratios_pbox = get_ratio(n_pbox,n_tbox)
bin_centers = (bins[:-1] + bins[1:]) / 2
ax[1].scatter(bin_centers, ratios_pbox, label='PBox Jets',marker='_',color='red',s=50)
ax[1].axhline(1,ls='--',color='green',alpha=0.5)
ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
ax[1].grid()
f.subplots_adjust(hspace=0)
f.savefig(save_loc + 'total_unmatch_boxes_energy.png')
plt.close()






#####################################################################################################################################
#Plot 2, the etas of our boxes and the true clusters
#total
total_clus_etas = np.concatenate(load_object(file_to_look_in + 'total_clus_eta.pkl'))
total_pred_etas = np.concatenate(load_object(file_to_look_in + 'total_pred_eta.pkl'))
total_tru_etas = np.concatenate(load_object(file_to_look_in + 'total_tru_eta.pkl'))

f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
n_clus, bins, _ = ax[0].hist(total_clus_etas,bins=50,histtype='step',color='tab:blue',label='True Clusters >1GeV ({})'.format(len(total_clus_etas)))
n_pbox, _, _ = ax[0].hist(total_pred_etas,bins=bins,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_etas)))
n_tbox, _, _ = ax[0].hist(total_tru_etas,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_etas)))
ax[0].grid()
ax[0].set(ylabel='Freq.',title='Cluster/Box Eta',yscale='log')
ax[0].legend()

ratios_pbox = get_ratio(n_pbox,n_clus)
ratios_tbox = get_ratio(n_tbox,n_clus)
bin_centers = (bins[:-1] + bins[1:]) / 2
ax[1].scatter(bin_centers, ratios_tbox, label='TBox Jets',marker='_',color='green',s=50)
ax[1].scatter(bin_centers, ratios_pbox, label='PBox Jets',marker='_',color='red',s=50)
ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)

ax[1].set(xlabel="Cluster Eta",ylabel='Ratio')
ax[1].grid()
# ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000)))
f.subplots_adjust(hspace=0)
f.savefig(save_loc + 'total_cluster_boxes_eta.png')
plt.close()


#Matched
total_match_pred_eta = np.concatenate(load_object(file_to_look_in + 'total_match_pred_eta.pkl'))
total_match_tru_eta = np.concatenate(load_object(file_to_look_in + 'total_match_tru_eta.pkl'))

f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
n_pbox, bins, _ = ax[0].hist(total_match_pred_eta,bins=50,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_match_pred_eta)))
n_tbox, _, _ = ax[0].hist(total_match_tru_eta,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_match_tru_eta)))
ax[0].grid()
ax[0].set(ylabel='Freq.',title='Matched Box Eta',yscale='log')
ax[0].legend()

ratios_pbox = get_ratio(n_pbox,n_tbox)
bin_centers = (bins[:-1] + bins[1:]) / 2
ax[1].scatter(bin_centers, ratios_pbox, label='PBox Jets',marker='_',color='red',s=50)
ax[1].axhline(1,ls='--',color='green',alpha=0.5)
ax[1].set(xlabel="Eta",ylabel='Ratio')
ax[1].grid()
f.subplots_adjust(hspace=0)
f.savefig(save_loc + 'total_match_boxes_eta.png')
plt.close()



#Unmatched
total_unmatch_pred_eta = np.concatenate(load_object(file_to_look_in + 'total_unmatch_pred_eta.pkl'))
total_unmatch_tru_eta = np.concatenate(load_object(file_to_look_in + 'total_unmatch_tru_eta.pkl'))

f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
n_pbox, bins, _ = ax[0].hist(total_unmatch_pred_eta,bins=20,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_unmatch_pred_eta)))
n_tbox, _, _ = ax[0].hist(total_unmatch_tru_eta,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_unmatch_tru_eta)))
ax[0].grid()
ax[0].set(ylabel='Freq.',title='Unmatched Box Eta',yscale='log')
ax[0].legend()

ratios_pbox = get_ratio(n_pbox,n_tbox)
bin_centers = (bins[:-1] + bins[1:]) / 2
ax[1].scatter(bin_centers, ratios_pbox, label='PBox Jets',marker='_',color='red',s=50)
ax[1].axhline(1,ls='--',color='green',alpha=0.5)
ax[1].set(xlabel="Eta",ylabel='Ratio')
ax[1].grid()
f.subplots_adjust(hspace=0)
f.savefig(save_loc + 'total_unmatch_boxes_eta.png')
plt.close()





#####################################################################################################################################
#Plot 3, the phis of our boxes and the true clusters
#total
total_clus_phis = np.concatenate(load_object(file_to_look_in + 'total_clus_phi.pkl'))
total_pred_phis = np.concatenate(load_object(file_to_look_in + 'total_pred_phi.pkl'))
total_tru_phis = np.concatenate(load_object(file_to_look_in + 'total_tru_phi.pkl'))

f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
n_clus, bins, _ = ax[0].hist(total_clus_phis,bins=50,histtype='step',color='tab:blue',label='True Clusters >1GeV ({})'.format(len(total_clus_phis)))
n_pbox, _, _ = ax[0].hist(total_pred_phis,bins=bins,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_phis)))
n_tbox, _, _ = ax[0].hist(total_tru_phis,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_phis)))
ax[0].grid()
ax[0].set(ylabel='Freq.',title='Cluster/Box Phi',yscale='log')
ax[0].legend()

ratios_pbox = get_ratio(n_pbox,n_clus)
ratios_tbox = get_ratio(n_tbox,n_clus)
bin_centers = (bins[:-1] + bins[1:]) / 2
ax[1].scatter(bin_centers, ratios_tbox, label='TBox Jets',marker='_',color='green',s=50)
ax[1].scatter(bin_centers, ratios_pbox, label='PBox Jets',marker='_',color='red',s=50)
ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)

ax[1].set(xlabel="Cluster Phi",ylabel='Ratio')
ax[1].grid()
# ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000)))
f.subplots_adjust(hspace=0)
f.savefig(save_loc + 'total_cluster_boxes_phi.png')
plt.close()


#Matched
total_match_pred_phi = np.concatenate(load_object(file_to_look_in + 'total_match_pred_phi.pkl'))
total_match_tru_phi = np.concatenate(load_object(file_to_look_in + 'total_match_tru_phi.pkl'))

f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
n_pbox, bins, _ = ax[0].hist(total_match_pred_phi,bins=50,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_match_pred_phi)))
n_tbox, _, _ = ax[0].hist(total_match_tru_phi,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_match_tru_phi)))
ax[0].grid()
ax[0].set(ylabel='Freq.',title='Matched Box Phi',yscale='log')
ax[0].legend()

ratios_pbox = get_ratio(n_pbox,n_tbox)
bin_centers = (bins[:-1] + bins[1:]) / 2
ax[1].scatter(bin_centers, ratios_pbox, label='PBox Jets',marker='_',color='red',s=50)
ax[1].axhline(1,ls='--',color='green',alpha=0.5)
ax[1].set(xlabel="Phi",ylabel='Ratio')
ax[1].grid()
f.subplots_adjust(hspace=0)
f.savefig(save_loc + 'total_match_boxes_phi.png')
plt.close()



#Unmatched
total_unmatch_pred_phi = np.concatenate(load_object(file_to_look_in + 'total_unmatch_pred_phi.pkl'))
total_unmatch_tru_phi = np.concatenate(load_object(file_to_look_in + 'total_unmatch_tru_phi.pkl'))

f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
n_pbox, bins, _ = ax[0].hist(total_unmatch_pred_phi,bins=20,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_unmatch_pred_phi)))
n_tbox, _, _ = ax[0].hist(total_unmatch_tru_phi,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_unmatch_tru_phi)))
ax[0].grid()
ax[0].set(ylabel='Freq.',title='Unmatched Box Phi',yscale='log')
ax[0].legend()

ratios_pbox = get_ratio(n_pbox,n_tbox)
bin_centers = (bins[:-1] + bins[1:]) / 2
ax[1].scatter(bin_centers, ratios_pbox, label='PBox Jets',marker='_',color='red',s=50)
ax[1].axhline(1,ls='--',color='green',alpha=0.5)
ax[1].set(xlabel="Phi",ylabel='Ratio')
ax[1].grid()
f.subplots_adjust(hspace=0)
f.savefig(save_loc + 'total_unmatch_boxes_phi.png')
plt.close()




#####################################################################################################################################
#Plot 4, the energies of the central boxes/clusters
#total
central_mask_clus = abs(total_clus_etas) < 2.5
central_mask_pred = abs(total_pred_etas) < 2.5
central_mask_truth = abs(total_tru_etas) < 2.5

central_clus_energies = total_clus_energies[central_mask_clus]
central_pred_energies = total_pred_energies[central_mask_pred]
central_tru_energies = total_tru_energies[central_mask_truth]

f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
n_clus, bins, _ = ax[0].hist(central_clus_energies/1000,bins=50,histtype='step',color='tab:blue',label='True $|\eta|<2.5$ Clusters >1GeV ({})'.format(len(central_clus_energies)))
n_pbox, _, _ = ax[0].hist(central_pred_energies/1000,bins=bins,histtype='step',color='red',label='Predicted $|\eta|<2.5$ Boxes ({})'.format(len(central_pred_energies)))
n_tbox, _, _ = ax[0].hist(central_tru_energies/1000,bins=bins,histtype='step',color='green',label='Truth $|\eta|<2.5$ Boxes ({})'.format(len(central_tru_energies)))
ax[0].grid()
ax[0].set(ylabel='Freq.',title='Central Cluster/Box Energies',yscale='log')
ax[0].legend()

ratios_pbox = get_ratio(n_pbox,n_clus)
ratios_tbox = get_ratio(n_tbox,n_clus)
bin_centers = (bins[:-1] + bins[1:]) / 2
ax[1].scatter(bin_centers, ratios_tbox, label='TBox Jets',marker='_',color='green',s=50)
ax[1].scatter(bin_centers, ratios_pbox, label='PBox Jets',marker='_',color='red',s=50)
ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
ax[1].grid()
# ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000)))
f.subplots_adjust(hspace=0)
f.savefig(save_loc + 'central_cluster_boxes_energy.png')
plt.close()






