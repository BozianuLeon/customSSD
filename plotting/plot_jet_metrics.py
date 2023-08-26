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
vectorised_angle = np.vectorize(transform_angle)





save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/comp3_SSD_model_15_real/jets/"
if not os.path.exists(save_loc):
   os.makedirs(save_loc)

file_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp3_SSD_model_15_real/20230526-05/"

file_names = [['total_energy_esdjets','total_energy_fjets','total_energy_tboxjets','total_energy_pboxjets'],
              ['total_pt_esdjets','total_pt_fjets','total_pt_tboxjets','total_pt_pboxjets'],
              ['total_eta_esdjets','total_eta_fjets','total_eta_tboxjets','total_eta_pboxjets'],
              ['total_phi_esdjets','total_phi_fjets','total_phi_tboxjets','total_phi_pboxjets'],
              ['total_n_esdjets','total_n_fjets','total_n_tboxjets','total_n_pboxjets']]

xlabs = ['Jet E', 'Jet $p_{T}$ (MeV)', '$\eta$', '$\phi$', 'N jets']



#removing low pt FJets
pt_fjets = load_object(file_to_look_in + '/' + 'total_pt_fjets' + '.pkl')
highpt_indices = np.argwhere(np.concatenate(pt_fjets)>5000)
filtered_ptfjets = [[item for item in sublist if item > 5000] for sublist in pt_fjets]



for i in range(len(file_names)):
    labels = ['ESD Jets', 'FJets>5GeV', 'True Box', 'Pred Box']
    f,ax = plt.subplots()
    for j in range(len(file_names[i])):
        events_list = load_object(file_to_look_in + '/' + file_names[i][j] + '.pkl')
        try:
            total_array = np.concatenate(events_list)
            if file_names[i][j] in ['total_energy_fjets','total_pt_fjets','total_eta_fjets','total_phi_fjets','total_n_fjets']:
                total_array = total_array[highpt_indices]
            if file_names[i][j] in ['total_phi_fjets','total_phi_tboxjets','total_phi_pboxjets']:
                total_array = vectorised_angle(total_array)
        except ValueError:
            total_array = events_list #when theres no list within list structure i.e n_jets
            if file_names[i][j]=='total_n_fjets':
                total_array = [len(x) for x in filtered_ptfjets]


        ax.hist(total_array,bins=100,label="{} ({} Jets)".format(labels[j],len(total_array)),histtype='step')
        # try:
        #     ax.hist(total_array,bins=bins_same,label="{} ({} Jets)".format(labels[j],len(total_array)),histtype='step')
        # except NameError:
        #     _, bins_same, _ = ax.hist(total_array,bins=50,label="{} ({} Jets)".format(labels[j],len(total_array)),histtype='step')
    # del bins_same
    if i==1:
        ax.axvline(x=5000,ls='--',color='cyan',label='5GeV')
        ax.set_yscale('log')
        formatter = matplotlib.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)) 
        ax.xaxis.set_major_formatter(formatter)
        # ax.set_xscale('log')

    ax.legend()
    ax.grid(color="0.95")
    ax.set(xlabel=xlabs[i],ylabel='Freq.')
    f.savefig(save_loc + file_names[i][0] + '.png')
    plt.close()





pt_tbox = load_object(file_to_look_in + '/' + 'total_pt_tboxjets' + '.pkl')
tot_pt_tbox = np.concatenate(pt_tbox)

pt_pbox = load_object(file_to_look_in + '/' + 'total_pt_pboxjets' + '.pkl')
tot_pt_pbox = np.concatenate(pt_pbox)
f,ax = plt.subplots()
ax.hist(tot_pt_tbox,bins=100,histtype='step',label='true box')
ax.hist(tot_pt_pbox,bins=100,histtype='step',label='pred box')
ax.legend()
f.savefig(save_loc + 'box_pts.png')




#####################################################################################################################################
#Plot 1, the energies of our boxes and the true clusters
#total:
total_clus_energies = np.concatenate(load_object(file_to_look_in + 'total_clus_energy.pkl'))
total_pred_energies = np.concatenate(load_object(file_to_look_in + 'total_pred_energy.pkl'))
total_tru_energies = np.concatenate(load_object(file_to_look_in + 'total_tru_energy.pkl'))

f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
n_clus, bins, _ = ax[0].hist(total_clus_energies/1000,bins=50,histtype='step',color='tab:blue',label='True Clusters >5GeV ({})'.format(len(total_clus_energies)))
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

