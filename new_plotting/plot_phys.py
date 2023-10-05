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

def get_ratio(numer,denom):
    result = np.zeros_like(numer, dtype=float)
    non_zero_indices = denom != 0
    # Perform element-wise division, handling zeros in the denominator
    result[non_zero_indices] = numer[non_zero_indices] / denom[non_zero_indices]
    return result





def make_phys_plots(
    folder_containing_lists,
    save_folder,
):
    save_loc = save_folder + "/phys/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    #####################################################################################################################################
    #Plot 1, the energies of our boxes and the true clusters
    #total:
    total_clus_energies = np.concatenate(load_object(folder_containing_lists + '/total_clus_energy.pkl'))
    total_pred_energies = np.concatenate(load_object(folder_containing_lists + '/total_pred_energy.pkl'))
    total_tru_energies = np.concatenate(load_object(folder_containing_lists + '/total_tru_energy.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_clus, bins, _ = ax[0].hist(total_clus_energies/1000,bins=50,histtype='step',color='tab:blue',label='True Clusters >1GeV ({})'.format(len(total_clus_energies)))
    n_pbox, _, _ = ax[0].hist(total_pred_energies/1000,bins=bins,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_energies)))
    n_tbox, _, _ = ax[0].hist(total_tru_energies/1000,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_energies)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Cluster/Box Energies',yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
    ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(save_loc + '/total_cluster_boxes_energy.png')
    plt.close()


    #Matched
    total_match_pred_energy = np.concatenate(load_object(folder_containing_lists + '/total_match_pred_energy.pkl'))
    total_match_tru_energy = np.concatenate(load_object(folder_containing_lists + '/total_match_tru_energy.pkl'))
    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_match_pred_energy/1000,bins=50,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_match_pred_energy)))
    n_tbox, _, _ = ax[0].hist(total_match_tru_energy/1000,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_match_tru_energy)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Matched Box Energies',yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(save_loc + '/total_match_boxes_energy.png')
    plt.close()


    #Unmatched
    total_unmatch_pred_energy = np.concatenate(load_object(folder_containing_lists + '/total_unmatch_pred_energy.pkl'))
    total_unmatch_tru_energy = np.concatenate(load_object(folder_containing_lists + '/total_unmatch_tru_energy.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_unmatch_pred_energy/1000,bins=50,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_unmatch_pred_energy)))
    n_tbox, _, _ = ax[0].hist(total_unmatch_tru_energy/1000,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_unmatch_tru_energy)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Unmatched Box Energies',yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(save_loc + '/total_unmatch_boxes_energy.png')
    plt.close()


    #####################################################################################################################################
    #Plot 2, the etas of our boxes and the true clusters
    #total
    total_clus_etas = np.concatenate(load_object(folder_containing_lists + '/total_clus_eta.pkl'))
    total_pred_etas = np.concatenate(load_object(folder_containing_lists + '/total_pred_eta.pkl'))
    total_tru_etas = np.concatenate(load_object(folder_containing_lists + '/total_tru_eta.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_clus, bins, _ = ax[0].hist(total_clus_etas,bins=50,histtype='step',color='tab:blue',label='True Clusters >1GeV ({})'.format(len(total_clus_etas)))
    n_pbox, _, _ = ax[0].hist(total_pred_etas,bins=bins,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_etas)))
    n_tbox, _, _ = ax[0].hist(total_tru_etas,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_etas)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Cluster/Box Eta',yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)

    ax[1].set(xlabel="Cluster Eta",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(save_loc + '/total_cluster_boxes_eta.png')
    plt.close()


    #Matched
    total_match_pred_eta = np.concatenate(load_object(folder_containing_lists + '/total_match_pred_eta.pkl'))
    total_match_tru_eta = np.concatenate(load_object(folder_containing_lists + '/total_match_tru_eta.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_match_pred_eta,bins=50,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_match_pred_eta)))
    n_tbox, _, _ = ax[0].hist(total_match_tru_eta,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_match_tru_eta)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Matched Box Eta',yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Eta",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(save_loc + '/total_match_boxes_eta.png')
    plt.close()


    #Unmatched
    total_unmatch_pred_eta = np.concatenate(load_object(folder_containing_lists + '/total_unmatch_pred_eta.pkl'))
    total_unmatch_tru_eta = np.concatenate(load_object(folder_containing_lists + '/total_unmatch_tru_eta.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_unmatch_pred_eta,bins=20,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_unmatch_pred_eta)))
    n_tbox, _, _ = ax[0].hist(total_unmatch_tru_eta,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_unmatch_tru_eta)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Unmatched Box Eta',yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Eta",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(save_loc + '/total_unmatch_boxes_eta.png')
    plt.close()



    #####################################################################################################################################
    #Plot 3, the phis of our boxes and the true clusters
    #total
    total_clus_phis = np.concatenate(load_object(folder_containing_lists + '/total_clus_phi.pkl'))
    total_pred_phis = np.concatenate(load_object(folder_containing_lists + '/total_pred_phi.pkl'))
    total_tru_phis = np.concatenate(load_object(folder_containing_lists + '/total_tru_phi.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_clus, bins, _ = ax[0].hist(total_clus_phis,bins=50,histtype='step',color='tab:blue',label='True Clusters >1GeV ({})'.format(len(total_clus_phis)))
    n_pbox, _, _ = ax[0].hist(total_pred_phis,bins=bins,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_phis)))
    n_tbox, _, _ = ax[0].hist(total_tru_phis,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_phis)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Cluster/Box Phi',yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)

    ax[1].set(xlabel="Cluster Phi",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(save_loc + '/total_cluster_boxes_phi.png')
    plt.close()


    #Matched
    total_match_pred_phi = np.concatenate(load_object(folder_containing_lists + '/total_match_pred_phi.pkl'))
    total_match_tru_phi = np.concatenate(load_object(folder_containing_lists + '/total_match_tru_phi.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_match_pred_phi,bins=50,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_match_pred_phi)))
    n_tbox, _, _ = ax[0].hist(total_match_tru_phi,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_match_tru_phi)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Matched Box Phi',yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Phi",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(save_loc + 'total_match_boxes_phi.png')
    plt.close()


    #Unmatched
    total_unmatch_pred_phi = np.concatenate(load_object(folder_containing_lists + '/total_unmatch_pred_phi.pkl'))
    total_unmatch_tru_phi = np.concatenate(load_object(folder_containing_lists + '/total_unmatch_tru_phi.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_unmatch_pred_phi,bins=20,histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_unmatch_pred_phi)))
    n_tbox, _, _ = ax[0].hist(total_unmatch_tru_phi,bins=bins,histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_unmatch_tru_phi)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Unmatched Box Phi',yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Phi",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(save_loc + 'total_unmatch_boxes_phi.png')
    plt.close()

    return 




folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD_50k5_mu_20e/phys_metrics/"
save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_plots/SSD_50k5_mu_20e/"
if __name__=="__main__":
    print('Making physics plots')
    make_phys_plots(folder_to_look_in,save_at)
    print('Completed physics plots\n')


