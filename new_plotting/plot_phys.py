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
    log = True
):
    save_loc = save_folder + "/phys/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    #####################################################################################################################################
    #Plot 1, the energies of our boxes and the true clusters
    #total:
    total_clus_energies = np.concatenate(load_object(folder_containing_lists + '/cluster_energies.pkl'))
    total_pred_energies = np.concatenate(load_object(folder_containing_lists + '/tbox_energies.pkl'))
    total_tru_energies = np.concatenate(load_object(folder_containing_lists + '/pbox_energies.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_clus, bins, _ = ax[0].hist(total_clus_energies/1000,bins=50,density=(not log),histtype='step',color='tab:blue',label='TopoCl >5GeV ({})'.format(len(total_clus_energies)))
    n_pbox, _, _ = ax[0].hist(total_pred_energies/1000,bins=bins,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_energies)))
    n_tbox, _, _ = ax[0].hist(total_tru_energies/1000,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_energies)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Cluster/Box Energies')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
    ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_cluster_boxes_energy_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_cluster_boxes_energy.png')
    plt.close()


    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_pred_energies/1000,bins=100,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_energies)))
    n_clus, _, _ = ax[0].hist(total_clus_energies/1000,bins=bins,density=(not log),histtype='step',color='tab:blue',label='TopoCl >5GeV ({})'.format(len(total_clus_energies)))
    n_tbox, _, _ = ax[0].hist(total_tru_energies/1000,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_energies)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Cluster/Box Energies')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
    ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_boxes_energy_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_boxes_energy.png')
    plt.close()


    #Matched
    total_match_pred_energy = np.concatenate(load_object(folder_containing_lists + '/tbox_match_energies.pkl'))
    total_match_tru_energy = np.concatenate(load_object(folder_containing_lists + '/pbox_match_energies.pkl'))
    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_match_pred_energy/1000,bins=50,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_match_pred_energy)))
    n_tbox, _, _ = ax[0].hist(total_match_tru_energy/1000,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_match_tru_energy)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Matched Box Energies')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_match_boxes_energy_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_match_boxes_energy.png')
    plt.close()


    #Unmatched
    total_unmatch_pred_energy = np.concatenate(load_object(folder_containing_lists + '/tbox_unmatch_energies.pkl'))
    total_unmatch_tru_energy = np.concatenate(load_object(folder_containing_lists + '/pbox_unmatch_energies.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_unmatch_pred_energy/1000,bins=50,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_unmatch_pred_energy)))
    n_tbox, _, _ = ax[0].hist(total_unmatch_tru_energy/1000,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_unmatch_tru_energy)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Unmatched Box Energies')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_unmatch_boxes_energy_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_unmatch_boxes_energy.png')
    plt.close()


    #####################################################################################################################################
    #Plot 2, the etas of our boxes and the true clusters
    #total
    total_clus_etas = np.concatenate(load_object(folder_containing_lists + '/cluster_etas.pkl'))
    total_pred_etas = np.concatenate(load_object(folder_containing_lists + '/tbox_etas.pkl'))
    total_tru_etas = np.concatenate(load_object(folder_containing_lists + '/pbox_etas.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_clus, bins, _ = ax[0].hist(total_clus_etas,bins=50,density=(not log),histtype='step',color='tab:blue',label='TopoCl >5GeV ({})'.format(len(total_clus_etas)))
    n_pbox, _, _ = ax[0].hist(total_pred_etas,bins=bins,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_etas)))
    n_tbox, _, _ = ax[0].hist(total_tru_etas,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_etas)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Cluster/Box Eta')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)

    ax[1].set(xlabel="Cluster Eta",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_cluster_boxes_eta_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_cluster_boxes_eta.png')
    # f.savefig(save_loc + '/total_cluster_boxes_eta.png')
    plt.close()


    #Matched
    total_match_pred_eta = np.concatenate(load_object(folder_containing_lists + '/tbox_match_etas.pkl'))
    total_match_tru_eta = np.concatenate(load_object(folder_containing_lists + '/pbox_match_etas.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_match_pred_eta,bins=50,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_match_pred_eta)))
    n_tbox, _, _ = ax[0].hist(total_match_tru_eta,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_match_tru_eta)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Matched Box Eta')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Eta",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_match_boxes_eta_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_match_boxes_eta.png')
    # f.savefig(save_loc + '/total_match_boxes_eta.png')
    plt.close()


    #Unmatched
    total_unmatch_pred_eta = np.concatenate(load_object(folder_containing_lists + '/tbox_unmatch_etas.pkl'))
    total_unmatch_tru_eta = np.concatenate(load_object(folder_containing_lists + '/pbox_unmatch_etas.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_unmatch_pred_eta,bins=50,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_unmatch_pred_eta)))
    n_tbox, _, _ = ax[0].hist(total_unmatch_tru_eta,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_unmatch_tru_eta)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Unmatched Box Eta')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Eta",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_unmatch_boxes_eta_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_unmatch_boxes_eta.png')
    # f.savefig(save_loc + '/total_unmatch_boxes_eta.png')
    plt.close()



    #####################################################################################################################################
    #Plot 3, the phis of our boxes and the true clusters
    #total
    total_clus_phis = np.concatenate(load_object(folder_containing_lists + '/cluster_phis.pkl'))
    total_pred_phis = np.concatenate(load_object(folder_containing_lists + '/tbox_phis.pkl'))
    total_tru_phis = np.concatenate(load_object(folder_containing_lists + '/pbox_phis.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_clus, bins, _ = ax[0].hist(total_clus_phis,bins=50,density=(not log),histtype='step',color='tab:blue',label='TopoCl >5GeV ({})'.format(len(total_clus_phis)))
    n_pbox, _, _ = ax[0].hist(total_pred_phis,bins=bins,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_phis)))
    n_tbox, _, _ = ax[0].hist(total_tru_phis,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_phis)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Cluster/Box Phi')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)

    ax[1].set(xlabel="Cluster Phi",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_cluster_boxes_phi_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_cluster_boxes_phi.png')
    # f.savefig(save_loc + '/total_cluster_boxes_phi.png')
    plt.close()


    #Matched
    total_match_pred_phi = np.concatenate(load_object(folder_containing_lists + '/tbox_match_phis.pkl'))
    total_match_tru_phi = np.concatenate(load_object(folder_containing_lists + '/pbox_match_phis.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_match_pred_phi,bins=50,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_match_pred_phi)))
    n_tbox, _, _ = ax[0].hist(total_match_tru_phi,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_match_tru_phi)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Matched Box Phi')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Phi",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_match_boxes_phi_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_match_boxes_phi.png')
    # f.savefig(save_loc + 'total_match_boxes_phi.png')
    plt.close()


    #Unmatched
    total_unmatch_pred_phi = np.concatenate(load_object(folder_containing_lists + '/tbox_unmatch_phis.pkl'))
    total_unmatch_tru_phi = np.concatenate(load_object(folder_containing_lists + '/pbox_unmatch_phis.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_unmatch_pred_phi,bins=50,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_unmatch_pred_phi)))
    n_tbox, _, _ = ax[0].hist(total_unmatch_tru_phi,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_unmatch_tru_phi)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Unmatched Box Phi')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Phi",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_unmatch_boxes_phi_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_unmatch_boxes_phi.png')
    # f.savefig(save_loc + 'total_unmatch_boxes_phi.png')
    plt.close()



    #####################################################################################################################################
    #Plot 4, the number of cells in our boxes and the true clusters
    #total
    total_clus_n_cells = np.concatenate(load_object(folder_containing_lists + '/cluster_n_cells.pkl'))
    total_pred_n_cells = np.concatenate(load_object(folder_containing_lists + '/tbox_n_cells.pkl'))
    total_tru_n_cells = np.concatenate(load_object(folder_containing_lists + '/pbox_n_cells.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_clus, bins, _ = ax[0].hist(total_clus_n_cells,bins=50,density=(not log),histtype='step',color='tab:blue',label='TopoCl >5GeV ({})'.format(len(total_clus_n_cells)))
    n_pbox, _, _ = ax[0].hist(total_pred_n_cells,bins=bins,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_n_cells)))
    n_tbox, _, _ = ax[0].hist(total_tru_n_cells,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_n_cells)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Number of cells in cluster/box')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)

    ax[1].set(xlabel="# Cells",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_cluster_boxes_n_cells_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_cluster_boxes_n_cells.png')
    # f.savefig(save_loc + '/total_cluster_boxes_n_cells.png')
    plt.close()

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_pred_n_cells,bins=100,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_n_cells)))
    n_clus, _, _ = ax[0].hist(total_clus_n_cells,bins=bins,density=(not log),histtype='step',color='tab:blue',label='TopoCl >5GeV ({})'.format(len(total_clus_n_cells)))
    n_tbox, _, _ = ax[0].hist(total_tru_n_cells,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_n_cells)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Number of cells in cluster/box')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)

    ax[1].set(xlabel="# Cells",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_boxes_n_cells_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_boxes_n_cells.png')
    plt.close()


    #Matched
    total_match_pred_n_cells = np.concatenate(load_object(folder_containing_lists + '/tbox_match_n_cells.pkl'))
    total_match_tru_n_cells = np.concatenate(load_object(folder_containing_lists + '/pbox_match_n_cells.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_match_pred_n_cells,bins=50,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_match_pred_n_cells)))
    n_tbox, _, _ = ax[0].hist(total_match_tru_n_cells,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_match_tru_n_cells)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Matched Number of cells in cluster/box')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="# Cells",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_match_boxes_n_cells_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_match_boxes_n_cells.png')
    # f.savefig(save_loc + 'total_match_boxes_n_cells.png')
    plt.close()


    #Unmatched
    total_unmatch_pred_n_cells = np.concatenate(load_object(folder_containing_lists + '/tbox_unmatch_n_cells.pkl'))
    total_unmatch_tru_n_cells = np.concatenate(load_object(folder_containing_lists + '/pbox_unmatch_n_cells.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_unmatch_pred_n_cells,bins=50,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_unmatch_pred_n_cells)))
    n_tbox, _, _ = ax[0].hist(total_unmatch_tru_n_cells,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_unmatch_tru_n_cells)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Unmatched Number of cells in cluster/box')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="# Cells",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_unmatch_boxes_n_cells_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_unmatch_boxes_n_cells.png')
    # f.savefig(save_loc + 'total_unmatch_boxes_n_cells.png')
    plt.close()


    #####################################################################################################################################
    #Plot 5, the transverse energy plots
    #total
    total_clus_eT = total_clus_energies/np.cosh(total_clus_etas)
    print(total_clus_energies.shape,total_clus_etas.shape,total_clus_eT.shape)
    total_pred_eT = np.concatenate(load_object(folder_containing_lists + '/tbox_eT.pkl'))
    total_tru_eT = np.concatenate(load_object(folder_containing_lists + '/pbox_eT.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_clus, bins, _ = ax[0].hist(total_clus_eT/1000,bins=50,density=(not log),histtype='step',color='tab:blue',label='TopoCl >5GeV ({})'.format(len(total_clus_energies)))
    n_pbox, _, _ = ax[0].hist(total_pred_eT/1000,bins=bins,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_energies)))
    n_tbox, _, _ = ax[0].hist(total_tru_eT/1000,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_energies)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Transverse Energy')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
    ax[1].set(xlabel="Cluster $E_T$ (GeV)", ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_cluster_boxes_eT_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_cluster_boxes_eT.png')
    plt.close()


    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_pred_eT/1000,bins=100,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_pred_energies)))
    n_clus, _, _ = ax[0].hist(total_clus_eT/1000,bins=bins,density=(not log),histtype='step',color='tab:blue',label='TopoCl >5GeV ({})'.format(len(total_clus_energies)))
    n_tbox, _, _ = ax[0].hist(total_tru_eT/1000,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_tru_energies)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Transverse Energy')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
    ax[1].set(xlabel="Cluster $E_T$ (GeV)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_boxes_eT_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_boxes_eT.png')
    plt.close()


    #Matched
    total_match_pred_eT = np.concatenate(load_object(folder_containing_lists + '/tbox_match_eT.pkl'))
    total_match_tru_eT = np.concatenate(load_object(folder_containing_lists + '/pbox_match_eT.pkl'))
    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_match_pred_eT/1000,bins=50,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_match_pred_energy)))
    n_tbox, _, _ = ax[0].hist(total_match_tru_eT/1000,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_match_tru_energy)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Matched Box Transverse Energy')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Cluster $E_T$ (GeV)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_match_boxes_eT_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_match_boxes_eT.png')
    plt.close()


    #Unmatched
    total_unmatch_pred_eT = np.concatenate(load_object(folder_containing_lists + '/tbox_unmatch_eT.pkl'))
    total_unmatch_tru_eT = np.concatenate(load_object(folder_containing_lists + '/pbox_unmatch_eT.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_unmatch_pred_eT/1000,bins=50,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(total_unmatch_pred_energy)))
    n_tbox, _, _ = ax[0].hist(total_unmatch_tru_eT/1000,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(total_unmatch_tru_energy)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Unmatched Box Transverse Energy')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Cluster $E_T$ (GeV)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_unmatch_boxes_eT_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_unmatch_boxes_eT.png')
    plt.close()



    #####################################################################################################################################
    #Now only looking at central boxes [-2,2]
    eta_min,eta_max = -2.0, 2.0  
    
    def phys_plots_eta_bins(eta_region, observable_name, save_loc):
        cent_save_loc = save_loc + "/central/"
        if not os.path.exists(cent_save_loc):
            os.makedirs(cent_save_loc)
        
        eta_1,eta_2,eta_3,eta_4 = eta_region
        # clus_eta_mask = (total_clus_etas > eta_min) & (total_clus_etas < eta_max)
        clus_eta_mask = ((total_clus_etas > eta_1) & (total_clus_etas < eta_2) )| ((total_clus_etas > eta_3) & (total_clus_etas < eta_4))
        tbox_eta_mask = ((total_tru_etas > eta_1) & (total_tru_etas < eta_2)) | ((total_tru_etas > eta_3) & (total_tru_etas < eta_4))
        pbox_eta_mask = ((total_pred_etas > eta_1) & (total_pred_etas < eta_2)) | ((total_pred_etas > eta_3) & (total_pred_etas < eta_4))

        if observable_name=='energy':
            cent_clus_obs = total_clus_energies[clus_eta_mask]/1000
            cent_tboxes_obs = total_tru_energies[tbox_eta_mask]/1000
            cent_pboxes_obs = total_pred_energies[pbox_eta_mask]/1000
            title = f'Cluster/Box Energies in central [{eta_3},{eta_4}] region'
            xlab = f'Cluster energy (GeV)'
            
        elif observable_name=='eta':
            cent_clus_obs = total_clus_etas[clus_eta_mask]
            cent_tboxes_obs = total_tru_etas[tbox_eta_mask]
            cent_pboxes_obs = total_pred_etas[pbox_eta_mask]
            title = f'Cluster/Box eta in central [{eta_3},{eta_4}] region'
            xlab = f'eta'
            
        elif observable_name=='phi':
            cent_clus_obs = total_clus_phis[clus_eta_mask]
            cent_tboxes_obs = total_tru_phis[tbox_eta_mask]
            cent_pboxes_obs = total_pred_phis[pbox_eta_mask]
            title = f'Cluster/Box phi in central [{eta_3},{eta_4}] region'
            xlab = f'phi'
            
        elif observable_name=='eT':
            cent_clus_obs = total_clus_eT[clus_eta_mask]/1000
            cent_tboxes_obs = total_tru_eT[tbox_eta_mask]/1000
            cent_pboxes_obs = total_pred_eT[pbox_eta_mask]/1000
            title = f'Cluster/Box Transverse energy in central [{eta_3},{eta_4}] region'
            xlab = f'$E_T$'

        f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        n_clus, bins, _ = ax[0].hist(cent_clus_obs,bins=50,density=(not log),histtype='step',color='tab:blue',label='TopoCl >5GeV ({})'.format(len(cent_clus_obs)))
        n_pbox, _, _ = ax[0].hist(cent_pboxes_obs,bins=bins,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(cent_tboxes_obs)))
        n_tbox, _, _ = ax[0].hist(cent_tboxes_obs,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(cent_pboxes_obs)))
        ax[0].grid()
        ax[0].set(ylabel='Freq.',title=title)
        ax[0].legend()

        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
        ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
        ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
        ax[1].set(xlabel=xlab,ylabel='Ratio')
        ax[1].grid()
        f.subplots_adjust(hspace=0)
        if log:
            ax[0].set(yscale='log')
            f.savefig(cent_save_loc + f'/central_cluster_boxes_eta{eta_4}_{observable_name}_log.png')
        else:
            ax[0].set(ylabel='Freq. Density')
            f.savefig(cent_save_loc + f'/central_cluster_boxes_eta{eta_4}_{observable_name}.png')
        plt.close()
        


    phys_plots_eta_bins([-1.4,0.0,0.0,1.4], 'energy', save_loc)
    phys_plots_eta_bins([-1.4,0.0,0.0,1.4], 'eT', save_loc)
    phys_plots_eta_bins([-1.4,0.0,0.0,1.4], 'eta', save_loc)
    phys_plots_eta_bins([-1.4,0.0,0.0,1.4], 'phi', save_loc)

    phys_plots_eta_bins([-3.2,-1.4,1.4,3.2], 'energy', save_loc)
    phys_plots_eta_bins([-3.2,-1.4,1.4,3.2], 'eT', save_loc)
    phys_plots_eta_bins([-3.2,-1.4,1.4,3.2], 'eta', save_loc)
    phys_plots_eta_bins([-3.2,-1.4,1.4,3.2], 'phi', save_loc)

    phys_plots_eta_bins([-5.0,-3.2,3.2,5.0], 'eT', save_loc)
    phys_plots_eta_bins([-5.0,-3.2,3.2,5.0], 'energy', save_loc)
    phys_plots_eta_bins([-5.0,-3.2,3.2,5.0], 'eta', save_loc)
    phys_plots_eta_bins([-5.0,-3.2,3.2,5.0], 'phi', save_loc)

    phys_plots_eta_bins([-2.5,-2.1,2.1,2.5], 'energy', save_loc)
    phys_plots_eta_bins([-2.5,-2.1,2.1,2.5], 'eT', save_loc)
    phys_plots_eta_bins([-2.5,-2.1,2.1,2.5], 'eta', save_loc)
    phys_plots_eta_bins([-2.5,-2.1,2.1,2.5], 'phi', save_loc)



    ##Matched
    def phys_plots_eta_bins_matched(eta_region, observable_name, save_loc):
        cent_save_loc = save_loc + "/central/matched/"
        if not os.path.exists(cent_save_loc):
            os.makedirs(cent_save_loc)
        
        eta_1,eta_2,eta_3,eta_4 = eta_region
        clus_eta_mask = ((total_clus_etas > eta_1) & (total_clus_etas < eta_2) )| ((total_clus_etas > eta_3) & (total_clus_etas < eta_4))
        tbox_eta_mask = ((total_match_tru_eta > eta_1) & (total_match_tru_eta < eta_2)) | ((total_match_tru_eta > eta_3) & (total_match_tru_eta < eta_4))
        pbox_eta_mask = ((total_match_pred_eta > eta_1) & (total_match_pred_eta < eta_2)) | ((total_match_pred_eta > eta_3) & (total_match_pred_eta < eta_4))

        if observable_name=='energy':
            cent_clus_obs = total_clus_energies[clus_eta_mask]/1000
            cent_tboxes_obs = total_match_tru_energy[tbox_eta_mask]/1000
            cent_pboxes_obs = total_match_pred_energy[pbox_eta_mask]/1000
            title = f'Matched Cluster/Box Energies in central [{eta_3},{eta_4}] region'
            xlab = f'Cluster energy (GeV)'
            
        elif observable_name=='eta':
            cent_clus_obs = total_clus_etas[clus_eta_mask]
            cent_tboxes_obs = total_match_tru_eta[tbox_eta_mask]
            cent_pboxes_obs = total_match_pred_eta[pbox_eta_mask]
            title = f'Matched Cluster/Box eta in central [{eta_3},{eta_4}] region'
            xlab = f'eta'
            
        elif observable_name=='phi':
            cent_clus_obs = total_clus_phis[clus_eta_mask]
            cent_tboxes_obs = total_match_tru_phi[tbox_eta_mask]
            cent_pboxes_obs = total_match_pred_phi[pbox_eta_mask]
            title = f'Matched Cluster/Box phi in central [{eta_3},{eta_4}] region'
            xlab = f'phi'
            
        elif observable_name=='eT':
            cent_clus_obs = total_clus_eT[clus_eta_mask]/1000
            cent_tboxes_obs = total_match_tru_eT[tbox_eta_mask]/1000
            cent_pboxes_obs = total_match_pred_eT[pbox_eta_mask]/1000
            title = f'Matched Cluster/Box Transverse energy in central [{eta_3},{eta_4}] region'
            xlab = f'$E_T$'
        


        f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        n_clus, bins, _ = ax[0].hist(cent_clus_obs,bins=50,density=(not log),histtype='step',color='tab:blue',label='TopoCl >5GeV ({})'.format(len(cent_clus_obs)))
        n_pbox, _, _ = ax[0].hist(cent_pboxes_obs,bins=bins,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(cent_tboxes_obs)))
        n_tbox, _, _ = ax[0].hist(cent_tboxes_obs,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(cent_pboxes_obs)))
        ax[0].grid()
        ax[0].set(ylabel='Freq.',title=title)
        ax[0].legend()

        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
        ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
        ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
        ax[1].set(xlabel=xlab,ylabel='Ratio')
        ax[1].grid()
        f.subplots_adjust(hspace=0)
        if log:
            ax[0].set(yscale='log')
            f.savefig(cent_save_loc + f'/central_cluster_matched_boxes_eta{eta_4}_{observable_name}_log.png')
        else:
            ax[0].set(ylabel='Freq. Density')
            f.savefig(cent_save_loc + f'/central_cluster_matched_boxes_eta{eta_4}_{observable_name}.png')
        plt.close()



    phys_plots_eta_bins_matched([-1.4,0.0,0.0,1.4], 'energy', save_loc)
    phys_plots_eta_bins_matched([-3.2,-1.4,1.4,3.2], 'energy', save_loc)
    phys_plots_eta_bins_matched([-1.4,0.0,0.0,1.4], 'eta', save_loc)
    phys_plots_eta_bins_matched([-3.2,-1.4,1.4,3.2], 'eta', save_loc)
    phys_plots_eta_bins_matched([-1.4,0.0,0.0,1.4], 'eT', save_loc)
    phys_plots_eta_bins_matched([-3.2,-1.4,1.4,3.2], 'eT', save_loc)


    ##UnMatched
    def phys_plots_eta_bins_unmatched(eta_region, observable_name, save_loc):
        cent_save_loc = save_loc + "/central/unmatched/"
        if not os.path.exists(cent_save_loc):
            os.makedirs(cent_save_loc)
        
        eta_1,eta_2,eta_3,eta_4 = eta_region
        tbox_eta_mask = ((total_unmatch_tru_eta > eta_1) & (total_unmatch_tru_eta < eta_2)) | ((total_unmatch_tru_eta > eta_3) & (total_unmatch_tru_eta < eta_4))
        pbox_eta_mask = ((total_unmatch_pred_eta > eta_1) & (total_unmatch_pred_eta < eta_2)) | ((total_unmatch_pred_eta > eta_3) & (total_unmatch_pred_eta < eta_4))

        if observable_name=='energy':
            cent_tboxes_obs = total_unmatch_tru_energy[tbox_eta_mask]/1000
            cent_pboxes_obs = total_unmatch_pred_energy[pbox_eta_mask]/1000
            title = f'Unmatched Cluster/Box Energies in central [{eta_3},{eta_4}] region'
            xlab = f'Cluster energy (GeV)'
            
        elif observable_name=='eta':
            cent_tboxes_obs = total_unmatch_tru_eta[tbox_eta_mask]
            cent_pboxes_obs = total_unmatch_pred_eta[pbox_eta_mask]
            title = f'Unmatched Cluster/Box eta in central [{eta_3},{eta_4}] region'
            xlab = f'eta'
            
        elif observable_name=='phi':
            cent_tboxes_obs = total_unmatch_tru_phi[tbox_eta_mask]
            cent_pboxes_obs = total_unmatch_pred_phi[pbox_eta_mask]
            title = f'Unmatched Cluster/Box phi in central [{eta_3},{eta_4}] region'
            xlab = f'phi'
            
        elif observable_name=='eT':
            cent_tboxes_obs = total_unmatch_tru_eT[tbox_eta_mask]/1000
            cent_pboxes_obs = total_unmatch_pred_eT[pbox_eta_mask]/1000
            title = f'Unmatched Cluster/Box Transverse energy in central [{eta_3},{eta_4}] region'
            xlab = f'$E_T$'
        

        f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        n_pbox, bins, _ = ax[0].hist(cent_pboxes_obs,bins=50,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(cent_pboxes_obs)))
        n_tbox, _, _ = ax[0].hist(cent_tboxes_obs,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(cent_tboxes_obs)))    
        ax[0].grid()
        ax[0].set(ylabel='Freq.',title=title)
        ax[0].legend()

        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax[1].scatter(bin_centers, get_ratio(n_pbox,n_tbox), label='PBox Clusters',marker='_',color='red',s=50)
        ax[1].axhline(1,ls='--',color='green',alpha=0.5)
        ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
        ax[1].set(xlabel=xlab,ylabel='Ratio')
        ax[1].grid()
        f.subplots_adjust(hspace=0)
        if log:
            ax[0].set(yscale='log')
            f.savefig(cent_save_loc + f'/central_cluster_unmatched_boxes_eta{eta_4}_{observable_name}_log.png')
        else:
            ax[0].set(ylabel='Freq. Density')
            f.savefig(cent_save_loc + f'/central_cluster_unmatched_boxes_eta{eta_4}_{observable_name}.png')
        plt.close()

    phys_plots_eta_bins_unmatched([-1.4,0.0,0.0,1.4], 'energy', save_loc)
    phys_plots_eta_bins_unmatched([-1.4,0.0,0.0,1.4], 'eta', save_loc)
    phys_plots_eta_bins_unmatched([-1.4,0.0,0.0,1.4], 'eT', save_loc)
    
    phys_plots_eta_bins_unmatched([-3.2,-1.4,1.4,3.2], 'energy', save_loc)
    phys_plots_eta_bins_unmatched([-3.2,-1.4,1.4,3.2], 'eT', save_loc)
    phys_plots_eta_bins_unmatched([-3.2,-1.4,1.4,3.2], 'eta', save_loc)

    phys_plots_eta_bins_unmatched([-5.0,-3.2,3.2,5.0], 'eT', save_loc)
    phys_plots_eta_bins_unmatched([-5.0,-3.2,3.2,5.0], 'energy', save_loc)
    phys_plots_eta_bins_unmatched([-5.0,-3.2,3.2,5.0], 'eta', save_loc)
    phys_plots_eta_bins_unmatched([-5.0,-3.2,3.2,5.0], 'phi', save_loc)



    def phys_box_plots_eta_bins(eta_region, observable_name, save_loc):
        cent_save_loc = save_loc + "/central/"
        if not os.path.exists(cent_save_loc):
            os.makedirs(cent_save_loc)
        
        eta_1,eta_2,eta_3,eta_4 = eta_region

        # clus_eta_mask = (total_clus_etas > eta_min) & (total_clus_etas < eta_max)
        clus_eta_mask = ((total_clus_etas > eta_1) & (total_clus_etas < eta_2) )| ((total_clus_etas > eta_3) & (total_clus_etas < eta_4))
        tbox_eta_mask = ((total_tru_etas > eta_1) & (total_tru_etas < eta_2)) | ((total_tru_etas > eta_3) & (total_tru_etas < eta_4))
        pbox_eta_mask = ((total_pred_etas > eta_1) & (total_pred_etas < eta_2)) | ((total_pred_etas > eta_3) & (total_pred_etas < eta_4))

        if observable_name=='energy':
            cent_clus_obs = total_clus_energies[clus_eta_mask]/1000
            cent_tboxes_obs = total_tru_energies[tbox_eta_mask]/1000
            cent_pboxes_obs = total_pred_energies[pbox_eta_mask]/1000
            title = f'Cluster/Box Energies in central [{eta_3},{eta_4}] region'
            xlab = f'Cluster energy (GeV)'
            
        elif observable_name=='eta':
            cent_clus_obs = total_clus_etas[clus_eta_mask]
            cent_tboxes_obs = total_tru_etas[tbox_eta_mask]
            cent_pboxes_obs = total_pred_etas[pbox_eta_mask]
            title = f'Cluster/Box eta in central [{eta_3},{eta_4}] region'
            xlab = f'eta'
            
        elif observable_name=='phi':
            cent_clus_obs = total_clus_phis[clus_eta_mask]
            cent_tboxes_obs = total_tru_phis[tbox_eta_mask]
            cent_pboxes_obs = total_pred_phis[pbox_eta_mask]
            title = f'Cluster/Box phi in central [{eta_3},{eta_4}] region'
            xlab = f'phi'
            
        elif observable_name=='eT':
            cent_clus_obs = total_clus_eT[clus_eta_mask]/1000
            cent_tboxes_obs = total_tru_eT[tbox_eta_mask]/1000
            cent_pboxes_obs = total_pred_eT[pbox_eta_mask]/1000
            title = f'Cluster/Box Transverse energy in central [{eta_3},{eta_4}] region'
            xlab = f'$E_T$'

        f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        n_pbox, bins, _ = ax[0].hist(cent_pboxes_obs,bins=75,density=(not log),histtype='step',color='red',label='Predicted Boxes ({})'.format(len(cent_tboxes_obs)))
        n_clus, _, _ = ax[0].hist(cent_clus_obs,bins=bins,density=(not log),histtype='step',color='tab:blue',label='TopoCl >5GeV ({})'.format(len(cent_clus_obs)))
        n_tbox, _, _ = ax[0].hist(cent_tboxes_obs,bins=bins,density=(not log),histtype='step',color='green',label='Truth Boxes ({})'.format(len(cent_pboxes_obs)))
        ax[0].grid()
        ax[0].set(ylabel='Freq.',title=title)
        ax[0].legend()

        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
        ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
        ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
        ax[1].set(xlabel=xlab,ylabel='Ratio')
        ax[1].grid()
        f.subplots_adjust(hspace=0)
        if log:
            ax[0].set(yscale='log')
            f.savefig(cent_save_loc + f'/central_boxes_eta{eta_4}_{observable_name}_log.png')
        else:
            ax[0].set(ylabel='Freq. Density')
            f.savefig(cent_save_loc + f'/central_boxes_eta{eta_4}_{observable_name}.png')
        plt.close()
    

    phys_box_plots_eta_bins([-1.4,0.0,0.0,1.4], 'energy', save_loc)
    phys_box_plots_eta_bins([-1.4,0.0,0.0,1.4], 'eT', save_loc)
    phys_box_plots_eta_bins([-1.4,0.0,0.0,1.4], 'eta', save_loc)
    phys_box_plots_eta_bins([-1.4,0.0,0.0,1.4], 'phi', save_loc)

    phys_box_plots_eta_bins([-3.2,-1.4,1.4,3.2], 'energy', save_loc)
    phys_box_plots_eta_bins([-3.2,-1.4,1.4,3.2], 'eT', save_loc)
    phys_box_plots_eta_bins([-3.2,-1.4,1.4,3.2], 'eta', save_loc)
    phys_box_plots_eta_bins([-3.2,-1.4,1.4,3.2], 'phi', save_loc)

    #####################################################################################################################################
    #Number of clusters/tboxes/pboxes in the event
    num_clusts = load_object(folder_containing_lists + '/n_clusters.pkl')
    num_tboxes = load_object("/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD_50k5_mu_20e/box_metrics" + '/n_truth.pkl')
    num_pboxes = load_object("/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD_50k5_mu_20e/box_metrics" + '/n_preds.pkl')
    # num_tboxes = load_object(folder_containing_lists + '/n_tboxes.pkl')
    # num_pboxes = load_object(folder_containing_lists + '/n_pboxes.pkl')

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_clus, bins, _ = ax[0].hist(num_clusts,bins=50,density=(not log),histtype='step',color='tab:blue',label='TopoCl(>5GeV): {:.2f}+-{:.1f}'.format(np.mean(num_clusts),np.std(num_clusts)))
    n_pbox, _, _ = ax[0].hist(num_pboxes,bins=bins,density=(not log),histtype='step',color='red',label='Preds: {:.2f}+-{:.1f}'.format(np.mean(num_pboxes),np.std(num_pboxes)))
    n_tbox, _, _ = ax[0].hist(num_tboxes,bins=bins,density=(not log),histtype='step',color='green',label='Truth: {:.2f}+-{:.1f}'.format(np.mean(num_tboxes),np.std(num_tboxes)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Number of clusters/boxes per event')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)

    ax[1].set(xlabel="# Clusters",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + '/total_event_n_clusters_log.png')
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + '/total_event_n_clusters.png')
    plt.close()








    return 




folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD_50k5_mu_20e/phys_metrics/"
save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_plots/SSD_50k5_mu_20e/"
if __name__=="__main__":
    print('Making physics plots')
    make_phys_plots(folder_to_look_in,save_at)
    make_phys_plots(folder_to_look_in,save_at,log=False)
    print('Completed physics plots\n')


