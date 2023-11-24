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

import mplhep as hep
hep.style.use(hep.style.ATLAS)


def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)

def get_ratio(numer,denom):
    result = np.zeros_like(numer, dtype=float)
    non_zero_indices = denom != 0
    # Perform element-wise division, handling zeros in the denominator
    result[non_zero_indices] = numer[non_zero_indices] / denom[non_zero_indices]
    return result





def make_phys_plots2(
    physics_folder,
    topo_folder,
    save_folder,
    mode = 'total',
    log = True,
    image_format = 'png',
):
    save_loc = save_folder + f"/new_phys/{mode}/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    save_loc1 = save_folder + f"/new_phys/{mode}/energy/"
    if not os.path.exists(save_loc1):
        os.makedirs(save_loc1)

    #####################################################################################################################################
    #Plot 1, the energies of our boxes and the true clusters

    total_clus_energies = np.concatenate(load_object(physics_folder + '/topocl_energies.pkl'))
    total_cl_cell_energies = np.concatenate(load_object(topo_folder + '/cl_cell_energies.pkl'))
    total_pred_energies = np.concatenate(load_object(physics_folder + '/pboxes_energies.pkl'))
    total_tru_energies = np.concatenate(load_object(physics_folder + '/tboxes_energies.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_pred_energies/1000,bins=100,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes ({})'.format(len(total_pred_energies)))
    n_clus, _, _ = ax[0].hist(total_clus_energies/1000,bins=bins,density=(not log),histtype='step',color='tab:blue',lw=1.5,label='TopoCl >5GeV ({})'.format(len(total_clus_energies)))
    n_cl_cell, _, _ = ax[0].hist(total_cl_cell_energies/1000,bins=bins,density=(not log),histtype='step',color='mediumpurple',lw=1.5,label='cl_cell >5GeV ({})'.format(len(total_cl_cell_energies)))
    n_tbox, _, _ = ax[0].hist(total_tru_energies/1000,bins=bins,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes ({})'.format(len(total_tru_energies)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Cluster/Box Energies')
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.5, 0.45),fontsize="medium")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_cl_cell,n_clus), label='cl_cell',marker='_',color='mediumpurple',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
    ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc1 + f'/total_boxes_energy_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc1 + f'/total_boxes_energy.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()



    #####################################################################################################################################
    #Plot 2, the etas of our boxes and the true clusters

    save_loc2 = save_folder + f"/new_phys/{mode}/eta/"
    if not os.path.exists(save_loc2):
        os.makedirs(save_loc2)

    #total
    total_clus_etas = np.concatenate(load_object(physics_folder + '/topocl_etas.pkl'))
    total_cl_cell_etas = np.concatenate(load_object(topo_folder + '/cl_cell_eta.pkl'))
    total_tru_etas = np.concatenate(load_object(physics_folder + '/tboxes_eta.pkl'))
    total_pred_etas = np.concatenate(load_object(physics_folder + '/pboxes_eta.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_clus, bins, _ = ax[0].hist(total_clus_etas,bins=50,density=(not log),histtype='step',color='tab:blue',lw=1.5,label='TopoCl >5GeV ({})'.format(len(total_clus_etas)))
    n_pbox, _, _ = ax[0].hist(total_pred_etas,bins=bins,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes ({})'.format(len(total_pred_etas)))
    n_tbox, _, _ = ax[0].hist(total_tru_etas,bins=bins,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes ({})'.format(len(total_tru_etas)))
    n_cl_cell, _, _ = ax[0].hist(total_cl_cell_etas,bins=bins,density=(not log),histtype='step',color='mediumpurple',lw=1.5,label='cl_cell ({})'.format(len(total_cl_cell_etas)))
    ax[0].grid()
    ax[0].set_title('Cluster/Box Eta',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.57, 0.68),fontsize="small")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_cl_cell,n_clus), label='cl_cell',marker='_',color='mediumpurple',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)

    ax[1].set(xlabel="Cluster Eta",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc2 + f'/total_cluster_boxes_eta_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc2 + f'/total_cluster_boxes_eta.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()



    #####################################################################################################################################
    #Plot 3, the phis of our boxes and the true clusters

    save_loc3 = save_folder + f"/new_phys/{mode}/phi/"
    if not os.path.exists(save_loc3):
        os.makedirs(save_loc3)

    #total
    total_clus_phis = np.concatenate(load_object(physics_folder + '/topocl_phis.pkl'))
    total_cl_cell_phis = np.concatenate(load_object(topo_folder + '/cl_cell_phi.pkl'))
    total_tru_phis = np.concatenate(load_object(physics_folder + '/tboxes_phi.pkl'))
    total_pred_phis = np.concatenate(load_object(physics_folder + '/pboxes_phi.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_clus, bins, _ = ax[0].hist(total_clus_phis,bins=50,density=(not log),histtype='step',color='tab:blue',lw=1.5,label='TopoCl >5GeV ({})'.format(len(total_clus_phis)))
    n_cl_cell, _, _ = ax[0].hist(total_cl_cell_phis,bins=50,density=(not log),histtype='step',color='mediumpurple',lw=1.5,label='cl_cell >5GeV ({})'.format(len(total_cl_cell_phis)))
    n_pbox, _, _ = ax[0].hist(total_pred_phis,bins=bins,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes ({})'.format(len(total_pred_phis)))
    n_tbox, _, _ = ax[0].hist(total_tru_phis,bins=bins,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes ({})'.format(len(total_tru_phis)))
    ax[0].grid()
    ax[0].set_title('Cluster/Box Phi',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.57, 0.65),fontsize="small")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_cl_cell,n_clus), label='cl_cell',marker='_',color='mediumpurple',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)

    ax[1].set(xlabel="Cluster Phi",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc3 + f'/total_cluster_boxes_phi_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc3 + f'/total_cluster_boxes_phi.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    #####################################################################################################################################
    #Plot 4, the number of cells in our boxes and the true clusters

    save_loc4 = save_folder + f"/new_phys/{mode}/n_cells/"
    if not os.path.exists(save_loc4):
        os.makedirs(save_loc4)

    #total
    total_clus_n_cells = np.concatenate(load_object(physics_folder + '/topocl_n_cells.pkl'))
    total_cl_cell_n_cells = np.concatenate(load_object(topo_folder + '/cl_cell_n_cells.pkl'))
    total_pred_n_cells = np.concatenate(load_object(physics_folder + '/tboxes_n_cells.pkl'))
    total_tru_n_cells = np.concatenate(load_object(physics_folder + '/pboxes_n_cells.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_pred_n_cells,bins=100,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes ({})'.format(len(total_pred_n_cells)))
    n_clus, _, _ = ax[0].hist(total_clus_n_cells,bins=bins,density=(not log),histtype='step',color='tab:blue',lw=1.5,label='TopoCl >5GeV ({})'.format(len(total_clus_n_cells)))
    n_cl_cell, _, _ = ax[0].hist(total_cl_cell_n_cells,bins=bins,density=(not log),histtype='step',color='mediumpurple',lw=1.5,label='cl_cell >5GeV ({})'.format(len(total_cl_cell_n_cells)))
    n_tbox, _, _ = ax[0].hist(total_tru_n_cells,bins=bins,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes ({})'.format(len(total_tru_n_cells)))
    ax[0].grid()
    ax[0].set_title('Number of cells in cluster/box',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.57, 0.65),fontsize="small")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_cl_cell,n_clus), label='cl_cell',marker='_',color='mediumpurple',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)

    ax[1].set(xlabel="Number of cells per cluster/box",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc4 + f'/total_boxes_n_cells_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc4 + f'/total_boxes_n_cells.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    #####################################################################################################################################
    #Plot 5, the transverse energy plots
    #total
    total_clus_eT = total_clus_energies/np.cosh(total_clus_etas)
    total_cl_cell_eT = np.concatenate(load_object(topo_folder + '/cl_cell_eT.pkl'))
    total_pred_eT = np.concatenate(load_object(physics_folder + '/pboxes_eT.pkl'))
    total_tru_eT = np.concatenate(load_object(physics_folder + '/tboxes_eT.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_pred_eT/1000,bins=100,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes ({})'.format(len(total_pred_energies)))
    n_clus, _, _ = ax[0].hist(total_clus_eT/1000,bins=bins,density=(not log),histtype='step',color='tab:blue',lw=1.5,label='TopoCl >5GeV ({})'.format(len(total_clus_energies)))
    n_cl_cell, _, _ = ax[0].hist(total_cl_cell_eT/1000,bins=bins,density=(not log),histtype='step',color='mediumpurple',lw=1.5,label='cl_cell >5GeV ({})'.format(len(total_cl_cell_eT)))
    n_tbox, _, _ = ax[0].hist(total_tru_eT/1000,bins=bins,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes ({})'.format(len(total_tru_energies)))
    ax[0].grid()
    ax[0].set_title('Transverse Energy',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.4, 0.5),fontsize="small")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_cl_cell,n_clus), label='cl_cell',marker='_',color='mediumpurple',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
    ax[1].set(xlabel="Cluster $E_T$ (GeV)",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc1 + f'/total_boxes_eT_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc1 + f'/total_boxes_eT.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()





    #####################################################################################################################################
    #Plot 6, the cluster/box noise

    save_loc5 = save_folder + f"/new_phys/{mode}/sig/"
    if not os.path.exists(save_loc5):
        os.makedirs(save_loc5)

    total_cl_cell_noise = np.concatenate(load_object(topo_folder + '/cl_cell_noise.pkl'))
    total_tru_noise = np.concatenate(load_object(physics_folder + '/tboxes_noise.pkl'))
    total_pred_noise = np.concatenate(load_object(physics_folder + '/pboxes_noise.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_pred_noise/1000,bins=100,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes ({})'.format(len(total_pred_noise)))
    n_cl_cell, _, _ = ax[0].hist(total_cl_cell_noise/1000,bins=bins,density=(not log),histtype='step',color='mediumpurple',lw=1.5,label='cl_cell >5GeV ({})'.format(len(total_cl_cell_noise)))
    n_tbox, _, _ = ax[0].hist(total_tru_noise/1000,bins=bins,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes ({})'.format(len(total_tru_noise)))
    ax[0].grid()
    ax[0].set_title('Cluster/Box Noise',fontsize=16,fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.48, 0.55),fontsize="medium")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_cl_cell), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_cl_cell), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='mediumpurple',alpha=0.5)
    ax[1].set(xlabel="Cluster Noise (GeV)",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc5 + f'/total_boxes_noise_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc5 + f'/total_boxes_noise.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    #####################################################################################################################################
    #Plot 7, the cluster/box significance

    total_cl_cell_sig = np.concatenate(load_object(topo_folder + '/cl_cell_significance.pkl'))
    total_tru_sig = np.concatenate(load_object(physics_folder + '/tboxes_significance.pkl'))
    total_pred_sig = np.concatenate(load_object(physics_folder + '/pboxes_significance.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_tbox, bins, _ = ax[0].hist(total_tru_sig,bins=100,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes ({})'.format(len(total_tru_sig)))
    n_cl_cell, _, _ = ax[0].hist(total_cl_cell_sig,bins=bins,density=(not log),histtype='step',color='mediumpurple',lw=1.5,label='cl_cell >5GeV ({})'.format(len(total_cl_cell_sig)))
    n_pbox, _, _ = ax[0].hist(total_pred_sig,bins=bins,density=(not log),histtype='step',color='red',lw=1.5,label='Pred Boxes ({})'.format(len(total_pred_sig)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title=r'cluster sig. $\zeta_{clus}^{EM} = \frac{E_{clus}^{EM}}{\sqrt{\sum (\sigma_{noise,cell,i}^{EM})^2 } }$')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_cl_cell), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_cl_cell), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='mediumpurple',alpha=0.5)
    ax[1].set(xlabel="Cluster Significance",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc5 + f'/total_boxes_sig_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc5 + f'/total_boxes_sig.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    #repeat but different x-axis
    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_clus, bins, _ = ax[0].hist(total_cl_cell_sig,bins=50,density=(not log),histtype='step',color='mediumpurple',lw=1.5,label='cl_cell >5GeV ({})'.format(len(total_cl_cell_sig)))
    n_pbox, _, _ = ax[0].hist(total_pred_sig,bins=bins,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes ({})'.format(len(total_pred_sig)))
    n_tbox, _, _ = ax[0].hist(total_tru_sig,bins=bins,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes ({})'.format(len(total_tru_sig)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title=r'cluster sig. $\zeta_{clus}^{EM} = \frac{E_{clus}^{EM}}{\sqrt{\sum (\sigma_{noise,cell,i}^{EM})^2 } }$')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='mediumpurple',alpha=0.5)

    ax[1].set(xlabel="Cluster significance",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc5 + f'/total_cluster_sig_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc5 + f'/total_cluster_sig.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    #####################################################################################################################################
    #Plot 8, the cluster/box negative frac: sum(negE) / sum(posE)
    #bug, divide by 0 encountered when there are no positive cells in a box
    total_cl_cell_neg_frac = np.concatenate(load_object(topo_folder + '/cl_cell_neg_frac.pkl'))
    total_tru_neg_frac = np.concatenate(load_object(physics_folder + '/tboxes_neg_frac.pkl'))
    total_pred_neg_frac = np.concatenate(load_object(physics_folder + '/pboxes_neg_frac.pkl'))
    # print(min(total_pred_neg_frac),max(total_pred_neg_frac))
    # print(min(total_tru_neg_frac),max(total_tru_neg_frac))
    # print(min(total_cl_cell_neg_frac),max(total_cl_cell_neg_frac))
    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_pred_neg_frac,bins=100,density=(not log),histtype='step',color='red',label='Preds {:.2f}+-{:.1f}'.format(np.mean(total_pred_neg_frac),np.std(total_pred_neg_frac)))
    n_cl_cell, _, _ = ax[0].hist(total_cl_cell_neg_frac,bins=bins,density=(not log),histtype='step',color='mediumpurple',label='cl_cell >5GeV {:.2f}+-{:.1f}'.format(np.mean(total_cl_cell_neg_frac),np.std(total_cl_cell_neg_frac)))
    n_tbox, _, _ = ax[0].hist(total_tru_neg_frac,bins=bins,density=(not log),histtype='step',color='green',label='Truth {:.2f}+-{:.1f}'.format(np.mean(total_tru_neg_frac),np.std(total_tru_neg_frac)))
    ax[0].grid()
    ax[0].set_title('Cluster/Box Negative fraction sum(negE)/sum(posE)',fontsize=16,fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_cl_cell), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_cl_cell), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='mediumpurple',alpha=0.5)
    ax[1].set(xlabel="Cluster Negative frac.",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc1 + f'/total_boxes_neg_frac_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc1 + f'/total_boxes_neg_frac.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    #####################################################################################################################################
    #Plot 9, the cluster/box fraction energy one cell

    total_cl_cell_max_frac = np.concatenate(load_object(topo_folder + '/cl_cell_max_frac.pkl'))
    total_tru_max_frac = np.concatenate(load_object(physics_folder + '/tboxes_max_frac.pkl'))
    total_pred_max_frac = np.concatenate(load_object(physics_folder + '/pboxes_max_frac.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]}) 
    n_pbox, bins, _ = ax[0].hist(total_pred_max_frac,bins=100,density=(not log),histtype='step',color='red',label='Preds {:.2f}+-{:.1f}'.format(np.mean(total_pred_max_frac),np.std(total_pred_max_frac)))
    n_cl_cell, _, _ = ax[0].hist(total_cl_cell_max_frac,bins=bins,density=(not log),histtype='step',color='mediumpurple',label='cl_cell >5GeV {:.2f}+-{:.1f}'.format(np.mean(total_cl_cell_max_frac),np.std(total_cl_cell_max_frac)))
    n_tbox, _, _ = ax[0].hist(total_tru_max_frac,bins=bins,density=(not log),histtype='step',color='green',label='Truth {:.2f}+-{:.1f}'.format(np.mean(total_tru_max_frac),np.std(total_tru_max_frac)))
    ax[0].grid()
    ax[0].set_title('Cluster/Box Max fraction energy one cell',fontsize=16,fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_cl_cell), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_cl_cell), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='mediumpurple',alpha=0.5)
    ax[1].set(xlabel="Cluster max frac. 1 cell",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc1 + f'/total_boxes_max_frac_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc1 + f'/total_boxes_max_frac.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()




    #####################################################################################################################################
    #####################################################################################################################################
    #####################################################################################################################################
    #Plot10 how many truth boxes per truth box (as a result of merging)

    save_loc6 = save_folder + f"/new_phys/{mode}/truth_box_val/"
    if not os.path.exists(save_loc6):
        os.makedirs(save_loc6)



    n_cl_per_box = np.concatenate(load_object(topo_folder + '/n_clus_per_box.pkl'))

    f,ax = plt.subplots(1,1,figsize=(8, 6)) 
    n_pbox, bins, _ = ax.hist(n_cl_per_box,bins=50,density=(not log),histtype='step',color='green',lw=1.5,label='Truth {:.2f}+-{:.1f}'.format(np.mean(n_cl_per_box),np.std(n_cl_per_box)))
    ax.grid()
    ax.set_title(f'Number of clusters per truth box ({len(n_cl_per_box)} boxes)',fontsize=16,fontfamily="TeX Gyre Heros")
    ax.set(ylabel='Freq.',xlabel='# clusters in box',title=f'Number of clusters per truth box ({len(n_cl_per_box)} boxes)')
    ax.legend()
    hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
    if log:
        ax.set(yscale='log')
        f.savefig(save_loc6 + f'/total_n_cl_per_box_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax.set(ylabel='Freq. Density')
        f.savefig(save_loc6 + f'/total_n_cl_per_box.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()




    #####################################################################################################################################
    #Plot 11, the total energy inside truth boxes (sum of multiple clusters)

    total_merged_clus_E = np.concatenate(load_object(topo_folder + '/merged_clus_E.pkl'))
    total_merged_cl_cell_E = np.concatenate(load_object(topo_folder + '/merged_cl_cell_E.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]}) 
    n_tbox, bins, _ = ax[0].hist(total_tru_energies/1000,bins=100,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes ({:.2f}+-{:.0f})'.format(np.mean(total_tru_energies/1000),np.std(total_tru_energies/1000)))
    n_merged_cl_cell, bins, _ = ax[0].hist(total_merged_cl_cell_E/1000,bins=bins,density=(not log),histtype='step',color='pink',lw=1.5,label='Merged cl_cell >5GeV ({:.2f}+-{:.0f})'.format(np.mean(total_merged_cl_cell_E/1000),np.std(total_merged_cl_cell_E/1000)))
    n_merged_cl, _, _ = ax[0].hist(total_merged_clus_E/1000,bins=bins,density=(not log),histtype='step',color='fuchsia',lw=1.5,label='Merged clust >5GeV ({:.2f}+-{:.0f})'.format(np.mean(total_merged_clus_E/1000),np.std(total_merged_clus_E/1000)))
    n_clus, _, _ = ax[0].hist(total_clus_energies/1000,bins=bins,density=(not log),histtype='step',color='tab:blue',lw=1.5,label='TopoCl >5GeV ({:.2f}+-{:.0f})'.format(np.mean(total_clus_energies/1000),np.std(total_clus_energies/1000)))
    n_cl_cell, _, _ = ax[0].hist(total_cl_cell_energies/1000,bins=bins,density=(not log),histtype='step',color='mediumpurple',lw=1.5,label='cl_cell >5GeV ({:.2f}+-{:.0f})'.format(np.mean(total_cl_cell_energies/1000),np.std(total_cl_cell_energies/1000)))
    ax[0].grid()
    ax[0].set_title('Cluster/Box Energy (incl. merged clusters)',fontsize=16,fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.45, 0.5),fontsize="small")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_cl_cell,n_clus), label='cl_cell',marker='_',color='mediumpurple',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_merged_cl,n_clus), label='Merged TopoCl',marker='_',color='fuchsia',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_merged_cl_cell,n_clus), label='Merged cl_cell',marker='_',color='pink',s=50)

    ax[1].axhline(1,ls='--',color='mediumpurple',alpha=0.5)
    ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc6 + f'/total_merged_clus_E_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc6 + f'/total_merged_clus_E.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()







    #####################################################################################################################################
    #plot 12, energy but only including cells >2sig

    save_loc7 = save_folder + f"/new_phys/{mode}/two_sig/"
    if not os.path.exists(save_loc7):
        os.makedirs(save_loc7)

    total_pred_energies_2sig = np.concatenate(load_object(physics_folder + '/pboxes_energies_2sig.pkl'))
    total_tru_energies_2sig = np.concatenate(load_object(physics_folder + '/tboxes_energies_2sig.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_pred_energies_2sig/1000,bins=100,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes ({}) cell sig. >2'.format(len(total_pred_energies)))
    n_clus, _, _ = ax[0].hist(total_clus_energies/1000,bins=bins,density=(not log),histtype='step',color='tab:blue',lw=1.5,label='TopoCl >5GeV ({})'.format(len(total_clus_energies)))
    n_cl_cell, _, _ = ax[0].hist(total_cl_cell_energies/1000,bins=bins,density=(not log),histtype='step',color='mediumpurple',lw=1.5,label='cl_cell >5GeV ({})'.format(len(total_cl_cell_energies)))
    n_tbox, _, _ = ax[0].hist(total_tru_energies_2sig/1000,bins=bins,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes ({}) cell sig. >2'.format(len(total_tru_energies)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Cluster/Box Energies')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_cl_cell,n_clus), label='cl_cell',marker='_',color='mediumpurple',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
    ax[1].set(xlabel="Cluster Energy (GeV)",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc7 + f'/total_boxes_energy_2sig_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc7 + f'/total_boxes_energy_2sig.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    #####################################################################################################################################
    #plot 13, e transverse but only including cells >2sig

    total_pred_et_2sig = np.concatenate(load_object(physics_folder + '/pboxes_eT_2sig.pkl'))
    total_tru_et_2sig = np.concatenate(load_object(physics_folder + '/tboxes_eT_2sig.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_pred_et_2sig/1000,bins=100,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes ({}) cell sig. >2'.format(len(total_pred_energies)))
    n_clus, _, _ = ax[0].hist(total_clus_eT/1000,bins=bins,density=(not log),histtype='step',color='tab:blue',lw=1.5,label='TopoCl >5GeV ({})'.format(len(total_clus_energies)))
    n_cl_cell, _, _ = ax[0].hist(total_cl_cell_eT/1000,bins=bins,density=(not log),histtype='step',color='mediumpurple',lw=1.5,label='cl_cell >5GeV ({})'.format(len(total_cl_cell_energies)))
    n_tbox, _, _ = ax[0].hist(total_tru_et_2sig/1000,bins=bins,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes ({}) cell sig. >2'.format(len(total_tru_energies)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Cluster/Box Transverse Energy')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_cl_cell,n_clus), label='cl_cell',marker='_',color='mediumpurple',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
    ax[1].set(xlabel="Cluster $E_T$ (GeV)",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc7 + f'/total_boxes_eT_2sig_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc7 + f'/total_boxes_eT_2sig.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    #####################################################################################################################################
    #plot 14, n cells but only including cells >2sig

    total_pred_n_cells_2sig = np.concatenate(load_object(physics_folder + '/pboxes_n_cells_2sig.pkl'))
    total_tru_n_cells_2sig = np.concatenate(load_object(physics_folder + '/tboxes_n_cells_2sig.pkl'))

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(total_pred_n_cells_2sig,bins=100,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes ({}) |cell sig.| >2'.format(len(total_pred_n_cells_2sig)))
    n_clus, _, _ = ax[0].hist(total_clus_n_cells,bins=bins,density=(not log),histtype='step',color='tab:blue',lw=1.5,label='TopoCl >5GeV ({})'.format(len(total_clus_n_cells)))
    # n_cl_cell, _, _ = ax[0].hist(total_cl_cell_n_cells,bins=bins,density=(not log),histtype='step',color='mediumpurple',lw=1.5,label='cl_cell >5GeV ({})'.format(len(total_cl_cell_n_cells)))
    n_tbox, _, _ = ax[0].hist(total_tru_n_cells_2sig,bins=bins,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes ({}) |cell sig.| >2'.format(len(total_tru_n_cells_2sig)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Number of cells per Cluster/Box ')
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.4, 0.55),fontsize="small")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox Clusters',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox Clusters',marker='_',color='red',s=50)
    # ax[1].scatter(bin_centers, get_ratio(n_cl_cell,n_clus), label='cl_cell',marker='_',color='mediumpurple',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
    ax[1].set(xlabel="Num. cells in cluster",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc7 + f'/total_boxes_n_cells_2sig_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc7 + f'/total_boxes_n_cells_2sig.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()








    #####################################################################################################################################
    #####################################################################################################################################
    #####################################################################################################################################
    #Number of clusters/tboxes/pboxes in the event
    num_clusts = load_object(physics_folder + '/n_clusters.pkl')
    num_tboxes = load_object(physics_folder + '/num_tboxes.pkl')
    n_tboxes = load_object(physics_folder + '/n_tboxes.pkl')
    num_pboxes = load_object(physics_folder + '/num_pboxes.pkl')
    n_pboxes = load_object(physics_folder + '/n_pboxes.pkl')

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_clus, bins, _ = ax[0].hist(num_clusts,bins=50,density=(not log),histtype='step',color='tab:blue',label='TopoCl(>5GeV): {:.2f}+-{:.1f}'.format(np.mean(num_clusts),np.std(num_clusts)))
    n_pbox, _, _ = ax[0].hist(num_pboxes,bins=bins,density=(not log),histtype='step',color='red',label='Preds: {:.2f}+-{:.1f} (removed no cell boxes)'.format(np.mean(num_pboxes),np.std(num_pboxes)))
    n_tbox, _, _ = ax[0].hist(num_tboxes,bins=bins,density=(not log),histtype='step',color='green',label='Truth: {:.2f}+-{:.1f}'.format(np.mean(num_tboxes),np.std(num_tboxes)))
    n_pbox2, _, _ = ax[0].hist(n_pboxes,bins=bins,density=(not log),histtype='step',color='orange',label='Preds2: {:.2f}+-{:.1f}'.format(np.mean(n_pboxes),np.std(n_pboxes)))
    n_tbox2, _, _ = ax[0].hist(n_tboxes,bins=bins,density=(not log),histtype='step',color='limegreen',label='Truth2: {:.2f}+-{:.1f}'.format(np.mean(n_tboxes),np.std(n_tboxes)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Number of clusters/boxes per event')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(n_tbox,n_clus), label='TBox ',marker='_',color='green',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox,n_clus), label='PBox ',marker='_',color='red',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_tbox2,n_clus), label='TBox2 ',marker='_',color='limegreen',s=50)
    ax[1].scatter(bin_centers, get_ratio(n_pbox2,n_clus), label='PBox2 ',marker='_',color='orange',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)

    ax[1].set(xlabel="# Clusters",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log')
        f.savefig(save_loc + f'/total_event_n_clusters_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + f'/total_event_n_clusters.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    return 




physics_folder = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD1_50k5_mu_15e/new_phys_metrics/total/"
topocl_cell_folder = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD1_50k5_mu_15e/truth_box_eval/"
    
save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_plots/SSD1_50k5_mu_15e/"
if __name__=="__main__":
    print('Making physics plots')
    make_phys_plots2(physics_folder,topocl_cell_folder,save_at)
    print(1)
    make_phys_plots2(physics_folder,topocl_cell_folder,save_at,log=False)
    print(2)
    make_phys_plots2(physics_folder,topocl_cell_folder,save_at,image_format='pdf')
    print(3)
    make_phys_plots2(physics_folder,topocl_cell_folder,save_at,log=False,image_format='pdf')
    print('Completed physics plots\n')


