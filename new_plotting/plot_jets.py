import numpy as np 
import pandas as pd
import scipy
import math
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

def transform_angle(angle):
    # Maps angle to [-π, π]
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_ratio(numer,denom):
    result = np.zeros_like(numer, dtype=float)
    non_zero_indices = denom != 0
    # Perform element-wise division, handling zeros in the denominator
    result[non_zero_indices] = numer[non_zero_indices] / denom[non_zero_indices]
    return result



def clopper_pearson(x, n, alpha=0.05):
    """
    Estimate the confidence interval for a sampled Bernoulli random
    variable.
    `x` is the number of successes and `n` is the number trials (x <=
    n). `alpha` is the confidence level (i.e., the true probability is
    inside the confidence interval with probability 1-alpha). The
    function returns a `(low, high)` pair of numbers indicating the
    interval on the probability.
    https://root.cern.ch/doc/master/classTEfficiency.html#ae80c3189bac22b7ad15f57a1476ef75b
    """

    lo = scipy.stats.beta.ppf(alpha / 2, x, n - x + 1)
    hi = scipy.stats.beta.ppf(1 - alpha / 2, x + 1, n - x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi

def get_errorbars(success_array, total_array, alpha=0.05):
    """
    Function to calculate and return errorbars in matplotlib preferred format.
    Current usage of Clopper-Pearon may generalise later. Function currently
    returns interval.
    'success_array' is the count of each histogram bins after(!) cut applied
    'total_array' is the count of each histogram before trigger applied
    'alpha' is the confidence level
    Returns errors array to be used in ax.errorbars kwarg yerr
    """
    confidence_intervals = []
    
    lo, hi = np.vectorize(clopper_pearson)(success_array, total_array, alpha)
    
    confidence_intervals = np.array([lo, hi]).T
    
    zeros_mask = total_array == 0
    lower_error_bars = np.where(zeros_mask, lo, success_array/total_array - lo)
    upper_error_bars = np.where(zeros_mask, hi, hi - success_array/total_array)
    
    errors = np.array([lower_error_bars, upper_error_bars])
    
    return errors




def leading_jet_pt(list_of_jet_energies_in_event):
    try:
        return max(list_of_jet_energies_in_event)
    except ValueError:
        #Doesn't have enough (or any) jets, automatically lost in cut
        return np.nan

def nth_leading_jet_pt(list_of_jet_energies_in_event,n):
    try:
        return sorted(list_of_jet_energies_in_event,reverse=True)[n-1]
    except IndexError or ValueError:
        # Doesn't have enough (or any) jets, automatically lost in cut
        return np.nan







def make_jet_plots(
    folder_containing_lists,
    save_folder,
    log=False,
    image_format="png",
):
    save_loc = save_folder + "/jets/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    

    pt_files = ['esdjet_pts','fjet_pts','tboxjet_pts','pboxjet_pts']
    eta_files = ['esdjet_etas','fjet_etas','tboxjet_etas','pboxjet_etas']
    phi_files = ['esdjet_phis','fjet_phis','tboxjet_phis','pboxjet_phis']
    njet_files = ['n_esdjets','n_fjets','n_tboxjets','n_pboxjets']

    esdjets_pt = load_object(folder_containing_lists + pt_files[0] + '.pkl')
    fjets_pt = load_object(folder_containing_lists + pt_files[1] + '.pkl')
    tboxjets_pt = load_object(folder_containing_lists + pt_files[2] + '.pkl')
    pboxjets_pt = load_object(folder_containing_lists + pt_files[3] + '.pkl')

    #-----------------------------------------------------------------------------------------------------------------
    # Efficiency Jet Plots
    #-----------------------------------------------------------------------------------------------------------------
    eff_save_loc = save_loc + "eff"
    if not os.path.exists(eff_save_loc):
        os.makedirs(eff_save_loc)  

    esdj_lead_pt = np.array([leading_jet_pt(x) for x in esdjets_pt])
    fjj_lead_pt = np.array([leading_jet_pt(y) for y in fjets_pt])
    tbox_lead_pt = np.array([leading_jet_pt(x) for x in tboxjets_pt])
    pbox_lead_pt = np.array([leading_jet_pt(x) for x in pboxjets_pt])

    lead_jet_pt_cut = 250_000 # 400GeV


    #make a trigger decision based on ESD jet
    trig_decision_esd = np.argwhere(esdj_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_fjj = np.argwhere(fjj_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_tb = np.argwhere(tbox_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_pb = np.argwhere(pbox_lead_pt>lead_jet_pt_cut).T[0]

    bin_width = 10
    num_bins=40
    bins = [250+i*bin_width for i in range(num_bins+1)]

    f,ax = plt.subplots(3,1,figsize=(6.5,12))
    n_esd,bins,_ = ax[0].hist(esdj_lead_pt/1000,bins=bins,histtype='step',label='ESD Jet')
    ax[0].set_title(f'Before {lead_jet_pt_cut/1000:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(xlabel="Leading EMTopo (Offline) jet pT (GeV)",ylabel='Freq.')
    ax[0].legend()

    n2_esd,bins,_ = ax[1].hist(esdj_lead_pt[trig_decision_esd]/1000,bins=bins,histtype='step',label='ESD Jet')
    n2_fj,_,_ = ax[1].hist(esdj_lead_pt[trig_decision_fjj]/1000,bins=bins,histtype='step',label='All clus.')
    n2_tbox,_,_ = ax[1].hist(esdj_lead_pt[trig_decision_tb]/1000,bins=bins,histtype='step',label='TBox')
    n2_pbox,_,_ = ax[1].hist(esdj_lead_pt[trig_decision_pb]/1000,bins=bins,histtype='step',label='PBox')
    ax[1].axvline(x=lead_jet_pt_cut/1000,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {lead_jet_pt_cut/1000:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel="Leading EMTopo (Offline) jet pT (GeV)",ylabel='Freq.')
    ax[1].legend()

    ax[2].axvline(x=lead_jet_pt_cut/1000,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_esd,n_esd)
        step_err = get_errorbars(n2_esd,n_esd)
        fj_eff = get_ratio(n2_fj,n_esd)
        fj_err = get_errorbars(n2_fj,n_esd)
        tbox_eff = get_ratio(n2_tbox,n_esd)
        tbox_err = get_errorbars(n2_tbox,n_esd)
        pbox_eff = get_ratio(n2_pbox,n_esd)
        pbox_err = get_errorbars(n2_pbox,n_esd)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    if True:
        # ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='ESD',color='goldenrod')
        # ax[2].errorbar(bin_centers,fj_eff,xerr=bin_width/2,yerr=fj_err,elinewidth=0.4,marker='.',ls='none',label='All clus.')
        ax[2].errorbar(bin_centers,tbox_eff,xerr=bin_width/2,yerr=tbox_err,elinewidth=0.4,marker='.',ls='none',label='TBox',color='green')
        ax[2].errorbar(bin_centers,pbox_eff,xerr=bin_width/2,yerr=pbox_err,elinewidth=0.4,marker='.',ls='none',label='PBox',color='red')
    else:
        ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='ESD')
        ax[2].errorbar(bin_centers,fj_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='All clus.')
        ax[2].errorbar(bin_centers,tbox_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='TBox')
        ax[2].errorbar(bin_centers,pbox_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='PBox')
    ax[2].grid()
    ax[2].set(xlabel="Leading EMTopo (Offline) jet pT (GeV)",ylabel='Efficiency')
    # ax[2].xaxis.label.set_fontsize(18)
    # ax[2].yaxis.label.set_fontsize(18)
    # ax[2].legend(loc='lower right',fontsize=18)
    ax[2].legend(loc='lower right')
    hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    if log:
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        f.savefig(eff_save_loc + f'/eff_plot_leading{lead_jet_pt_cut/1000:.0f}GeV_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        f.savefig(eff_save_loc + f'/eff_plot_leading{lead_jet_pt_cut/1000:.0f}GeV.{image_format}',dpi=400,format=image_format,bbox_inches="tight")




    #-----------------------------------------------------------------------------------------------------------------
    # Nth Leading Jet
    nth_jet = 2
    nth_lead_jet_pt_cut = 450_000 # 400GeV


    esdj_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in esdjets_pt])
    fjj_nlead_pt = np.array([nth_leading_jet_pt(y,nth_jet) for y in fjets_pt])
    tbox_nlead_pt = np.array([nth_leading_jet_pt(z,nth_jet) for z in tboxjets_pt])
    pbox_nlead_pt = np.array([nth_leading_jet_pt(z,nth_jet) for z in pboxjets_pt])
    #make a trigger decision based on ESD jet
    trigger_decision = np.argwhere(esdj_nlead_pt>nth_lead_jet_pt_cut).T[0]

    trig_decision_esd = np.argwhere(esdj_nlead_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_fjj = np.argwhere(fjj_nlead_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_tb = np.argwhere(tbox_nlead_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_pb = np.argwhere(pbox_nlead_pt>nth_lead_jet_pt_cut).T[0]

    bin_width = 25
    num_bins=25
    bins = [50+i*bin_width for i in range(num_bins+1)]

    f,ax = plt.subplots(3,1,figsize=(6.5,12))
    n_esd,bins,_ = ax[0].hist(esdj_nlead_pt/1000,bins=bins,histtype='step',label='ESD Jet')
    # n_fj,_,_ = ax[0].hist(fjj_nlead_pt,bins=bins,histtype='step',label='All clus.')
    # n_tbox,_,_ = ax[0].hist(tbox_nlead_pt,bins=bins,histtype='step',label='TBox')
    # n_pbox,_,_ = ax[0].hist(pbox_nlead_pt,bins=bins,histtype='step',label='PBox')
    ax[0].set_title(f'Before {nth_lead_jet_pt_cut/1000:.0f}GeV Cut')
    ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading EMTopo (Offline) jet pT (GeV)",ylabel='Freq.')
    ax[0].grid()
    ax[0].legend()

    n2_esd,bins,_ = ax[1].hist(esdj_nlead_pt[trig_decision_esd]/1000,bins=bins,histtype='step',label='ESD Jet')
    n2_fj,_,_ = ax[1].hist(esdj_nlead_pt[trig_decision_fjj]/1000,bins=bins,histtype='step',label='All clus.')
    n2_tbox,_,_ = ax[1].hist(esdj_nlead_pt[trig_decision_tb]/1000,bins=bins,histtype='step',label='TBox')
    n2_pbox,_,_ = ax[1].hist(esdj_nlead_pt[trig_decision_pb]/1000,bins=bins,histtype='step',label='PBox')
    ax[1].axvline(x=nth_lead_jet_pt_cut/1000,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {nth_lead_jet_pt_cut/1000:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading EMTopo (Offline) jet pT (GeV)",ylabel='Freq.')
    ax[1].grid()
    ax[1].legend()

    ax[2].axvline(x=nth_lead_jet_pt_cut/1000,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        #calculate efficiencies and errors
        step_eff = get_ratio(n2_esd,n_esd)
        step_err = get_errorbars(n2_esd,n_esd)

        fj_eff = get_ratio(n2_fj,n_esd)
        fj_err = get_errorbars(n2_fj,n_esd)

        tbox_eff = get_ratio(n2_tbox,n_esd)
        tbox_err = get_errorbars(n2_tbox,n_esd)

        pbox_eff = get_ratio(n2_pbox,n_esd)
        pbox_err = get_errorbars(n2_pbox,n_esd)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    errorbs = True
    if errorbs:
        # ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='ESD',color='goldenrod')
        # ax[2].errorbar(bin_centers,fj_eff,xerr=bin_width/2,yerr=fj_err,elinewidth=0.4,marker='.',ls='none',label='All clus.')
        ax[2].errorbar(bin_centers,tbox_eff,xerr=bin_width/2,yerr=tbox_err,elinewidth=0.4,marker='.',ls='none',label='TBox',color='green')
        ax[2].errorbar(bin_centers,pbox_eff,xerr=bin_width/2,yerr=pbox_err,elinewidth=0.4,marker='.',ls='none',label='PBox',color='red')
    else:
        ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='ESD')
        ax[2].errorbar(bin_centers,fj_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='All clus.')
        ax[2].errorbar(bin_centers,tbox_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='TBox')
        ax[2].errorbar(bin_centers,pbox_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='PBox')
    # ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[2].grid()
    ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading EMTopo (Offline) jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='lower right')
    hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    if log:
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        f.savefig(eff_save_loc + f'/eff_plot_{nth_jet}leading{nth_lead_jet_pt_cut/1000:.0f}GeV_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        f.savefig(eff_save_loc + f'/eff_plot_{nth_jet}leading{nth_lead_jet_pt_cut/1000:.0f}GeV.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


















    #-----------------------------------------------------------------------------------------------------------------
    # pT PLOTS
    #-----------------------------------------------------------------------------------------------------------------

    save_loc1 = save_loc + "/pT/"
    if not os.path.exists(save_loc1):
        os.makedirs(save_loc1)


    all_esdjets_pt = np.concatenate(load_object(folder_containing_lists + '/' + pt_files[0] + '.pkl'))
    all_fjets_pt = np.concatenate(load_object(folder_containing_lists + '/' + pt_files[1] + '.pkl'))
    all_tboxjets_pt = np.concatenate(load_object(folder_containing_lists + '/' + pt_files[2] + '.pkl'))
    all_pboxjets_pt = np.concatenate(load_object(folder_containing_lists + '/' + pt_files[3] + '.pkl'))

    pt_cut = 20_000 #20GeV
    hi_fjets_pt = all_fjets_pt[np.argwhere(all_fjets_pt>pt_cut)]
    hi_esdjets_pt = all_esdjets_pt[np.argwhere(all_esdjets_pt>pt_cut)]
    hi_tboxets_pt = all_tboxjets_pt[np.argwhere(all_tboxjets_pt>pt_cut)]
    hi_pboxets_pt = all_pboxjets_pt[np.argwhere(all_pboxjets_pt>pt_cut)]


    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_esd, bins, _ = ax[0].hist(hi_esdjets_pt/1000,bins=50,color='gold',histtype='step',density=(not log),lw=1.5,label=f'Offline Jets {len(hi_esdjets_pt)}')
    n_tbox, bins, _ = ax[0].hist(hi_tboxets_pt/1000,bins=bins,color='green',histtype='step',density=(not log),lw=1.5,label=f'TBox Jets {len(hi_tboxets_pt)}')
    # n_fj, bins, _ = ax[0].hist(hi_fjets_pt/1000,bins=bins,color='dodgerblue',histtype='step',density=(not log),lw=1.5,label=f'All clus. {len(hi_fjets_pt)}')
    n_pbox, bins, _ = ax[0].hist(hi_pboxets_pt/1000,bins=bins,color='red',histtype='step',density=(not log),lw=1.5,label=f'PBox Jets {len(hi_pboxets_pt)}')
    # ax[0].axvline(pt_cut,ls='--',color='red',label='pT cut')
    ax[0].set_title(f'{len(esdjets_pt)} Test Events, All Jets pT > {pt_cut/1000:.0f}GeV',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].grid()
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.57, 0.68),fontsize="medium")

    # ratios_fj = get_ratio(n_fj,n_esd) # np.where(n_esd != 0, n_fj / n_esd, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # ax[1].plot(bin_centers, ratios_fj, label='All clus.',color='dodgerblue',marker='_')
    ax[1].plot(bin_centers, get_ratio(n_tbox,n_esd), label='TBox Jets',color='green',marker='_')
    ax[1].plot(bin_centers, get_ratio(n_pbox,n_esd), label='PBox Jets',color='red',marker='_')
    ax[1].axhline(1,ls='--',color='gold')
    ax[1].set(xlabel="Jet pT (GeV)")
    ax[1].grid()
    # ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000)))
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    fig.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set_yscale('log')
        fig.savefig(save_loc1 + f'/jet_pt_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        fig.savefig(save_loc1 + f'/jet_pt.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()



    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_tbox, bins, _ = ax[0].hist(hi_tboxets_pt/1000,bins=50,color='green',histtype='step',density=(not log),lw=1.5,label=f'TBox Jets {len(hi_tboxets_pt)}')
    n_pbox, bins, _ = ax[0].hist(hi_pboxets_pt/1000,bins=bins,color='red',histtype='step',density=(not log),lw=1.5,label=f'PBox Jets {len(hi_pboxets_pt)}')
    n_esd, bins, _ = ax[0].hist(hi_esdjets_pt/1000,bins=bins,color='gold',histtype='step',density=(not log),lw=1.5,label=f'Offline Jets {len(hi_esdjets_pt)}')
    # n_fj, bins, _ = ax[0].hist(hi_fjets_pt/1000,bins=bins,color='dodgerblue',histtype='step',density=(not log),lw=1.5,label=f'All clus. {len(hi_fjets_pt)}')
    # ax[0].axvline(pt_cut,ls='--',color='red',label='pT cut')
    ax[0].set_title(f'{len(esdjets_pt)} Test Events, All Jets pT > {pt_cut/1000:.0f}GeV',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].grid()
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.57, 0.5),fontsize="medium")

    # ratios_fj = get_ratio(n_fj,n_esd) # np.where(n_esd != 0, n_fj / n_esd, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # ax[1].plot(bin_centers, ratios_fj, label='All clus.',color='dodgerblue',marker='_')
    ax[1].plot(bin_centers, get_ratio(n_tbox,n_esd), label='TBox Jets',color='green',marker='_')
    ax[1].plot(bin_centers, get_ratio(n_pbox,n_esd), label='PBox Jets',color='red',marker='_')
    ax[1].axhline(1,ls='--',color='gold')
    ax[1].set(xlabel="Jet pT (GeV)")
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    fig.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set_yscale('log')
        fig.savefig(save_loc1 + f'/box_pt_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        fig.savefig(save_loc1 + f'/box_pt.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    #-----------------------------------------------------------------------------------------------------------------
    # eta PLOTS
    #-----------------------------------------------------------------------------------------------------------------


    save_loc2 = save_loc + "/eta/"
    if not os.path.exists(save_loc2):
        os.makedirs(save_loc2)

    esdjets_eta = load_object(folder_containing_lists + '/' + eta_files[0] + '.pkl')
    fjets_eta = load_object(folder_containing_lists + '/' + eta_files[1] + '.pkl')
    tboxjets_eta = load_object(folder_containing_lists + '/' + eta_files[2] + '.pkl')
    pboxjets_eta = load_object(folder_containing_lists + '/' + eta_files[3] + '.pkl')

    all_esdjets_eta = np.concatenate(load_object(folder_containing_lists + '/' + eta_files[0] + '.pkl'))
    all_fjets_eta = np.concatenate(load_object(folder_containing_lists + '/' + eta_files[1] + '.pkl'))
    all_tboxjets_eta = np.concatenate(load_object(folder_containing_lists + '/' + eta_files[2] + '.pkl'))
    all_pboxjets_eta = np.concatenate(load_object(folder_containing_lists + '/' + eta_files[3] + '.pkl'))

    pt_cut = 20_000 #20GeV
    hi_fjets_eta = all_fjets_eta[np.argwhere(all_fjets_pt>pt_cut)]
    hi_esdjets_eta = all_esdjets_eta[np.argwhere(all_esdjets_pt>pt_cut)]
    hi_tboxets_eta = all_tboxjets_eta[np.argwhere(all_tboxjets_pt>pt_cut)]
    hi_pboxets_eta = all_pboxjets_eta[np.argwhere(all_pboxjets_pt>pt_cut)]



    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_esd, bins, _ = ax[0].hist(hi_esdjets_eta,bins=50,color='gold',histtype='step',density=(not log),lw=1.5,label=f'Offline Jets {len(hi_esdjets_eta)}')
    # n_fj, bins, _ = ax[0].hist(hi_fjets_eta,bins=bins,color='dodgerblue',histtype='step',density=(not log),lw=1.5,label=f'All clus. {len(hi_fjets_eta)}')
    n_tbox, bins, _ = ax[0].hist(hi_tboxets_eta,bins=bins,color='red',histtype='step',density=(not log),lw=1.5,label=f'TBox Jets {len(hi_tboxets_eta)}')
    n_pbox, bins, _ = ax[0].hist(hi_pboxets_eta,bins=bins,color='green',histtype='step',density=(not log),lw=1.5,label=f'PBox Jets {len(hi_pboxets_eta)}')
    ax[0].set_title(f'{len(esdjets_pt)} Test Events, All Jets pT > {pt_cut/1000:.0f}GeV',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].grid()
    ax[0].legend()

    # ratios_fj = get_ratio(n_fj,n_esd) # np.where(n_esd != 0, n_fj / n_esd, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].plot(bin_centers, get_ratio(n_tbox,n_esd),color='green',label='TBox Jets',marker='_')
    ax[1].plot(bin_centers, get_ratio(n_pbox,n_esd),color='red',label='PBox Jets',marker='_')
    # ax[1].plot(bin_centers, ratios_fj,color='dodgerblue',label='All clus.',marker='_')
    ax[1].axhline(1,ls='--',color='gold')
    ax[1].set(xlabel="$\eta$")
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    fig.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set_yscale('log')
        fig.savefig(save_loc2 + f'/jet_eta_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        fig.savefig(save_loc2 + f'/jet_eta.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    #-----------------------------------------------------------------------------------------------------------------
    # phi PLOTS
    #-----------------------------------------------------------------------------------------------------------------

    save_loc3 = save_loc + "/phi/"
    if not os.path.exists(save_loc3):
        os.makedirs(save_loc3)

    all_esdjets_phi = np.concatenate(load_object(folder_containing_lists + '/' + phi_files[0] + '.pkl'))
    all_fjets_phi = transform_angle(np.concatenate(load_object(folder_containing_lists + '/' + phi_files[1] + '.pkl')))
    all_tboxjets_phi = transform_angle(np.concatenate(load_object(folder_containing_lists + '/' + phi_files[2] + '.pkl')))
    all_pboxjets_phi = transform_angle(np.concatenate(load_object(folder_containing_lists + '/' + phi_files[3] + '.pkl')))

    pt_cut = 20_000 #20GeV
    hi_fjets_phi = all_fjets_phi[np.argwhere(all_fjets_pt>pt_cut)]
    hi_esdjets_phi = all_esdjets_phi[np.argwhere(all_esdjets_pt>pt_cut)]
    hi_tboxets_phi = all_tboxjets_phi[np.argwhere(all_tboxjets_pt>pt_cut)]
    hi_pboxets_phi = all_pboxjets_phi[np.argwhere(all_pboxjets_pt>pt_cut)]

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_esd, bins, _ = ax[0].hist(hi_esdjets_phi,bins=50,color='gold',histtype='step',density=(not log),lw=1.5,label=f'Offline Jets {len(hi_esdjets_phi)}')
    # n_fj, bins, _ = ax[0].hist(hi_fjets_phi,bins=bins,color='dodgerblue',histtype='step',density=(not log),lw=1.5,label=f'All clus. {len(hi_fjets_phi)}')
    n_tbox, bins, _ = ax[0].hist(hi_tboxets_phi,bins=bins,color='red',histtype='step',density=(not log),lw=1.5,label=f'TBox Jets {len(hi_tboxets_phi)}')
    n_pbox, bins, _ = ax[0].hist(hi_pboxets_phi,bins=bins,color='green',histtype='step',density=(not log),lw=1.5,label=f'PBox Jets {len(hi_pboxets_phi)}')
    ax[0].set_title(f'{len(esdjets_pt)} Test Events, All Jets pT > {pt_cut/1000:.0f}GeV',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].grid()
    ax[0].legend()

    # ratios_fj = get_ratio(n_fj,n_esd) # np.where(n_esd != 0, n_fj / n_esd, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].plot(bin_centers, get_ratio(n_tbox,n_esd),color='green',label='TBox Jets',marker='_')
    ax[1].plot(bin_centers, get_ratio(n_pbox,n_esd),color='red',label='PBox Jets',marker='_')
    # ax[1].plot(bin_centers, ratios_fj,color='dodgerblue',label='All clus.',marker='_')
    ax[1].axhline(1,ls='--',color='gold')
    ax[1].set(xlabel="$\phi$")
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    fig.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set_yscale('log')
        fig.savefig(save_loc3 + f'/jet_phi_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        fig.savefig(save_loc3 + f'/jet_phi.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    #-----------------------------------------------------------------------------------------------------------------
    # central jets PLOTS
    #-----------------------------------------------------------------------------------------------------------------


    eta_min,eta_max = -2.5,2.5
    cent_fjets_pt = all_fjets_pt[(all_fjets_eta > eta_min) & (all_fjets_eta < eta_max) & (all_fjets_pt>pt_cut)]
    cent_esdjets_pt = all_esdjets_pt[(all_esdjets_eta > eta_min) & (all_esdjets_eta < eta_max) & (all_esdjets_pt>pt_cut)]
    cent_tboxets_pt = all_tboxjets_pt[(all_tboxjets_eta > eta_min) & (all_tboxjets_eta < eta_max) & (all_tboxjets_pt>pt_cut)]
    cent_pboxets_pt = all_pboxjets_pt[(all_pboxjets_eta > eta_min) & (all_pboxjets_eta < eta_max) & (all_pboxjets_pt>pt_cut)]

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_esd, bins, _ = ax[0].hist(cent_esdjets_pt/1000,bins=50,color='gold',histtype='step',density=(not log),lw=1.5,label=f'Offline Jets {len(cent_esdjets_pt)}')
    # n_fj, bins, _ = ax[0].hist(cent_fjets_pt/1000,bins=bins,color='dodgerblue',histtype='step',density=(not log),lw=1.5,label=f'All clus. {len(cent_fjets_pt)}')
    n_tbox, bins, _ = ax[0].hist(cent_tboxets_pt/1000,bins=bins,color='green',histtype='step',density=(not log),lw=1.5,label=f'TBox Jets {len(cent_tboxets_pt)}')
    n_pbox, bins, _ = ax[0].hist(cent_pboxets_pt/1000,bins=bins,color='red',histtype='step',density=(not log),lw=1.5,label=f'PBox Jets {len(cent_pboxets_pt)}')
    # ax[0].axvline(pt_cut,ls='--',color='red',label='pT cut')
    ax[0].set_title(f'{len(esdjets_pt)} Events Central Jets w/ pt & eta cut {pt_cut/1000:.0f}GeV, [{eta_min},{eta_max}]',fontsize=15, fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].grid()
    ax[0].legend()

    # ratios_fj = get_ratio(n_fj,n_esd) # np.where(n_esd != 0, n_fj / n_esd, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].plot(bin_centers, get_ratio(n_tbox,n_esd),color='green', label='TBox Jets',marker='_')
    ax[1].plot(bin_centers, get_ratio(n_pbox,n_esd), color='red',label='PBox Jets',marker='_')
    # ax[1].plot(bin_centers, ratios_fj, color='dodgerblue',label='All clus.',marker='_')
    ax[1].axhline(1,ls='--',color='gold')
    ax[1].set(xlabel="Jet pT (GeV)")
    ax[1].grid(color='0.95')
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    fig.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set_yscale('log')
        fig.savefig(save_loc1 + f'/central_jet_pt_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        fig.savefig(save_loc1 + f'/central_jet_pt.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_tbox, bins, _ = ax[0].hist(cent_tboxets_pt/1000,bins=50,color='green',histtype='step',density=(not log),lw=1.5,label=f'TBox Jets {len(cent_tboxets_pt)}')
    n_pbox, bins, _ = ax[0].hist(cent_pboxets_pt/1000,bins=bins,color='red',histtype='step',density=(not log),lw=1.5,label=f'PBox Jets {len(cent_pboxets_pt)}')
    n_esd, bins, _ = ax[0].hist(cent_esdjets_pt/1000,bins=bins,color='gold',histtype='step',density=(not log),lw=1.5,label=f'Offline Jets {len(cent_esdjets_pt)}')
    # n_fj, bins, _ = ax[0].hist(cent_fjets_pt/1000,bins=bins,color='dodgerblue',histtype='step',density=(not log),lw=1.5,label=f'All clus. {len(cent_fjets_pt)}')
    # ax[0].axvline(pt_cut,ls='--',color='red',label='pT cut')
    ax[0].set_title(f'{len(esdjets_pt)} Events Jets w/ pt & eta cut {pt_cut/1000:.0f}GeV, [{eta_min},{eta_max}]',fontsize=15, fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].grid()
    ax[0].legend()

    # ratios_fj = get_ratio(n_fj,n_esd) # np.where(n_esd != 0, n_fj / n_esd, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].plot(bin_centers, get_ratio(n_tbox,n_esd),color='green', label='TBox Jets',marker='_')
    ax[1].plot(bin_centers, get_ratio(n_pbox,n_esd), color='red',label='PBox Jets',marker='_')
    # ax[1].plot(bin_centers, ratios_fj, color='dodgerblue',label='All clus.',marker='_')
    ax[1].axhline(1,ls='--',color='gold')
    ax[1].set(xlabel="Jet pT (GeV)")
    ax[1].grid(color='0.95')
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    fig.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set_yscale('log')
        fig.savefig(save_loc1 + f'/central_box_pt_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        fig.savefig(save_loc1 + f'/central_box_pt.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()
    
    
    #-----------------------------------------------------------------------------------------------------------------
    # leading jets PLOTS
    #-----------------------------------------------------------------------------------------------------------------


    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_esd, bins, _ = ax[0].hist(esdj_lead_pt/1000,bins=50,color='gold',histtype='step',density=(not log),lw=1.5,label=f'Offline Jets ({np.mean(esdj_lead_pt/1000):.2f}$\pm${np.std(esdj_lead_pt/1000):.0f})')
    # n_fj, bins, _ = ax[0].hist(fjj_lead_pt/1000,bins=bins,color='dodgerblue',histtype='step',density=(not log),lw=1.5,label=f'All clus. ({len(fjj_lead_pt)})')
    n_tbox, bins, _ = ax[0].hist(tbox_lead_pt/1000,bins=bins,color='green',histtype='step',density=(not log),lw=1.5,label=f'TBox Jets ({np.mean(tbox_lead_pt/1000):.2f}$\pm${np.std(tbox_lead_pt/1000):.0f})')
    n_pbox, bins, _ = ax[0].hist(pbox_lead_pt/1000,bins=bins,color='red',histtype='step',density=(not log),lw=1.5,label=f'PBox Jets ({len(pbox_lead_pt)})')
    # ax[0].axvline(pt_cut,ls='--',color='red',label='pT cut')
    ax[0].set_title(f'{len(esdj_lead_pt)} Test Events Leading Jets (Zoomed In)',fontsize=15, fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].grid(color='0.95')
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.34, 0.07),fontsize="small")

    # ratios_fj = get_ratio(n_fj,n_esd) # np.where(n_esd != 0, n_fj / n_esd, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].plot(bin_centers, get_ratio(n_tbox,n_esd),color='green', label='TBox Jets',marker='_')
    ax[1].plot(bin_centers, get_ratio(n_pbox,n_esd), color='red',label='PBox Jets',marker='_')
    # ax[1].plot(bin_centers, ratios_fj, color='dodgerblue',label='All clus.',marker='_')
    ax[1].axhline(1,ls='--',color='gold')
    ax[1].set(xlabel="Leading Jet pT (GeV)")
    ax[1].grid(color='0.95')
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    fig.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set_yscale('log')
        fig.savefig(save_loc1 + f'/leading_jet_pt_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        fig.savefig(save_loc1 + f'/leading_jet_pt.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()
    




    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_pbox, bins, _ = ax[0].hist(pbox_lead_pt/1000,bins=50,color='red',histtype='step',density=(not log),lw=1.5,label=f'PBox Jets ({np.mean(pbox_lead_pt/1000):.2f}$\pm${np.std(pbox_lead_pt/1000):.0f})')
    n_tbox, bins, _ = ax[0].hist(tbox_lead_pt/1000,bins=bins,color='green',histtype='step',density=(not log),lw=1.5,label=f'TBox Jets ({np.mean(tbox_lead_pt/1000):.2f}$\pm${np.std(tbox_lead_pt/1000):.0f})')
    n_esd, bins, _ = ax[0].hist(esdj_lead_pt/1000,bins=bins,color='gold',histtype='step',density=(not log),lw=1.5,label=f'Offline Jets ({np.mean(esdj_lead_pt/1000):.2f}$\pm${np.std(esdj_lead_pt/1000):.0f})')
    # n_fj, bins, _ = ax[0].hist(fjj_lead_pt,bins=bins,color='dodgerblue',histtype='step',density=(not log),lw=1.5,label=f'All clus. ({np.mean(fjj_lead_pt):.2f}$\pm${np.std(tbox_lead_pt):.0f})')
    # ax[0].axvline(pt_cut,ls='--',color='red',label='pT cut')
    ax[0].set_title(f'{len(esdj_lead_pt)} Test Events Leading Jets',fontsize=15, fontfamily="TeX Gyre Heros")
    ax[0].set(ylabel='Freq.')
    ax[0].grid(color='0.95')
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.5, 0.5),fontsize="medium")

    # ratios_fj = get_ratio(n_fj,n_esd) # np.where(n_esd != 0, n_fj / n_esd, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].plot(bin_centers, get_ratio(n_tbox,n_esd),color='green', label='TBox Jets',marker='_')
    ax[1].plot(bin_centers, get_ratio(n_pbox,n_esd), color='red',label='PBox Jets',marker='_')
    # ax[1].plot(bin_centers, ratios_fj, color='dodgerblue',label='All clus.',marker='_')
    ax[1].axhline(1,ls='--',color='gold')
    ax[1].set(xlabel="Leading Jet pT (GeV)")
    ax[1].grid(color='0.95')
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    fig.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set_yscale('log')
        fig.savefig(save_loc1 + f'/leading_box_pt_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        fig.savefig(save_loc1 + f'/leading_box_pt.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    #-----------------------------------------------------------------------------------------------------------------
    # Number jets PLOTS
    #-----------------------------------------------------------------------------------------------------------------

    n_esdjets = load_object(folder_containing_lists + njet_files[0] + '.pkl')
    n_fjets = load_object(folder_containing_lists + njet_files[1] + '.pkl')
    n_tboxjets = load_object(folder_containing_lists + njet_files[2] + '.pkl')
    n_pboxjets = load_object(folder_containing_lists + njet_files[3] + '.pkl')
    # max_max_n = max(max(n_esdjets),max(n_fjets),max(n_tboxjets),max(n_pboxjets))
    max_max_n = max(max(n_esdjets),max(n_tboxjets),max(n_pboxjets))

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    n_esd, bins, _ = ax[0].hist(n_esdjets,bins=40,range=(0,max_max_n),density=(not log),color='gold',histtype='step',lw=1.5,label=f'ESD Jets ({np.mean(n_esdjets):.2f}$\pm${np.mean(n_esdjets):.1f})')
    # n_fj, bins, _ = ax[0].hist(n_fjets,bins=40,range=(0,max_max_n),density=(not log),color='dodgerblue',histtype='step',lw=1.5,label=f'All clus. ({np.mean(n_fjets):.2f}$\pm${np.mean(n_fjets):.1f})')
    n_tbox, bins, _ = ax[0].hist(n_tboxjets,bins=40,range=(0,max_max_n),density=(not log),color='green',histtype='step',lw=1.5,label=f'TBox Jets ({np.mean(n_tboxjets):.2f}$\pm${np.mean(n_tboxjets):.1f})')
    n_pbox, bins, _ = ax[0].hist(n_pboxjets,bins=40,range=(0,max_max_n),density=(not log),color='red',histtype='step',lw=1.5,label=f'PBox Jets ({np.mean(n_pboxjets):.2f}$\pm${np.mean(n_pboxjets):.1f})')
    # ax[0].axvline(pt_cut,ls='--',color='red',label='pT cut')
    ax[0].set(ylabel='Freq.',title=f'{len(n_fjets)} Test Events')
    ax[0].grid()
    

    # ratios_fj = get_ratio(n_fj,n_esd) # np.where(n_esd != 0, n_fj / n_esd, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].plot(bin_centers, get_ratio(n_tbox,n_esd),color='green', label='TBox Jets',marker='_')
    ax[1].plot(bin_centers, get_ratio(n_pbox,n_esd), color='red',label='PBox Jets',marker='_')
    # ax[1].plot(bin_centers, ratios_fj, color='dodgerblue',label='All clus.',marker='_')
    ax[1].axhline(1,ls='--',color='gold')
    ax[1].grid()
    ax[1].set(xlabel="Num. Jets per event")

    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    fig.subplots_adjust(hspace=0.075)
    if log:
        ax[0].legend(loc='lower left',bbox_to_anchor=(0.57, 0.62),fontsize="small")
        ax[0].set_yscale('log')
        fig.savefig(save_loc + f'/event_n_jets_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].legend(loc='lower left',bbox_to_anchor=(0.5, 0.5),fontsize="medium")
        ax[0].set(ylabel='Freq. Density')
        fig.savefig(save_loc + f'/event_n_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()






folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD_raw_50k5_mu_13e/new_jet_metrics/"
save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_plots/SSD_raw_50k5_mu_13e"
  
if __name__=="__main__":
    print('Making jet plots')
    make_jet_plots(folder_to_look_in,save_at)
    make_jet_plots(folder_to_look_in,save_at,log=True)
    # make_jet_plots(folder_to_look_in,save_at,image_format="pdf")
    # make_jet_plots(folder_to_look_in,save_at,log=True,image_format="pdf")
    print('Completed jet plots\n')
