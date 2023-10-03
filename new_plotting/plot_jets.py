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
):
    save_loc = save_folder + "/jets/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    pt_files = ['total_pt_esdjets','total_pt_fjets','total_pt_tboxjets','total_pt_pboxjets']
    eta_files = ['total_eta_esdjets','total_eta_fjets','total_eta_tboxjets','total_eta_pboxjets']
    phi_files = ['total_phi_esdjets','total_phi_fjets','total_phi_tboxjets','total_phi_pboxjets']

    esdjets_pt = load_object(folder_containing_lists + pt_files[0] + '.pkl')
    fjets_pt = load_object(folder_containing_lists + pt_files[1] + '.pkl')
    tboxjets_pt = load_object(folder_containing_lists + pt_files[2] + '.pkl')
    pboxjets_pt = load_object(folder_containing_lists + pt_files[3] + '.pkl')

    #-----------------------------------------------------------------------------------------------------------------
    # Leading Jet

    esdj_lead_pt = np.array([leading_jet_pt(x) for x in esdjets_pt])
    fjj_lead_pt = np.array([leading_jet_pt(y) for y in fjets_pt])
    tbox_lead_pt = np.array([leading_jet_pt(x) for x in tboxjets_pt])
    pbox_lead_pt = np.array([leading_jet_pt(x) for x in pboxjets_pt])

    #make a trigger decision based on ESD jet
    lead_jet_pt_cut = 400_000 # 400GeV
    trig_decision_esd = np.argwhere(esdj_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_fjj = np.argwhere(fjj_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_tb = np.argwhere(tbox_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_pb = np.argwhere(pbox_lead_pt>lead_jet_pt_cut).T[0]

    bin_width = 10_000
    num_bins=45
    bins = [250_000+i*bin_width for i in range(num_bins+1)]

    f,ax = plt.subplots(3,1,figsize=(7.5,12))
    n_esd,bins,_ = ax[0].hist(esdj_lead_pt,bins=bins,histtype='step',label='ESD Jet')
    ax[0].set(xlabel="Leading jet pT (GeV)",ylabel='Freq.',title=f'Before {lead_jet_pt_cut/1000:.0f}GeV Cut')
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000)))
    ax[0].legend()

    n2_esd,bins,_ = ax[1].hist(esdj_lead_pt[trig_decision_esd],bins=bins,histtype='step',label='ESD Jet')
    n2_fj,_,_ = ax[1].hist(esdj_lead_pt[trig_decision_fjj],bins=bins,histtype='step',label='All clus.')
    n2_tbox,_,_ = ax[1].hist(esdj_lead_pt[trig_decision_tb],bins=bins,histtype='step',label='TBox')
    n2_pbox,_,_ = ax[1].hist(esdj_lead_pt[trig_decision_pb],bins=bins,histtype='step',label='PBox')
    ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set(xlabel="Leading jet pT (GeV)",ylabel='Freq.',title=f'After {lead_jet_pt_cut/1000:.0f}GeV Cut')
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000)))
    ax[1].legend()


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
        ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='ESD')
        ax[2].errorbar(bin_centers,fj_eff,xerr=bin_width/2,yerr=fj_err,elinewidth=0.4,marker='.',ls='none',label='All clus.')
        ax[2].errorbar(bin_centers,tbox_eff,xerr=bin_width/2,yerr=tbox_err,elinewidth=0.4,marker='.',ls='none',label='TBox')
        ax[2].errorbar(bin_centers,pbox_eff,xerr=bin_width/2,yerr=pbox_err,elinewidth=0.4,marker='.',ls='none',label='PBox')
    else:
        ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='ESD')
        ax[2].errorbar(bin_centers,fj_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='All clus.')
        ax[2].errorbar(bin_centers,tbox_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='TBox')
        ax[2].errorbar(bin_centers,pbox_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='PBox')
    ax[2].grid(color="0.95")
    ax[2].set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency',ylim=(-0.2,1.2))
    ax[2].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000)))
    ax[2].legend(loc='lower right')

    plt.tight_layout()
    f.savefig(save_loc+f'eff_plot_leading{lead_jet_pt_cut/1000:.0f}GeV.png')

    #-----------------------------------------------------------------------------------------------------------------
    # Nth Leading Jet
    nth_jet = 4
    esdj_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in esdjets_pt])
    fjj_nlead_pt = np.array([nth_leading_jet_pt(y,nth_jet) for y in fjets_pt])
    tbox_nlead_pt = np.array([nth_leading_jet_pt(z,nth_jet) for z in tboxjets_pt])
    pbox_nlead_pt = np.array([nth_leading_jet_pt(z,nth_jet) for z in pboxjets_pt])
    #make a trigger decision based on ESD jet
    nth_lead_jet_pt_cut = 45_000 # 400GeV
    trigger_decision = np.argwhere(esdj_nlead_pt>nth_lead_jet_pt_cut).T[0]

    trig_decision_esd = np.argwhere(esdj_nlead_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_fjj = np.argwhere(fjj_nlead_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_tb = np.argwhere(tbox_nlead_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_pb = np.argwhere(pbox_nlead_pt>nth_lead_jet_pt_cut).T[0]


    f,ax = plt.subplots(3,1,figsize=(7.5,12))
    n_esd,bins,_ = ax[0].hist(esdj_nlead_pt,bins=20,histtype='step',label='ESD Jet')
    # n_fj,_,_ = ax[0].hist(fjj_nlead_pt,bins=bins,histtype='step',label='All clus.')
    # n_tbox,_,_ = ax[0].hist(tbox_nlead_pt,bins=bins,histtype='step',label='TBox')
    # n_pbox,_,_ = ax[0].hist(pbox_nlead_pt,bins=bins,histtype='step',label='PBox')
    ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Freq.',title=f'Before {nth_lead_jet_pt_cut/1000:.0f}GeV Cut')
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000)))
    ax[0].legend()

    n2_esd,bins,_ = ax[1].hist(esdj_nlead_pt[trig_decision_esd],bins=bins,histtype='step',label='ESD Jet')
    n2_fj,_,_ = ax[1].hist(esdj_nlead_pt[trig_decision_fjj],bins=bins,histtype='step',label='All clus.')
    n2_tbox,_,_ = ax[1].hist(esdj_nlead_pt[trig_decision_tb],bins=bins,histtype='step',label='TBox')
    n2_pbox,_,_ = ax[1].hist(esdj_nlead_pt[trig_decision_pb],bins=bins,histtype='step',label='PBox')
    ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Freq.',title=f'After {nth_lead_jet_pt_cut/1000:.0f}GeV Cut')
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000)))
    ax[1].legend()


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
    errorbs = False
    if errorbs:
        ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='ESD')
        ax[2].errorbar(bin_centers,fj_eff,xerr=bin_width/2,yerr=fj_err,elinewidth=0.4,marker='.',ls='none',label='All clus.')
        ax[2].errorbar(bin_centers,tbox_eff,xerr=bin_width/2,yerr=tbox_err,elinewidth=0.4,marker='.',ls='none',label='TBox')
        ax[2].errorbar(bin_centers,pbox_eff,xerr=bin_width/2,yerr=pbox_err,elinewidth=0.4,marker='.',ls='none',label='PBox')
    else:
        ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='ESD')
        ax[2].errorbar(bin_centers,fj_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='All clus.')
        ax[2].errorbar(bin_centers,tbox_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='TBox')
        ax[2].errorbar(bin_centers,pbox_eff,xerr=bin_width/2,elinewidth=0.4,marker='.',ls='none',label='PBox')
    # ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[2].grid(color="0.95")
    ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency',ylim=(-0.2,1.2))
    ax[2].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000)))
    ax[2].legend(loc='lower right')

    plt.tight_layout()
    f.savefig(save_loc+f'eff_plot_{nth_jet}leading{nth_lead_jet_pt_cut/1000:.0f}GeV.png')



    #-----------------------------------------------------------------------------------------------------------------
    # pT PLOTS
    #-----------------------------------------------------------------------------------------------------------------


    all_esdjets_pt = np.concatenate(load_object(folder_containing_lists + '/' + pt_files[0] + '.pkl'))
    all_fjets_pt = np.concatenate(load_object(folder_containing_lists + '/' + pt_files[1] + '.pkl'))
    all_tboxjets_pt = np.concatenate(load_object(folder_containing_lists + '/' + pt_files[2] + '.pkl'))
    all_pboxjets_pt = np.concatenate(load_object(folder_containing_lists + '/' + pt_files[3] + '.pkl'))

    pt_cut = 20_000 #20GeV
    hi_fjets_pt = all_fjets_pt[np.argwhere(all_fjets_pt>pt_cut)]
    hi_esdjets_pt = all_esdjets_pt[np.argwhere(all_esdjets_pt>pt_cut)]
    hi_tboxets_pt = all_tboxjets_pt[np.argwhere(all_tboxjets_pt>pt_cut)]
    hi_pboxets_pt = all_pboxjets_pt[np.argwhere(all_pboxjets_pt>pt_cut)]



    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    n_esd, bins, _ = ax[0].hist(hi_esdjets_pt,bins=50,color='goldenrod',histtype='step',label=f'ESD Jets {len(hi_esdjets_pt)}')
    n_fj, bins, _ = ax[0].hist(hi_fjets_pt,bins=bins,color='dodgerblue',histtype='step',label=f'FJets {len(hi_fjets_pt)}')
    n_tbox, bins, _ = ax[0].hist(hi_tboxets_pt,bins=bins,color='green',histtype='step',label=f'TBox Jets {len(hi_tboxets_pt)}')
    n_pbox, bins, _ = ax[0].hist(hi_pboxets_pt,bins=bins,color='red',histtype='step',label=f'PBox Jets {len(hi_pboxets_pt)}')
    # ax[0].axvline(pt_cut,ls='--',color='red',label='pT cut')
    ax[0].set(ylabel='Freq.',title=f'{len(esdjets_pt)} Events, All Jets pT > {pt_cut/1000:.0f}GeV')
    ax[0].set_yscale('log')
    ax[0].grid()
    ax[0].legend()

    ratios_pbox = get_ratio(n_pbox,n_esd)
    ratios_tbox = get_ratio(n_tbox,n_esd)
    ratios_fj = get_ratio(n_fj,n_esd) # np.where(n_esd != 0, n_fj / n_esd, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].plot(bin_centers, ratios_fj, label='FJets',color='dodgerblue',marker='_')
    ax[1].plot(bin_centers, ratios_tbox, label='TBox Jets',color='green',marker='_')
    ax[1].plot(bin_centers, ratios_pbox, label='PBox Jets',color='red',marker='_')
    ax[1].axhline(1,ls='--',color='goldenrod')
    ax[1].legend()
    ax[1].set(xlabel="Jet pT (GeV)")
    ax[1].grid()
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000)))
    fig.tight_layout()
    fig.savefig(save_loc+'jet_pt.png')


    #-----------------------------------------------------------------------------------------------------------------
    # eta PLOTS
    #-----------------------------------------------------------------------------------------------------------------

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



    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    n_esd, bins, _ = ax[0].hist(hi_esdjets_eta,bins=50,color='goldenrod',histtype='step',label=f'ESD Jets {len(hi_esdjets_eta)}')
    n_fj, bins, _ = ax[0].hist(hi_fjets_eta,bins=bins,color='dodgerblue',histtype='step',label=f'FJets {len(hi_fjets_eta)}')
    n_tbox, bins, _ = ax[0].hist(hi_tboxets_eta,bins=bins,color='red',histtype='step',label=f'TBox Jets {len(hi_tboxets_eta)}')
    n_pbox, bins, _ = ax[0].hist(hi_pboxets_eta,bins=bins,color='green',histtype='step',label=f'PBox Jets {len(hi_pboxets_eta)}')
    ax[0].set(ylabel='Freq.',title=f'{len(esdjets_pt)} Events, All Jets pT > {pt_cut/1000:.0f}GeV')
    # ax[0].set_yscale('log')
    ax[0].grid()
    ax[0].legend()

    ratios_pbox = get_ratio(n_pbox,n_esd)
    ratios_tbox = get_ratio(n_tbox,n_esd)
    ratios_fj = get_ratio(n_fj,n_esd) # np.where(n_esd != 0, n_fj / n_esd, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].plot(bin_centers, ratios_tbox,color='green',label='TBox Jets',marker='_')
    ax[1].plot(bin_centers, ratios_pbox,color='red',label='PBox Jets',marker='_')
    ax[1].plot(bin_centers, ratios_fj,color='dodgerblue',label='FJets',marker='_')
    ax[1].axhline(1,ls='--',color='goldenrod')
    ax[1].legend()
    ax[1].set(xlabel="$\eta$")
    ax[1].grid()
    fig.tight_layout()
    fig.savefig(save_loc+'jet_eta.png')


    #-----------------------------------------------------------------------------------------------------------------
    # phi PLOTS
    #-----------------------------------------------------------------------------------------------------------------



    all_esdjets_phi = np.concatenate(load_object(folder_containing_lists + '/' + phi_files[0] + '.pkl'))
    all_fjets_phi = transform_angle(np.concatenate(load_object(folder_containing_lists + '/' + phi_files[1] + '.pkl')))
    all_tboxjets_phi = transform_angle(np.concatenate(load_object(folder_containing_lists + '/' + phi_files[2] + '.pkl')))
    all_pboxjets_phi = transform_angle(np.concatenate(load_object(folder_containing_lists + '/' + phi_files[3] + '.pkl')))

    pt_cut = 20_000 #20GeV
    hi_fjets_phi = all_fjets_phi[np.argwhere(all_fjets_pt>pt_cut)]
    hi_esdjets_phi = all_esdjets_phi[np.argwhere(all_esdjets_pt>pt_cut)]
    hi_tboxets_phi = all_tboxjets_phi[np.argwhere(all_tboxjets_pt>pt_cut)]
    hi_pboxets_phi = all_pboxjets_phi[np.argwhere(all_pboxjets_pt>pt_cut)]

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    n_esd, bins, _ = ax[0].hist(hi_esdjets_phi,bins=50,color='goldenrod',histtype='step',label=f'ESD Jets {len(hi_esdjets_phi)}')
    n_fj, bins, _ = ax[0].hist(hi_fjets_phi,bins=bins,color='dodgerblue',histtype='step',label=f'FJets {len(hi_fjets_phi)}')
    n_tbox, bins, _ = ax[0].hist(hi_tboxets_phi,bins=bins,color='red',histtype='step',label=f'TBox Jets {len(hi_tboxets_phi)}')
    n_pbox, bins, _ = ax[0].hist(hi_pboxets_phi,bins=bins,color='green',histtype='step',label=f'PBox Jets {len(hi_pboxets_phi)}')
    ax[0].set(ylabel='Freq.',title=f'{len(esdjets_pt)} Events, All Jets pT > {pt_cut/1000:.0f}GeV')
    # ax[0].set_yscale('log')
    ax[0].grid()
    ax[0].legend()

    ratios_pbox = get_ratio(n_pbox,n_esd)
    ratios_tbox = get_ratio(n_tbox,n_esd)
    ratios_fj = get_ratio(n_fj,n_esd) # np.where(n_esd != 0, n_fj / n_esd, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].plot(bin_centers, ratios_tbox,color='green',label='TBox Jets',marker='_')
    ax[1].plot(bin_centers, ratios_pbox,color='red',label='PBox Jets',marker='_')
    ax[1].plot(bin_centers, ratios_fj,color='dodgerblue',label='FJets',marker='_')
    ax[1].axhline(1,ls='--',color='goldenrod')
    ax[1].legend()
    ax[1].set(xlabel="$\phi$")
    ax[1].grid()
    fig.tight_layout()
    fig.savefig(save_loc+'jet_phi.png')



    #-----------------------------------------------------------------------------------------------------------------
    # central jets PLOTS
    #-----------------------------------------------------------------------------------------------------------------


    eta_min,eta_max = -2.5,2.5
    cent_fjets_pt = all_fjets_pt[(all_fjets_eta > eta_min) & (all_fjets_eta < eta_max) & (all_fjets_pt>pt_cut)]
    cent_esdjets_pt = all_esdjets_pt[(all_esdjets_eta > eta_min) & (all_esdjets_eta < eta_max)]
    cent_tboxets_pt = all_tboxjets_pt[(all_tboxjets_eta > eta_min) & (all_tboxjets_eta < eta_max)]
    cent_pboxets_pt = all_pboxjets_pt[(all_pboxjets_eta > eta_min) & (all_pboxjets_eta < eta_max)]

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    n_esd, bins, _ = ax[0].hist(cent_esdjets_pt,bins=50,color='goldenrod',histtype='step',label=f'ESD Jets {len(cent_esdjets_pt)}')
    n_fj, bins, _ = ax[0].hist(cent_fjets_pt,bins=bins,color='dodgerblue',histtype='step',label=f'FJets {len(cent_fjets_pt)}')
    n_tbox, bins, _ = ax[0].hist(cent_tboxets_pt,bins=bins,color='green',histtype='step',label=f'TBox Jets {len(cent_tboxets_pt)}')
    n_pbox, bins, _ = ax[0].hist(cent_pboxets_pt,bins=bins,color='red',histtype='step',label=f'PBox Jets {len(cent_pboxets_pt)}')
    # ax[0].axvline(pt_cut,ls='--',color='red',label='pT cut')
    ax[0].set(ylabel='Freq.',title=f'{len(esdjets_pt)} Events Central Jets w/ eta cut [{eta_min},{eta_max}]')
    ax[0].set_yscale('log')
    ax[0].grid(color='0.95')
    ax[0].legend()

    ratios_pbox = get_ratio(n_pbox,n_esd)
    ratios_tbox = get_ratio(n_tbox,n_esd)
    ratios_fj = get_ratio(n_fj,n_esd) # np.where(n_esd != 0, n_fj / n_esd, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].plot(bin_centers, ratios_tbox,color='green', label='TBox Jets',marker='_')
    ax[1].plot(bin_centers, ratios_pbox, color='red',label='PBox Jets',marker='_')
    ax[1].plot(bin_centers, ratios_fj, color='dodgerblue',label='FJets',marker='_')
    ax[1].axhline(1,ls='--',color='goldenrod')
    ax[1].legend()
    ax[1].set(xlabel="Jet pT (GeV)")
    ax[1].grid(color='0.95')
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x / 1000)))
    fig.tight_layout()
    fig.savefig(save_loc+'jet_cent_pt.png')



    return