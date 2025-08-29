import numpy as np 
import torch
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


model_name = "jetSSD_custom_convnext_central_32e"
proc = "JZcomb0_test"
date = "20250313-06"
# # date = "20250406-23"
# proc = "ttbar_test"
# date = "20250407-14"
# date = "20250626-18" # jun 79k

metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/{proc}/{date}/box_metrics"
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/{proc}/{date}/trig/"
if not os.path.exists(save_folder): os.makedirs(save_folder)
image_format = "png"

print("=======================================================================================================")
print(f"Loading jets from\n{metrics_folder}")
print("=======================================================================================================\n")

total_tar_pt      = load_object(metrics_folder+"/tarboxes_pt.pkl")
total_tru_pt      = load_object(metrics_folder+"/truboxes_pt.pkl")
total_p_pt      = load_object(metrics_folder+"/pboxes_pt.pkl")
total_p_scr     = load_object(metrics_folder+"/pboxes_scores.pkl")


leading_jet = False
leading_truth_jet = False
subleading_jet = False
subleading_truth_jet = False
leading_iso_jet = False # dR > 0.4 on the TRUTH jets
subleading_iso_jet = False # dR > 0.4 on the TRUTH jets
leading_akt_iso_jet = False
subleading_akt_iso_jet = False
leading_central_jet = False
leading_central_truth_jet = False
subleading_central_jet = False
subleading_central_truth_jet = False
leading_central_akt_iso_jet = False
subleading_central_akt_iso_jet = False
leading_jet_multi_cut = False
subleading_jet_multi_cut = False
leading_akt_iso_jet_multi_cut = True
subleading_iso_jet_multi_cut = True

def leading_jet_pt(list_of_jet_pts_in_event):
    try:
        return max(list_of_jet_pts_in_event)
    except ValueError:
        #Doesn't have enough (or any) jets, automatically lost in cut
        return np.nan

def nth_leading_jet_pt(list_of_jet_pts_in_event,n):
    try:
        return sorted(list_of_jet_pts_in_event,reverse=True)[n-1]
    except IndexError or ValueError:
        # Doesn't have enough (or any) jets, automatically lost in cut
        return np.nan

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



if leading_jet:
    print("=======================================================================================================")
    print(f"Making leading jet trigger decision")
    print("=======================================================================================================\n")
    # Make a trigger decision based on leading antikt jet pt
    tar_lead_pt = np.array([leading_jet_pt(x) for x in total_tar_pt])
    p_lead_pt = np.array([leading_jet_pt(x) for x in total_p_pt])

    lead_jet_pt_cut = 150 # GeV
    trig_decision_akt = np.argwhere(tar_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(p_lead_pt>lead_jet_pt_cut).T[0]

    # Set the x-axis and binning
    start,end = 20,550
    step = 10
    bins = np.arange(start, end, step)

    f,ax = plt.subplots(3,1,figsize=(8,14))
    n_akt,bins,_ = ax[0].hist(tar_lead_pt,bins=bins,histtype='step',label='AntiKt4EMTopo jetConstitScale')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(xlabel="Leading EMTopo (Offline) jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(tar_lead_pt[trig_decision_akt],bins=bins,histtype='step',label='Antikt jetConstitScale',color="limegreen")
    n2_p,_,_ = ax[1].hist(tar_lead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes',color="red")
    ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel="Leading AntiKt4EMTopo jet pT (jet constit. scale) [GeV]",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_akt)
        step_err = get_errorbars(n2_akt,n_akt)

        pred_eff = get_ratio(n2_p,n_akt)
        pred_err = get_errorbars(n2_p,n_akt)


    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes.',color='red')
    ax[2].set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='lower right')
    # hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/leading_{lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

    f,a = plt.subplots(1,1,figsize=(8,8))
    # a.axvline(x=lead_jet_pt_cut,ymin=0.0,ymax=1.0,ls='--',color='black',alpha=0.3, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.vlines(x=[lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"$p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel="Leading AKT4EMTopo jet " + r"$p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start,end),ylim=(-0.2,1.2))
    # a.legend(loc='lower right')
    a.legend(bbox_to_anchor=(0.4,0.025), loc="lower left")
    plt.text(start+15,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+98,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+15,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start+15,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/leading{lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")




if leading_truth_jet:
    print("=======================================================================================================")
    print(f"Making leading jet trigger decision with TRUTH jets")
    print("=======================================================================================================\n")

    # Make a trigger decision based on leading antikt jet pt
    tru_lead_pt = np.array([leading_jet_pt(x) for x in total_tru_pt])
    tar_lead_pt = np.array([leading_jet_pt(x) for x in total_tar_pt])
    p_lead_pt = np.array([leading_jet_pt(x) for x in total_p_pt])

    lead_jet_pt_cut = 150 # GeV
    trig_decision_tru = np.argwhere(tru_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_akt = np.argwhere(tar_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(p_lead_pt>lead_jet_pt_cut).T[0]

    # Set the x-axis and binning
    start,end = 20,550
    step = 10
    bins = np.arange(start, end, step)

    f,ax = plt.subplots(3,1,figsize=(8,15.5))
    n_tru,bins,_ = ax[0].hist(tru_lead_pt,bins=bins,histtype='step',label='Truth jets',color='gold')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(tru_lead_pt[trig_decision_akt],bins=bins,histtype='step',label='Antikt jetConstitScale',color="limegreen")
    n2_p,_,_ = ax[1].hist(tru_lead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes',color="red")
    ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Truth jet cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_tru)
        step_err = get_errorbars(n2_akt,n_tru)

        pred_eff = get_ratio(n2_p,n_tru)
        pred_err = get_errorbars(n2_p,n_tru)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt (jet constit. scale)',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred jets (jet constit. scale)',color='red')
    ax[2].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='lower right')
    # hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/leading_truth_{lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

    f,a = plt.subplots(1,1,figsize=(8,8))
    # a.axvline(x=lead_jet_pt_cut,ymin=0.0,ymax=1.0,ls='--',color='black',alpha=0.3, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.vlines(x=[lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='AKT4EMTopo',color='limegreen')
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel="Leading truth jet " + r"$p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start,end),ylim=(-0.2,1.2))
    # a.legend(loc='lower right')
    a.legend(bbox_to_anchor=(0.51,0.025), loc="lower left")
    plt.text(start+15,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+98,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+15,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start+15,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/leading_truth{lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


if subleading_jet:
    print("=======================================================================================================")
    print(f"Making subleading jet trigger decision")
    print("=======================================================================================================\n")

    start,end = 20,160
    step = 5
    bins = np.arange(start, end, step)

    nth_jet = 3
    nth_lead_jet_pt_cut = 60 # GeV
    tru_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_tru_pt])
    tar_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_tar_pt])
    p_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_p_pt])

    trig_decision_akt  = np.argwhere(tar_nlead_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(p_nlead_pt>nth_lead_jet_pt_cut).T[0]

    f,ax = plt.subplots(3,1,figsize=(6.5,12))
    n_akt,bins,_ = ax[0].hist(tar_nlead_pt,bins=bins,histtype='step',label='Anti-kt jetConstitScale')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
    ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(tar_nlead_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="limegreen")
    n2_pb,_,_ = ax[1].hist(tar_nlead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes Adj',color="red")
    ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_akt)
        step_err = get_errorbars(n2_akt,n_akt)
        pred_eff = get_ratio(n2_pb,n_akt)
        pred_err = get_errorbars(n2_pb,n_akt)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes',color='red')
    ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='upper left')
    hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


    f,a = plt.subplots(1,1,figsize=(8,8))
    # a.axvline(x=lead_jet_pt_cut,ymin=0.0,ymax=1.0,ls='--',color='black',alpha=0.3, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.vlines(x=[nth_lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"$p_T$" + f" cut ({nth_lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel=f"${{{nth_jet}}}^{{rd}}$ leading AKT4EMTopo jet " + r" $p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start-10,end),ylim=(-0.2,1.2))
    a.legend(loc='lower right')
    plt.text(start-3,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+20,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start-3,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start-3,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")




if subleading_truth_jet:
    print("=======================================================================================================")
    print(f"Making subleading jet trigger decision with TRUTH jets")
    print("=======================================================================================================\n")
    
    start,end = 20,160
    step = 5
    bins = np.arange(start, end, step)

    nth_jet = 3
    nth_lead_jet_pt_cut = 60 # 400GeV
    tru_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_tru_pt])
    tar_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_tar_pt])
    p_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_p_pt])

    trig_decision_akt  = np.argwhere(tar_nlead_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(p_nlead_pt>nth_lead_jet_pt_cut).T[0]

    f,ax = plt.subplots(3,1,figsize=(6.5,14))
    n_tru,bins,_ = ax[0].hist(tru_nlead_pt,bins=bins,histtype='step',label='Truth jets', color='gold')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
    ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(tru_nlead_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="limegreen")
    n2_pb,_,_ = ax[1].hist(tru_nlead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes Adj',color="red")
    ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Truth jet cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_tru)
        step_err = get_errorbars(n2_akt,n_tru)
        pred_eff = get_ratio(n2_pb,n_tru)
        pred_err = get_errorbars(n2_pb,n_tru)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt (jet constit. scale)',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred jets (jet constit. scale)',color='red')
    ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Efficiency')
    # ax[2].legend(loc='upper left')
    hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/{nth_jet}leading_truth_{nth_lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



    f,a = plt.subplots(1,1,figsize=(8,8))
    # a.axvline(x=lead_jet_pt_cut,ymin=0.0,ymax=1.0,ls='--',color='black',alpha=0.3, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.vlines(x=[nth_lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"Truth $p_T$" + f" cut ({nth_lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='AKT4EMTopo',color='limegreen')
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel=f"${{{nth_jet}}}^{{rd}}$ leading truth jet " + r" $p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start-10,end),ylim=(-0.2,1.2))
    a.legend(loc='lower right')
    plt.text(start-3,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+20,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start-3,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start-3,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/{nth_jet}leading_truth{nth_lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")





if leading_iso_jet:
    total_tru_eta      = load_object(metrics_folder+"/truboxes_eta.pkl")
    total_tru_phi      = load_object(metrics_folder+"/truboxes_phi.pkl")
    print("=======================================================================================================")
    print(f"Making leading jet trigger decision, but requiring truth jets be dR > 0.4 away from each other")
    print("=======================================================================================================\n")

    print("We get rid of some events")
    keep_iso = np.zeros(len(total_tru_pt))
    for i in range(len(total_tru_pt)):
        event_i_jet_pt = total_tru_pt[i] 
        event_i_jet_eta = total_tru_eta[i] 
        event_i_jet_phi = total_tru_phi[i] 
        # print(len(event_i_jet_pt),len(event_i_jet_pt[event_i_jet_pt>20]))
        # filter, only count truth jets above 20 GeV
        event_i_jet_eta = event_i_jet_eta[event_i_jet_pt>20]
        event_i_jet_phi = event_i_jet_phi[event_i_jet_pt>20]
        event_i_jet_pt = event_i_jet_pt[event_i_jet_pt>20]

        if len(event_i_jet_pt)>3:
            ind = np.argpartition(event_i_jet_pt,-4)[-4:]
            top4_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top4_idx[3]]
            jet0_eta = event_i_jet_eta[top4_idx[3]]
            jet0_phi = event_i_jet_phi[top4_idx[3]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top4_idx[2]]
            jet1_eta = event_i_jet_eta[top4_idx[2]]
            jet1_phi = event_i_jet_phi[top4_idx[2]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top4_idx[1]]
            jet2_eta = event_i_jet_eta[top4_idx[1]]
            jet2_phi = event_i_jet_phi[top4_idx[1]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            jet3_pt = event_i_jet_pt[top4_idx[0]]
            jet3_eta = event_i_jet_eta[top4_idx[0]]
            jet3_phi = event_i_jet_phi[top4_idx[0]]
            jet3_ = np.array((jet3_eta,jet3_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR03 = np.linalg.norm(jet0_-jet3_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            dR13 = np.linalg.norm(jet1_-jet3_)
            dR23 = np.linalg.norm(jet2_-jet3_)
            # print(jet0_pt,jet1_pt,jet2_pt,jet3_pt)
            # print(dR01,dR02,dR12)
            if (dR01 > 0.6) and (dR02 > 0.6) and (dR03 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>2:
            ind = np.argpartition(event_i_jet_pt,-3)[-3:]
            top3_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top3_idx[2]]
            jet0_eta = event_i_jet_eta[top3_idx[2]]
            jet0_phi = event_i_jet_phi[top3_idx[2]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top3_idx[1]]
            jet1_eta = event_i_jet_eta[top3_idx[1]]
            jet1_phi = event_i_jet_phi[top3_idx[1]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top3_idx[0]]
            jet2_eta = event_i_jet_eta[top3_idx[0]]
            jet2_phi = event_i_jet_phi[top3_idx[0]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            if (dR01 > 0.6) and (dR02 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>1:
            ind = np.argpartition(event_i_jet_pt,-2)[-2:]
            top2_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top2_idx[1]]
            jet0_eta = event_i_jet_eta[top2_idx[1]]
            jet0_phi = event_i_jet_phi[top2_idx[1]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top2_idx[0]]
            jet1_eta = event_i_jet_eta[top2_idx[0]]
            jet1_phi = event_i_jet_phi[top2_idx[0]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            if (dR01 > 0.6):
                keep_iso[i] = 1
        
        else:
            keep_iso[i] = 1

    print('--->',sum(keep_iso),len(keep_iso))
    print("now we can get rid of events where the leading truth jet is close to another top 4 jet (where possible)")
    iso_total_tar_pt = []
    iso_total_tru_pt = []
    iso_total_p_pt = []
    for j in range(len(keep_iso)):
        should_keep = keep_iso[j]
        if should_keep:
            iso_total_tru_pt.append(total_tru_pt[j])
            iso_total_tar_pt.append(total_tar_pt[j])
            iso_total_p_pt.append(total_p_pt[j])




    # Make a trigger decision based on leading antikt jet pt
    tar_lead_pt = np.array([leading_jet_pt(x) for x in iso_total_tar_pt])
    p_lead_pt = np.array([leading_jet_pt(x) for x in iso_total_p_pt])

    lead_jet_pt_cut = 150 # GeV
    trig_decision_akt = np.argwhere(tar_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(p_lead_pt>lead_jet_pt_cut).T[0]

    # Set the x-axis and binning
    start,end = 20,550
    step = 10
    bins = np.arange(start, end, step)

    f,ax = plt.subplots(3,1,figsize=(8,14))
    n_akt,bins,_ = ax[0].hist(tar_lead_pt,bins=bins,histtype='step',label='AntiKt4EMTopo jetConstitScale')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(xlabel="Leading EMTopo (Offline) jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(tar_lead_pt[trig_decision_akt],bins=bins,histtype='step',label='Antikt jetConstitScale',color="limegreen")
    n2_p,_,_ = ax[1].hist(tar_lead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes',color="red")
    ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel="Leading AntiKt4EMTopo jet pT (jet constit. scale) [GeV]",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_akt)
        step_err = get_errorbars(n2_akt,n_akt)

        pred_eff = get_ratio(n2_p,n_akt)
        pred_err = get_errorbars(n2_p,n_akt)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes.',color='red')
    ax[2].set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='lower right')
    # hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/iso_leading_{lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

    f,a = plt.subplots(1,1,figsize=(8,8))
    # a.axvline(x=lead_jet_pt_cut,ymin=0.0,ymax=1.0,ls='--',color='black',alpha=0.3, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.vlines(x=[lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"$p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel="Leading AKT4EMTopo jet " + r"$p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start,end),ylim=(-0.2,1.2))
    # a.legend(loc='lower right')
    a.legend(bbox_to_anchor=(0.40,0.025), loc="lower left")
    plt.text(end-295,-0.15, f"Truth jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    plt.text(start+15,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+98,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+15,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start+15,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/iso_leading{lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


    #####################################################################################################################################

    print("=======================================================================================================")
    print(f"Making truth leading jet trigger decision, but requiring truth jets be dR > 0.4 away from each other")
    print("=======================================================================================================\n")

    # Make a trigger decision based on leading antikt jet pt
    tru_lead_pt = np.array([leading_jet_pt(x) for x in iso_total_tru_pt])
    tar_lead_pt = np.array([leading_jet_pt(x) for x in iso_total_tar_pt])
    p_lead_pt = np.array([leading_jet_pt(x) for x in iso_total_p_pt])

    trig_decision_tru = np.argwhere(tru_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_akt = np.argwhere(tar_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(p_lead_pt>lead_jet_pt_cut).T[0]

    # Set the x-axis and binning
    start,end = 20,550
    step = 10
    bins = np.arange(start, end, step)

    f,ax = plt.subplots(3,1,figsize=(8,15.5))
    n_tru,bins,_ = ax[0].hist(tru_lead_pt,bins=bins,histtype='step',label='Truth jets',color='gold')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(tru_lead_pt[trig_decision_akt],bins=bins,histtype='step',label='Antikt jetConstitScale',color="limegreen")
    n2_p,_,_ = ax[1].hist(tru_lead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes',color="red")
    ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Truth jet cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_tru)
        step_err = get_errorbars(n2_akt,n_tru)

        pred_eff = get_ratio(n2_p,n_tru)
        pred_err = get_errorbars(n2_p,n_tru)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt (jet constit. scale)',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred jets (jet constit. scale)',color='red')
    ax[2].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='lower right')
    # hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/iso_leading_truth_{lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

    f,a = plt.subplots(1,1,figsize=(8,8))
    # a.axvline(x=lead_jet_pt_cut,ymin=0.0,ymax=1.0,ls='--',color='black',alpha=0.3, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.vlines(x=[lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='AKT4EMTopo',color='limegreen')
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel="Leading truth jet " + r"$p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start,end),ylim=(-0.2,1.2))
    # a.legend(loc='lower right')
    a.legend(bbox_to_anchor=(0.51,0.025), loc="lower left")
    plt.text(end-240,-0.15, f"Truth jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    plt.text(start+15,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+98,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+15,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start+15,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/iso_leading_truth{lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")







if subleading_iso_jet:
    total_tru_eta      = load_object(metrics_folder+"/truboxes_eta.pkl")
    total_tru_phi      = load_object(metrics_folder+"/truboxes_phi.pkl")
    print("=======================================================================================================")
    print(f"Making leading jet trigger decision, but requiring truth jets be dR > 0.4 away from each other")
    print("=======================================================================================================\n")

    print("We get rid of some events")
    keep_iso = np.zeros(len(total_tru_pt))
    for i in range(len(total_tru_pt)):
        event_i_jet_pt = total_tru_pt[i] 
        event_i_jet_eta = total_tru_eta[i] 
        event_i_jet_phi = total_tru_phi[i] 
        # filter, only count truth jets above 20 GeV
        event_i_jet_eta = event_i_jet_eta[event_i_jet_pt>20]
        event_i_jet_phi = event_i_jet_phi[event_i_jet_pt>20]
        event_i_jet_pt = event_i_jet_pt[event_i_jet_pt>20]

        if len(event_i_jet_pt)>3:
            ind = np.argpartition(event_i_jet_pt,-4)[-4:]
            top4_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top4_idx[3]]
            jet0_eta = event_i_jet_eta[top4_idx[3]]
            jet0_phi = event_i_jet_phi[top4_idx[3]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top4_idx[2]]
            jet1_eta = event_i_jet_eta[top4_idx[2]]
            jet1_phi = event_i_jet_phi[top4_idx[2]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top4_idx[1]]
            jet2_eta = event_i_jet_eta[top4_idx[1]]
            jet2_phi = event_i_jet_phi[top4_idx[1]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            jet3_pt = event_i_jet_pt[top4_idx[0]]
            jet3_eta = event_i_jet_eta[top4_idx[0]]
            jet3_phi = event_i_jet_phi[top4_idx[0]]
            jet3_ = np.array((jet3_eta,jet3_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR03 = np.linalg.norm(jet0_-jet3_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            dR13 = np.linalg.norm(jet1_-jet3_)
            dR23 = np.linalg.norm(jet2_-jet3_)
            # print(jet0_pt,jet1_pt,jet2_pt,jet3_pt)
            # print(dR01,dR02,dR12)
            if (dR01 > 0.6) and (dR02 > 0.6) and (dR03 > 0.6) and (dR12 > 0.6) and (dR13 > 0.6) and (dR23 > 0.6):
            # if (dR01 > 0.6) and (dR02 > 0.6) and (dR12 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>2:
            ind = np.argpartition(event_i_jet_pt,-3)[-3:]
            top3_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top3_idx[2]]
            jet0_eta = event_i_jet_eta[top3_idx[2]]
            jet0_phi = event_i_jet_phi[top3_idx[2]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top3_idx[1]]
            jet1_eta = event_i_jet_eta[top3_idx[1]]
            jet1_phi = event_i_jet_phi[top3_idx[1]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top3_idx[0]]
            jet2_eta = event_i_jet_eta[top3_idx[0]]
            jet2_phi = event_i_jet_phi[top3_idx[0]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            if (dR01 > 0.6) and (dR02 > 0.6) and (dR12 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>1:
            ind = np.argpartition(event_i_jet_pt,-2)[-2:]
            top2_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top2_idx[1]]
            jet0_eta = event_i_jet_eta[top2_idx[1]]
            jet0_phi = event_i_jet_phi[top2_idx[1]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top2_idx[0]]
            jet1_eta = event_i_jet_eta[top2_idx[0]]
            jet1_phi = event_i_jet_phi[top2_idx[0]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            if (dR01 > 0.6):
                keep_iso[i] = 1
        
        else:
            keep_iso[i] = 1

    print('-->',sum(keep_iso),len(keep_iso))
    print("now we can get rid of events where the leading truth jet is close to another top 4 jet (where possible)")
    iso_total_tar_pt = []
    iso_total_tru_pt = []
    iso_total_p_pt = []
    for j in range(len(keep_iso)):
        should_keep = keep_iso[j]
        if should_keep:
            iso_total_tru_pt.append(total_tru_pt[j])
            iso_total_tar_pt.append(total_tar_pt[j])
            iso_total_p_pt.append(total_p_pt[j])


    start,end = 20,160
    step = 5
    bins = np.arange(start, end, step)

    nth_jet = 3
    nth_lead_jet_pt_cut = 60 
    tru_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_tru_pt])
    tar_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_tar_pt])
    p_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_p_pt])

    trig_decision_akt  = np.argwhere(tar_nlead_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(p_nlead_pt>nth_lead_jet_pt_cut).T[0]

    f,ax = plt.subplots(3,1,figsize=(6.5,12))
    n_akt,bins,_ = ax[0].hist(tar_nlead_pt,bins=bins,histtype='step',label='Anti-kt jetConstitScale')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
    ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(tar_nlead_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="limegreen")
    n2_pb,_,_ = ax[1].hist(tar_nlead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes Adj',color="red")
    ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_akt)
        step_err = get_errorbars(n2_akt,n_akt)
        pred_eff = get_ratio(n2_pb,n_akt)
        pred_err = get_errorbars(n2_pb,n_akt)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes',color='red')
    ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='upper left')
    hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/iso_{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


    f,a = plt.subplots(1,1,figsize=(8,8))
    # a.axvline(x=lead_jet_pt_cut,ymin=0.0,ymax=1.0,ls='--',color='black',alpha=0.3, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.vlines(x=[nth_lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"$p_T$" + f" cut ({nth_lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel=f"${{{nth_jet}}}^{{rd}}$ leading AKT4EMTopo jet " + r" $p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start-10,end),ylim=(-0.2,1.2))
    a.legend(loc='lower right')
    plt.text(end-80,-0.175, f"Truth jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    plt.text(start-3,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+20,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start-3,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start-3,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/iso_{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



    print("=======================================================================================================")
    print(f"Making subleading jet trigger decision with TRUTH jets")
    print("=======================================================================================================\n")


    tru_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_tru_pt])
    tar_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_tar_pt])
    p_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_p_pt])

    trig_decision_akt  = np.argwhere(tar_nlead_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(p_nlead_pt>nth_lead_jet_pt_cut).T[0]

    f,ax = plt.subplots(3,1,figsize=(6.5,14))
    n_tru,bins,_ = ax[0].hist(tru_nlead_pt,bins=bins,histtype='step',label='Truth jets', color='gold')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
    ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(tru_nlead_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="limegreen")
    n2_pb,_,_ = ax[1].hist(tru_nlead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes Adj',color="red")
    ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Truth jet cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_tru)
        step_err = get_errorbars(n2_akt,n_tru)
        pred_eff = get_ratio(n2_pb,n_tru)
        pred_err = get_errorbars(n2_pb,n_tru)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt (jet constit. scale)',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred jets (jet constit. scale)',color='red')
    ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Efficiency')
    # ax[2].legend(loc='upper left')
    hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/iso_{nth_jet}leading_truth_{nth_lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

    f,a = plt.subplots(1,1,figsize=(8,8))
    # a.axvline(x=lead_jet_pt_cut,ymin=0.0,ymax=1.0,ls='--',color='black',alpha=0.3, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.vlines(x=[nth_lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"Truth $p_T$" + f" cut ({nth_lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='AKT4EMTopo',color='limegreen')
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel=f"${{{nth_jet}}}^{{rd}}$ leading truth jet " + r" $p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start-10,end),ylim=(-0.2,1.2))
    a.legend(loc='lower right')
    plt.text(end-60,-0.175, f"Truth jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    plt.text(start-3,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+20,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start-3,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start-3,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/iso_{nth_jet}leading_truth{nth_lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")









if leading_akt_iso_jet:
    total_tar_eta      = load_object(metrics_folder+"/tarboxes_eta.pkl")
    total_tar_phi      = load_object(metrics_folder+"/tarboxes_phi.pkl")
    print("=======================================================================================================")
    print(f"Making leading jet trigger decision, but requiring AKT jets be dR > 0.6 away from each other")
    print("=======================================================================================================\n")

    print("We get rid of some events")
    keep_iso = np.zeros(len(total_tru_pt))
    for i in range(len(total_tru_pt)):
        event_i_jet_pt = total_tar_pt[i] 
        event_i_jet_eta = total_tar_eta[i] 
        event_i_jet_phi = total_tar_phi[i] 
        # print(len(event_i_jet_pt),len(event_i_jet_pt[event_i_jet_pt>20]))
        # filter, only count truth jets above 20 GeV
        event_i_jet_eta = event_i_jet_eta[event_i_jet_pt>20]
        event_i_jet_phi = event_i_jet_phi[event_i_jet_pt>20]
        event_i_jet_pt = event_i_jet_pt[event_i_jet_pt>20]

        if len(event_i_jet_pt)>3:
            ind = np.argpartition(event_i_jet_pt,-4)[-4:]
            top4_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top4_idx[3]]
            jet0_eta = event_i_jet_eta[top4_idx[3]]
            jet0_phi = event_i_jet_phi[top4_idx[3]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top4_idx[2]]
            jet1_eta = event_i_jet_eta[top4_idx[2]]
            jet1_phi = event_i_jet_phi[top4_idx[2]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top4_idx[1]]
            jet2_eta = event_i_jet_eta[top4_idx[1]]
            jet2_phi = event_i_jet_phi[top4_idx[1]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            jet3_pt = event_i_jet_pt[top4_idx[0]]
            jet3_eta = event_i_jet_eta[top4_idx[0]]
            jet3_phi = event_i_jet_phi[top4_idx[0]]
            jet3_ = np.array((jet3_eta,jet3_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR03 = np.linalg.norm(jet0_-jet3_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            dR13 = np.linalg.norm(jet1_-jet3_)
            dR23 = np.linalg.norm(jet2_-jet3_)
            # print(jet0_pt,jet1_pt,jet2_pt,jet3_pt)
            # print(dR01,dR02,dR12)
            if (dR01 > 0.6) and (dR02 > 0.6) and (dR03 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>2:
            ind = np.argpartition(event_i_jet_pt,-3)[-3:]
            top3_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top3_idx[2]]
            jet0_eta = event_i_jet_eta[top3_idx[2]]
            jet0_phi = event_i_jet_phi[top3_idx[2]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top3_idx[1]]
            jet1_eta = event_i_jet_eta[top3_idx[1]]
            jet1_phi = event_i_jet_phi[top3_idx[1]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top3_idx[0]]
            jet2_eta = event_i_jet_eta[top3_idx[0]]
            jet2_phi = event_i_jet_phi[top3_idx[0]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            if (dR01 > 0.6) and (dR02 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>1:
            ind = np.argpartition(event_i_jet_pt,-2)[-2:]
            top2_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top2_idx[1]]
            jet0_eta = event_i_jet_eta[top2_idx[1]]
            jet0_phi = event_i_jet_phi[top2_idx[1]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top2_idx[0]]
            jet1_eta = event_i_jet_eta[top2_idx[0]]
            jet1_phi = event_i_jet_phi[top2_idx[0]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            if (dR01 > 0.6):
                keep_iso[i] = 1
        
        else:
            keep_iso[i] = 1

    print('--->',sum(keep_iso),len(keep_iso))
    print("now we can get rid of events where the leading akt jet is close to another top 4 jet (where possible)")
    iso_total_tar_pt = []
    iso_total_tru_pt = []
    iso_total_p_pt = []
    for j in range(len(keep_iso)):
        should_keep = keep_iso[j]
        if should_keep:
            iso_total_tru_pt.append(total_tru_pt[j])
            iso_total_tar_pt.append(total_tar_pt[j])
            iso_total_p_pt.append(total_p_pt[j])

    # Make a trigger decision based on leading antikt jet pt
    tar_lead_pt = np.array([leading_jet_pt(x) for x in iso_total_tar_pt])
    p_lead_pt = np.array([leading_jet_pt(x) for x in iso_total_p_pt])

    lead_jet_pt_cut = 150 # GeV
    trig_decision_akt = np.argwhere(tar_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(p_lead_pt>lead_jet_pt_cut).T[0]

    # Set the x-axis and binning
    start,end = 20,550
    step = 10
    bins = np.arange(start, end, step)

    f,ax = plt.subplots(3,1,figsize=(8,14))
    n_akt,bins,_ = ax[0].hist(tar_lead_pt,bins=bins,histtype='step',label='AntiKt4EMTopo jetConstitScale')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(xlabel="Leading EMTopo (Offline) jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(tar_lead_pt[trig_decision_akt],bins=bins,histtype='step',label='Antikt jetConstitScale',color="limegreen")
    n2_p,_,_ = ax[1].hist(tar_lead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes',color="red")
    ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel="Leading AntiKt4EMTopo jet pT (jet constit. scale) [GeV]",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_akt)
        step_err = get_errorbars(n2_akt,n_akt)

        pred_eff = get_ratio(n2_p,n_akt)
        pred_err = get_errorbars(n2_p,n_akt)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes.',color='red')
    ax[2].set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='lower right')
    # hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/aktiso_leading_{lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

    f,a = plt.subplots(1,1,figsize=(8,8))
    # a.axvline(x=lead_jet_pt_cut,ymin=0.0,ymax=1.0,ls='--',color='black',alpha=0.3, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.vlines(x=[lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"$p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel="Leading AKT4EMTopo jet " + r"$p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start,end),ylim=(-0.2,1.2))
    # a.legend(loc='lower right')
    a.legend(bbox_to_anchor=(0.4,0.025), loc="lower left")
    plt.text(end-295,-0.15, f"AKT4EMTopo jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    plt.text(start+15,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+98,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+15,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start+15,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/aktiso_leading{lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    
    

    #####################################################################################################################################

    print("=======================================================================================================")
    print(f"Making truth leading jet trigger decision, but requiring truth jets be dR > 0.4 away from each other")
    print("=======================================================================================================\n")

    # Make a trigger decision based on leading antikt jet pt
    tru_lead_pt = np.array([leading_jet_pt(x) for x in iso_total_tru_pt])
    tar_lead_pt = np.array([leading_jet_pt(x) for x in iso_total_tar_pt])
    p_lead_pt = np.array([leading_jet_pt(x) for x in iso_total_p_pt])

    trig_decision_tru = np.argwhere(tru_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_akt = np.argwhere(tar_lead_pt>lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(p_lead_pt>lead_jet_pt_cut).T[0]

    # Set the x-axis and binning
    start,end = 20,550
    step = 10
    bins = np.arange(start, end, step)

    f,ax = plt.subplots(3,1,figsize=(8,15.5))
    n_tru,bins,_ = ax[0].hist(tru_lead_pt,bins=bins,histtype='step',label='Truth jets',color='gold')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(tru_lead_pt[trig_decision_akt],bins=bins,histtype='step',label='Antikt jetConstitScale',color="limegreen")
    n2_p,_,_ = ax[1].hist(tru_lead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes',color="red")
    ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Truth jet cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_tru)
        step_err = get_errorbars(n2_akt,n_tru)

        pred_eff = get_ratio(n2_p,n_tru)
        pred_err = get_errorbars(n2_p,n_tru)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt (jet constit. scale)',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred jets (jet constit. scale)',color='red')
    ax[2].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='lower right')
    # hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/aktiso_leading_truth_{lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

    f,a = plt.subplots(1,1,figsize=(8,8))
    # a.axvline(x=lead_jet_pt_cut,ymin=0.0,ymax=1.0,ls='--',color='black',alpha=0.3, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.vlines(x=[lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='AKT4EMTopo',color='limegreen')
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel="Leading truth jet " + r"$p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start,end),ylim=(-0.2,1.2))
    # a.legend(loc='lower right')
    a.legend(bbox_to_anchor=(0.51,0.025), loc="lower left")
    plt.text(start+15,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+100,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+15,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start+15,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    plt.text(end-240,-0.15, f"AKT4EMTopo jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/aktiso_leading_truth{lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")






if subleading_akt_iso_jet:
    total_tar_eta      = load_object(metrics_folder+"/tarboxes_eta.pkl")
    total_tar_phi      = load_object(metrics_folder+"/tarboxes_phi.pkl")
    print("=======================================================================================================")
    print(f"Making subleading jet trigger decision, but requiring AKT jets be dR > 0.6 away from each other")
    print("=======================================================================================================\n")


    print("We get rid of some events")
    keep_iso = np.zeros(len(total_tar_pt))
    for i in range(len(total_tar_pt)):
        event_i_jet_pt = total_tar_pt[i] 
        event_i_jet_eta = total_tar_eta[i] 
        event_i_jet_phi = total_tar_phi[i] 
        # filter, only count truth jets above 20 GeV
        event_i_jet_eta = event_i_jet_eta[event_i_jet_pt>20]
        event_i_jet_phi = event_i_jet_phi[event_i_jet_pt>20]
        event_i_jet_pt = event_i_jet_pt[event_i_jet_pt>20]

        if len(event_i_jet_pt)>3:
            ind = np.argpartition(event_i_jet_pt,-4)[-4:]
            top4_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top4_idx[3]]
            jet0_eta = event_i_jet_eta[top4_idx[3]]
            jet0_phi = event_i_jet_phi[top4_idx[3]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top4_idx[2]]
            jet1_eta = event_i_jet_eta[top4_idx[2]]
            jet1_phi = event_i_jet_phi[top4_idx[2]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top4_idx[1]]
            jet2_eta = event_i_jet_eta[top4_idx[1]]
            jet2_phi = event_i_jet_phi[top4_idx[1]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            jet3_pt = event_i_jet_pt[top4_idx[0]]
            jet3_eta = event_i_jet_eta[top4_idx[0]]
            jet3_phi = event_i_jet_phi[top4_idx[0]]
            jet3_ = np.array((jet3_eta,jet3_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR03 = np.linalg.norm(jet0_-jet3_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            dR13 = np.linalg.norm(jet1_-jet3_)
            dR23 = np.linalg.norm(jet2_-jet3_)
            # print(jet0_pt,jet1_pt,jet2_pt,jet3_pt)
            # print(dR01,dR02,dR12)
            if (dR01 > 0.6) and (dR02 > 0.6) and (dR03 > 0.6) and (dR12 > 0.6) and (dR13 > 0.6) and (dR23 > 0.6):
            # if (dR01 > 0.6) and (dR02 > 0.6) and (dR12 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>2:
            ind = np.argpartition(event_i_jet_pt,-3)[-3:]
            top3_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top3_idx[2]]
            jet0_eta = event_i_jet_eta[top3_idx[2]]
            jet0_phi = event_i_jet_phi[top3_idx[2]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top3_idx[1]]
            jet1_eta = event_i_jet_eta[top3_idx[1]]
            jet1_phi = event_i_jet_phi[top3_idx[1]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top3_idx[0]]
            jet2_eta = event_i_jet_eta[top3_idx[0]]
            jet2_phi = event_i_jet_phi[top3_idx[0]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            if (dR01 > 0.6) and (dR02 > 0.6) and (dR12 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>1:
            ind = np.argpartition(event_i_jet_pt,-2)[-2:]
            top2_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top2_idx[1]]
            jet0_eta = event_i_jet_eta[top2_idx[1]]
            jet0_phi = event_i_jet_phi[top2_idx[1]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top2_idx[0]]
            jet1_eta = event_i_jet_eta[top2_idx[0]]
            jet1_phi = event_i_jet_phi[top2_idx[0]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            if (dR01 > 0.6):
                keep_iso[i] = 1
        
        else:
            keep_iso[i] = 1

    print('-->',sum(keep_iso),len(keep_iso))
    print("now we can get rid of events where the leading truth jet is close to another top 4 jet (where possible)")
    iso_total_tar_pt = []
    iso_total_tru_pt = []
    iso_total_p_pt = []
    for j in range(len(keep_iso)):
        should_keep = keep_iso[j]
        if should_keep:
            iso_total_tru_pt.append(total_tru_pt[j])
            iso_total_tar_pt.append(total_tar_pt[j])
            iso_total_p_pt.append(total_p_pt[j])


    start,end = 20,160
    step = 5
    bins = np.arange(start, end, step)

    nth_jet = 3
    nth_lead_jet_pt_cut = 60 
    tru_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_tru_pt])
    tar_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_tar_pt])
    p_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_p_pt])

    trig_decision_akt  = np.argwhere(tar_nlead_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(p_nlead_pt>nth_lead_jet_pt_cut).T[0]

    f,ax = plt.subplots(3,1,figsize=(6.5,12))
    n_akt,bins,_ = ax[0].hist(tar_nlead_pt,bins=bins,histtype='step',label='Anti-kt jetConstitScale')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
    ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(tar_nlead_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="limegreen")
    n2_pb,_,_ = ax[1].hist(tar_nlead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes Adj',color="red")
    ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_akt)
        step_err = get_errorbars(n2_akt,n_akt)
        pred_eff = get_ratio(n2_pb,n_akt)
        pred_err = get_errorbars(n2_pb,n_akt)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes',color='red')
    ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='upper left')
    hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/aktiso_{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


    f,a = plt.subplots(1,1,figsize=(8,8))
    # a.axvline(x=lead_jet_pt_cut,ymin=0.0,ymax=1.0,ls='--',color='black',alpha=0.3, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.vlines(x=[nth_lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"$p_T$" + f" cut ({nth_lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel=f"${{{nth_jet}}}^{{rd}}$ leading AKT4EMTopo jet " + r" $p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start-10,end),ylim=(-0.2,1.2))
    a.legend(loc='lower right')
    plt.text(end-80,-0.175, f"AKT4EMTopo jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    plt.text(start-3,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+20,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start-3,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start-3,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/aktiso_{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



    print("=======================================================================================================")
    print(f"Making subleading jet trigger decision with TRUTH jets")
    print("=======================================================================================================\n")


    tru_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_tru_pt])
    tar_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_tar_pt])
    p_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_p_pt])

    trig_decision_akt  = np.argwhere(tar_nlead_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(p_nlead_pt>nth_lead_jet_pt_cut).T[0]

    f,ax = plt.subplots(3,1,figsize=(6.5,14))
    n_tru,bins,_ = ax[0].hist(tru_nlead_pt,bins=bins,histtype='step',label='Truth jets', color='gold')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
    ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(tru_nlead_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="limegreen")
    n2_pb,_,_ = ax[1].hist(tru_nlead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes Adj',color="red")
    ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Truth jet cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_tru)
        step_err = get_errorbars(n2_akt,n_tru)
        pred_eff = get_ratio(n2_pb,n_tru)
        pred_err = get_errorbars(n2_pb,n_tru)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt (jet constit. scale)',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred jets (jet constit. scale)',color='red')
    ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Efficiency')
    # ax[2].legend(loc='upper left')
    hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/aktiso_{nth_jet}leading_truth_{nth_lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

    f,a = plt.subplots(1,1,figsize=(8,8))
    # a.axvline(x=lead_jet_pt_cut,ymin=0.0,ymax=1.0,ls='--',color='black',alpha=0.3, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.vlines(x=[nth_lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"Truth $p_T$" + f" cut ({nth_lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='AKT4EMTopo',color='limegreen')
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel=f"${{{nth_jet}}}^{{rd}}$ leading truth jet " + r" $p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start-10,end),ylim=(-0.2,1.2))
    a.legend(loc='lower right')
    plt.text(end-70,-0.175, f"AKT4EMTopo jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    plt.text(start-3,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+20,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start-3,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start-3,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/aktiso_{nth_jet}leading_truth{nth_lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")







if leading_central_jet:
    print("=======================================================================================================")
    print(f"Making leading central jet trigger decision")
    print("=======================================================================================================\n")
    
    total_tar_pt    = load_object(metrics_folder+"/tarboxes_pt.pkl")
    total_tar_eta   = load_object(metrics_folder+"/tarboxes_eta.pkl")
    total_p_pt      = load_object(metrics_folder+"/pboxes_pt.pkl")
    total_p_eta     = load_object(metrics_folder+"/pboxes_eta.pkl")

    # find leading jet & eta
    lead_tar_jet_idx  = np.array([np.argmax(event_tar_pt) for event_tar_pt in total_tar_pt])
    lead_tar_jet_pt   = np.array([total_tar_pt[i][lead_tar_jet_idx[i]] for i in range(len(lead_tar_jet_idx))])
    lead_tar_jet_eta  = np.array([total_tar_eta[i][lead_tar_jet_idx[i]] for i in range(len(lead_tar_jet_idx))])

    lead_pre_jet_idx  = np.array([np.argmax(event_pre_pt) for event_pre_pt in total_p_pt])
    lead_pre_jet_pt   = np.array([total_p_pt[i][lead_pre_jet_idx[i]] for i in range(len(lead_pre_jet_idx))])
    lead_pre_jet_eta  = np.array([total_p_eta[i][lead_pre_jet_idx[i]] for i in range(len(lead_pre_jet_idx))])
    
    # mask out events with truth leding jet outside central
    lead_tar_jet_pt = lead_tar_jet_pt[np.abs(lead_pre_jet_eta) < 0.8]
    lead_pre_jet_pt = lead_pre_jet_pt[np.abs(lead_pre_jet_eta) < 0.8]

    # Make a trigger decision based on leading antikt jet pt
    lead_jet_pt_cut = 150 # GeV
    trig_decision_akt  = np.argwhere(lead_tar_jet_pt>lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(lead_pre_jet_pt>lead_jet_pt_cut).T[0]

    # Set the x-axis and binning
    start,end = 20,550
    step = 10
    bins = np.arange(start, end, step)

    f,ax = plt.subplots(3,1,figsize=(8,14))
    n_akt,bins,_ = ax[0].hist(lead_tar_jet_pt,bins=bins,histtype='step',label='AntiKt4EMTopo jetConstitScale')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(xlabel="Leading EMTopo (Offline) jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(lead_tar_jet_pt[trig_decision_akt],bins=bins,histtype='step',label='Antikt jetConstitScale',color="limegreen")
    n2_p,_,_ = ax[1].hist(lead_tar_jet_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes',color="red")
    ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel="Leading AntiKt4EMTopo jet pT (jet constit. scale) [GeV]",ylabel='Events')
    ax[1].legend()
    ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_akt)
        step_err = get_errorbars(n2_akt,n_akt)
        pred_eff = get_ratio(n2_p,n_akt)
        pred_err = get_errorbars(n2_p,n_akt)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes.',color='red')
    ax[2].set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='lower right')
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/central_leading_{lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


    f,a = plt.subplots(1,1,figsize=(8,8))
    a.vlines(x=[lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"$p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel="Leading AKT4EMTopo jet " + r"$p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start,end),ylim=(-0.2,1.2))
    a.legend(bbox_to_anchor=(0.4,0.025), loc="lower left")
    plt.text(start+15,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+98,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+15,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start+15,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<0.8$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/central_leading{lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")




if leading_central_truth_jet:
    print("=======================================================================================================")
    print(f"Making leading central truth jet trigger decision")
    print("=======================================================================================================\n")
    
    total_tar_pt    = load_object(metrics_folder+"/tarboxes_pt.pkl")
    total_tar_eta   = load_object(metrics_folder+"/tarboxes_eta.pkl")
    total_tru_pt    = load_object(metrics_folder+"/truboxes_pt.pkl")
    total_tru_eta   = load_object(metrics_folder+"/truboxes_eta.pkl")
    total_p_pt      = load_object(metrics_folder+"/pboxes_pt.pkl")
    total_p_eta     = load_object(metrics_folder+"/pboxes_eta.pkl")

    # find leading jet & eta
    lead_tar_jet_idx  = np.array([np.argmax(event_tar_pt) for event_tar_pt in total_tar_pt])
    lead_tar_jet_pt   = np.array([total_tar_pt[i][lead_tar_jet_idx[i]] for i in range(len(lead_tar_jet_idx))])
    lead_tar_jet_eta  = np.array([total_tar_eta[i][lead_tar_jet_idx[i]] for i in range(len(lead_tar_jet_idx))])
    
    lead_tru_jet_idx  = np.array([np.argmax(event_tru_pt) for event_tru_pt in total_tru_pt])
    lead_tru_jet_pt   = np.array([total_tru_pt[i][lead_tru_jet_idx[i]] for i in range(len(lead_tru_jet_idx))])
    lead_tru_jet_eta  = np.array([total_tru_eta[i][lead_tru_jet_idx[i]] for i in range(len(lead_tru_jet_idx))])
    
    lead_pre_jet_idx  = np.array([np.argmax(event_pre_pt) for event_pre_pt in total_p_pt])
    lead_pre_jet_pt   = np.array([total_p_pt[i][lead_pre_jet_idx[i]] for i in range(len(lead_pre_jet_idx))])
    lead_pre_jet_eta  = np.array([total_p_eta[i][lead_pre_jet_idx[i]] for i in range(len(lead_pre_jet_idx))])
    
    # mask out events with truth leding jet outside central
    lead_tar_jet_pt = lead_tar_jet_pt[np.abs(lead_tru_jet_eta) < 0.8]
    lead_tru_jet_pt = lead_tru_jet_pt[np.abs(lead_tru_jet_eta) < 0.8]
    lead_pre_jet_pt = lead_pre_jet_pt[np.abs(lead_tru_jet_eta) < 0.8]

    # Make a trigger decision based on leading antikt jet pt
    lead_jet_pt_cut = 150 # GeV
    trig_decision_tru = np.argwhere(lead_tru_jet_pt>lead_jet_pt_cut).T[0]
    trig_decision_akt = np.argwhere(lead_tar_jet_pt>lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(lead_pre_jet_pt>lead_jet_pt_cut).T[0]

    # Set the x-axis and binning
    start,end = 20,550
    step = 10
    bins = np.arange(start, end, step)

    f,ax = plt.subplots(3,1,figsize=(8,15.5))
    n_tru,bins,_ = ax[0].hist(lead_tru_jet_pt,bins=bins,histtype='step',label='Truth jets',color='gold')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(lead_tru_jet_pt[trig_decision_akt],bins=bins,histtype='step',label='Antikt jetConstitScale',color="limegreen")
    n2_p,_,_ = ax[1].hist(lead_tru_jet_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes',color="red")
    ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Truth jet cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_tru)
        step_err = get_errorbars(n2_akt,n_tru)
        pred_eff = get_ratio(n2_p,n_tru)
        pred_err = get_errorbars(n2_p,n_tru)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt (jet constit. scale)',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred jets (jet constit. scale)',color='red')
    ax[2].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='lower right')
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/central_leading_truth_{lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


    f,a = plt.subplots(1,1,figsize=(8,8))
    a.vlines(x=[lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='AKT4EMTopo',color='limegreen')
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel="Leading truth jet " + r"$p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start,end),ylim=(-0.2,1.2))
    a.legend(bbox_to_anchor=(0.51,0.025), loc="lower left")
    plt.text(start+15,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+98,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+15,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start+15,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<0.8$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/central_leading_truth{lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")





if subleading_central_jet:
    print("=======================================================================================================")
    print(f"Making subleading central jet trigger decision")
    print("=======================================================================================================\n")

    nth_jet = 3
    nth_lead_jet_pt_cut = 60 # GeV
    
    total_tar_pt    = load_object(metrics_folder+"/tarboxes_pt.pkl")
    total_tar_eta   = load_object(metrics_folder+"/tarboxes_eta.pkl")
    total_p_pt      = load_object(metrics_folder+"/pboxes_pt.pkl")
    total_p_eta     = load_object(metrics_folder+"/pboxes_eta.pkl")

    # find leading jet & eta
    sublead_tar_jet_pt  = np.zeros(len(total_tar_pt))
    sublead_tar_jet_eta = np.zeros(len(total_tar_pt))
    for j in range(len(total_tar_pt)):
        event_tar_pt = total_tar_pt[j]
        event_tar_eta = total_tar_eta[j]
        if len(event_tar_pt) < 3:
            continue
        else:
            _, idxs = torch.topk(event_tar_pt,3)
            sublead_tar_jet_idx = idxs[-1]
            sublead_tar_jet_pt[j]  = event_tar_pt[sublead_tar_jet_idx]
            sublead_tar_jet_eta[j] = event_tar_eta[sublead_tar_jet_idx]

    sublead_pre_jet_pt  = np.zeros(len(total_tar_pt))
    sublead_pre_jet_eta = np.zeros(len(total_tar_pt))
    for j in range(len(total_tar_pt)):
        event_pre_pt = total_p_pt[j]
        event_pre_eta = total_p_eta[j]
        if len(event_pre_pt) < 3:
            continue
        else:
            _, idxs = torch.topk(event_pre_pt,3)
            sublead_pre_jet_idx = idxs[-1]
            sublead_pre_jet_pt[j]  = event_pre_pt[sublead_pre_jet_idx]
            sublead_pre_jet_eta[j] = event_pre_eta[sublead_pre_jet_idx]
    
    # mask out events with truth leding jet outside central
    print(len(sublead_tar_jet_pt),len(sublead_pre_jet_pt))
    sublead_tar_jet_pt = sublead_tar_jet_pt[np.abs(sublead_tar_jet_eta) < 0.8]
    sublead_pre_jet_pt = sublead_pre_jet_pt[np.abs(sublead_tar_jet_eta) < 0.8]
    print('-->',len(sublead_tar_jet_pt),len(sublead_pre_jet_pt))

    tru_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_tru_pt])
    tar_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_tar_pt])
    p_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_p_pt])

    ################################

    start,end = 20,160
    step = 5
    bins = np.arange(start, end, step)


    trig_decision_akt  = np.argwhere(sublead_tar_jet_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(sublead_pre_jet_pt>nth_lead_jet_pt_cut).T[0]

    f,ax = plt.subplots(3,1,figsize=(6.5,12))
    n_akt,bins,_ = ax[0].hist(sublead_tar_jet_pt,bins=bins,histtype='step',label='Anti-kt jetConstitScale')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
    ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(sublead_tar_jet_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="limegreen")
    n2_pb,_,_ = ax[1].hist(sublead_tar_jet_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes Adj',color="red")
    ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_akt)
        step_err = get_errorbars(n2_akt,n_akt)
        pred_eff = get_ratio(n2_pb,n_akt)
        pred_err = get_errorbars(n2_pb,n_akt)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes',color='red')
    ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='upper left')
    hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/central_{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


    f,a = plt.subplots(1,1,figsize=(8,8))
    a.vlines(x=[nth_lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"$p_T$" + f" cut ({nth_lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel=f"${{{nth_jet}}}^{{rd}}$ leading AKT4EMTopo jet " + r" $p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start-10,end),ylim=(-0.2,1.2))
    a.legend(loc='lower right')
    plt.text(start-3,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+20,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start-3,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start-3,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<0.8$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/central_{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")





if subleading_central_truth_jet:
    print("=======================================================================================================")
    print(f"Making subleading central truth jet trigger decision")
    print("=======================================================================================================\n")

    nth_jet = 3
    nth_lead_jet_pt_cut = 60 # GeV
    
    total_tar_pt    = load_object(metrics_folder+"/tarboxes_pt.pkl")
    total_tar_eta   = load_object(metrics_folder+"/tarboxes_eta.pkl")
    total_tru_pt    = load_object(metrics_folder+"/truboxes_pt.pkl")
    total_tru_eta   = load_object(metrics_folder+"/truboxes_eta.pkl")
    total_p_pt      = load_object(metrics_folder+"/pboxes_pt.pkl")
    total_p_eta     = load_object(metrics_folder+"/pboxes_eta.pkl")

    # find subleading jet & eta
    sublead_tar_jet_pt  = np.zeros(len(total_tar_pt))
    sublead_tar_jet_eta = np.zeros(len(total_tar_pt))
    for j in range(len(total_tar_pt)):
        event_tar_pt = total_tar_pt[j]
        event_tar_eta = total_tar_eta[j]
        if len(event_tar_pt) < 3:
            continue
        else:
            _, idxs = torch.topk(event_tar_pt,3)
            sublead_tar_jet_idx = idxs[-1]
            sublead_tar_jet_pt[j]  = event_tar_pt[sublead_tar_jet_idx]
            sublead_tar_jet_eta[j] = event_tar_eta[sublead_tar_jet_idx]

    sublead_tru_jet_pt  = np.zeros(len(total_tru_pt))
    sublead_tru_jet_eta = np.zeros(len(total_tru_pt))
    for j in range(len(total_tru_pt)):
        event_tru_pt = total_tru_pt[j]
        event_tru_eta = total_tru_eta[j]
        if len(event_tru_pt) < 3:
            continue
        else:
            _, idxs = torch.topk(event_tru_pt,3)
            sublead_tru_jet_idx = idxs[-1]
            sublead_tru_jet_pt[j]  = event_tru_pt[sublead_tru_jet_idx]
            sublead_tru_jet_eta[j] = event_tru_eta[sublead_tru_jet_idx]

    sublead_pre_jet_pt  = np.zeros(len(total_tar_pt))
    sublead_pre_jet_eta = np.zeros(len(total_tar_pt))
    for j in range(len(total_tar_pt)):
        event_pre_pt = total_p_pt[j]
        event_pre_eta = total_p_eta[j]
        if len(event_pre_pt) < 3:
            continue
        else:
            _, idxs = torch.topk(event_pre_pt,3)
            sublead_pre_jet_idx = idxs[-1]
            sublead_pre_jet_pt[j]  = event_pre_pt[sublead_pre_jet_idx]
            sublead_pre_jet_eta[j] = event_pre_eta[sublead_pre_jet_idx]
    
    # mask out events with truth leding jet outside central
    print(len(sublead_tar_jet_pt),len(sublead_tru_jet_pt),len(sublead_pre_jet_pt))
    sublead_tar_jet_pt = sublead_tar_jet_pt[np.abs(sublead_tru_jet_eta) < 0.8]
    sublead_tru_jet_pt = sublead_tru_jet_pt[np.abs(sublead_tru_jet_eta) < 0.8]
    sublead_pre_jet_pt = sublead_pre_jet_pt[np.abs(sublead_tru_jet_eta) < 0.8]
    print('-->',len(sublead_tar_jet_pt),len(sublead_tru_jet_pt),len(sublead_pre_jet_pt))

    ################################

    start,end = 20,160
    step = 5
    bins = np.arange(start, end, step)

    trig_decision_akt  = np.argwhere(sublead_tar_jet_pt>nth_lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(sublead_pre_jet_pt>nth_lead_jet_pt_cut).T[0]

    f,ax = plt.subplots(3,1,figsize=(6.5,14))
    n_tru,bins,_ = ax[0].hist(sublead_tru_jet_pt,bins=bins,histtype='step',label='Truth jets', color='gold')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
    ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(sublead_tru_jet_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="limegreen")
    n2_pb,_,_ = ax[1].hist(sublead_tru_jet_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes Adj',color="red")
    ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Truth jet cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_tru)
        step_err = get_errorbars(n2_akt,n_tru)
        pred_eff = get_ratio(n2_pb,n_tru)
        pred_err = get_errorbars(n2_pb,n_tru)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt (jet constit. scale)',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred jets (jet constit. scale)',color='red')
    ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Efficiency')
    hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/central_{nth_jet}leading_truth_{nth_lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


    f,a = plt.subplots(1,1,figsize=(8,8))
    a.vlines(x=[nth_lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"Truth $p_T$" + f" cut ({nth_lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='AKT4EMTopo',color='limegreen')
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel=f"${{{nth_jet}}}^{{rd}}$ leading truth jet " + r" $p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start-10,end),ylim=(-0.2,1.2))
    a.legend(loc='lower right')
    plt.text(start-3,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+20,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start-3,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start-3,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<0.8$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/central_{nth_jet}leading_truth{nth_lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")











if leading_central_akt_iso_jet:
  
    total_tar_pt    = load_object(metrics_folder+"/tarboxes_pt.pkl")
    total_tar_eta   = load_object(metrics_folder+"/tarboxes_eta.pkl")
    total_tar_phi      = load_object(metrics_folder+"/tarboxes_phi.pkl")
    total_tru_pt    = load_object(metrics_folder+"/truboxes_pt.pkl")
    total_tru_eta   = load_object(metrics_folder+"/truboxes_eta.pkl")
    total_p_pt      = load_object(metrics_folder+"/pboxes_pt.pkl")
    total_p_eta     = load_object(metrics_folder+"/pboxes_eta.pkl")
    print("=======================================================================================================")
    print(f"Making leading central jet trigger decision, but requiring AKT jets be dR > 0.6 away from each other")
    print("=======================================================================================================\n")

    print("We get rid of some events")
    keep_iso = np.zeros(len(total_tru_pt))
    for i in range(len(total_tru_pt)):
        event_i_jet_pt = total_tar_pt[i] 
        event_i_jet_eta = total_tar_eta[i] 
        event_i_jet_phi = total_tar_phi[i] 
        # filter, only count truth jets above 20 GeV
        event_i_jet_eta = event_i_jet_eta[event_i_jet_pt>20]
        event_i_jet_phi = event_i_jet_phi[event_i_jet_pt>20]
        event_i_jet_pt = event_i_jet_pt[event_i_jet_pt>20]

        if len(event_i_jet_pt)>3:
            ind = np.argpartition(event_i_jet_pt,-4)[-4:]
            top4_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top4_idx[3]]
            jet0_eta = event_i_jet_eta[top4_idx[3]]
            jet0_phi = event_i_jet_phi[top4_idx[3]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top4_idx[2]]
            jet1_eta = event_i_jet_eta[top4_idx[2]]
            jet1_phi = event_i_jet_phi[top4_idx[2]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top4_idx[1]]
            jet2_eta = event_i_jet_eta[top4_idx[1]]
            jet2_phi = event_i_jet_phi[top4_idx[1]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            jet3_pt = event_i_jet_pt[top4_idx[0]]
            jet3_eta = event_i_jet_eta[top4_idx[0]]
            jet3_phi = event_i_jet_phi[top4_idx[0]]
            jet3_ = np.array((jet3_eta,jet3_phi))

            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR03 = np.linalg.norm(jet0_-jet3_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            dR13 = np.linalg.norm(jet1_-jet3_)
            dR23 = np.linalg.norm(jet2_-jet3_)
            if (dR01 > 0.6) and (dR02 > 0.6) and (dR03 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>2:
            ind = np.argpartition(event_i_jet_pt,-3)[-3:]
            top3_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top3_idx[2]]
            jet0_eta = event_i_jet_eta[top3_idx[2]]
            jet0_phi = event_i_jet_phi[top3_idx[2]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top3_idx[1]]
            jet1_eta = event_i_jet_eta[top3_idx[1]]
            jet1_phi = event_i_jet_phi[top3_idx[1]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top3_idx[0]]
            jet2_eta = event_i_jet_eta[top3_idx[0]]
            jet2_phi = event_i_jet_phi[top3_idx[0]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            if (dR01 > 0.6) and (dR02 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>1:
            ind = np.argpartition(event_i_jet_pt,-2)[-2:]
            top2_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top2_idx[1]]
            jet0_eta = event_i_jet_eta[top2_idx[1]]
            jet0_phi = event_i_jet_phi[top2_idx[1]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top2_idx[0]]
            jet1_eta = event_i_jet_eta[top2_idx[0]]
            jet1_phi = event_i_jet_phi[top2_idx[0]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            dR01 = np.linalg.norm(jet0_-jet1_)
            if (dR01 > 0.6):
                keep_iso[i] = 1
        
        else:
            keep_iso[i] = 1

    print('--->',sum(keep_iso),len(keep_iso))
    print("now we can get rid of events where the leading akt jet is close to another top 4 jet (where possible)")
    iso_total_tar_pt,iso_total_tar_eta = [],[]
    iso_total_tru_pt,iso_total_tru_eta = [],[]
    iso_total_p_pt,iso_total_p_eta = [],[]
    for j in range(len(keep_iso)):
        should_keep = keep_iso[j]
        if should_keep:
            iso_total_tru_pt.append(total_tru_pt[j])
            iso_total_tru_eta.append(total_tru_eta[j])
            iso_total_tar_pt.append(total_tar_pt[j])
            iso_total_tar_eta.append(total_tar_eta[j])
            iso_total_p_pt.append(total_p_pt[j])
            iso_total_p_eta.append(total_p_eta[j])

    
    # remove events with non-central leading jet (|eta| < 0.8)
    # find leading jet & eta
    lead_iso_tar_jet_idx  = np.array([np.argmax(event_tar_pt) for event_tar_pt in iso_total_tar_pt])
    lead_iso_tar_jet_pt   = np.array([iso_total_tar_pt[i][lead_iso_tar_jet_idx[i]] for i in range(len(lead_iso_tar_jet_idx))])
    lead_iso_tar_jet_eta  = np.array([iso_total_tar_eta[i][lead_iso_tar_jet_idx[i]] for i in range(len(lead_iso_tar_jet_idx))])
    
    lead_iso_tru_jet_idx  = np.array([np.argmax(event_tru_pt) for event_tru_pt in iso_total_tru_pt])
    lead_iso_tru_jet_pt   = np.array([iso_total_tru_pt[i][lead_iso_tru_jet_idx[i]] for i in range(len(lead_iso_tru_jet_idx))])
    lead_iso_tru_jet_eta  = np.array([iso_total_tru_eta[i][lead_iso_tru_jet_idx[i]] for i in range(len(lead_iso_tru_jet_idx))])
    
    lead_iso_pre_jet_idx  = np.array([np.argmax(event_pre_pt) for event_pre_pt in iso_total_p_pt])
    lead_iso_pre_jet_pt   = np.array([iso_total_p_pt[i][lead_iso_pre_jet_idx[i]] for i in range(len(lead_iso_pre_jet_idx))])
    lead_iso_pre_jet_eta  = np.array([iso_total_p_eta[i][lead_iso_pre_jet_idx[i]] for i in range(len(lead_iso_pre_jet_idx))])
    
    # mask out events with truth leding jet outside central
    lead_iso_tar_jet_pt1 = lead_iso_tar_jet_pt[np.abs(lead_iso_tar_jet_eta) < 0.8]
    lead_iso_tru_jet_pt1 = lead_iso_tru_jet_pt[np.abs(lead_iso_tar_jet_eta) < 0.8]
    lead_iso_pre_jet_pt1 = lead_iso_pre_jet_pt[np.abs(lead_iso_tar_jet_eta) < 0.8]

    # Make a trigger decision based on leading antikt jet pt
    lead_jet_pt_cut = 150 # GeV
    trig_decision_akt = np.argwhere(lead_iso_tar_jet_pt1>lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(lead_iso_pre_jet_pt1>lead_jet_pt_cut).T[0]

    # Set the x-axis and binning
    start,end = 20,550
    step = 10
    bins = np.arange(start, end, step)

    f,ax = plt.subplots(3,1,figsize=(8,14))
    n_akt,bins,_ = ax[0].hist(lead_iso_tar_jet_pt1,bins=bins,histtype='step',label='AntiKt4EMTopo jetConstitScale')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(xlabel="Leading EMTopo (Offline) jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(lead_iso_tar_jet_pt1[trig_decision_akt],bins=bins,histtype='step',label='Antikt jetConstitScale',color="limegreen")
    n2_p,_,_ = ax[1].hist(lead_iso_tar_jet_pt1[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes',color="red")
    ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel="Leading AntiKt4EMTopo jet pT (jet constit. scale) [GeV]",ylabel='Events')
    ax[1].legend()
    ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_akt)
        step_err = get_errorbars(n2_akt,n_akt)
        pred_eff = get_ratio(n2_p,n_akt)
        pred_err = get_errorbars(n2_p,n_akt)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes.',color='red')
    ax[2].set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='lower right')
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/aktiso_central_leading_{lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

    f,a = plt.subplots(1,1,figsize=(8,8))
    a.vlines(x=[lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"$p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel="Leading AKT4EMTopo jet " + r"$p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start,end),ylim=(-0.2,1.2))
    a.legend(bbox_to_anchor=(0.4,0.025), loc="lower left")
    plt.text(end-295,-0.15, f"AKT4EMTopo jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    plt.text(start+15,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+98,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+15,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start+15,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<0.8$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/aktiso_central_leading{lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    

    #####################################################################################################################################

    print("=======================================================================================================")
    print(f"Making truth leading jet trigger decision, but requiring truth jets be dR > 0.4 away from each other")
    print("=======================================================================================================\n")
    # mask out events with truth leding jet outside central
    lead_iso_tar_jet_pt2 = lead_iso_tar_jet_pt[np.abs(lead_iso_tru_jet_eta) < 0.8]
    lead_iso_tru_jet_pt2 = lead_iso_tru_jet_pt[np.abs(lead_iso_tru_jet_eta) < 0.8]
    lead_iso_pre_jet_pt2 = lead_iso_pre_jet_pt[np.abs(lead_iso_tru_jet_eta) < 0.8]

    trig_decision_tru = np.argwhere(lead_iso_tru_jet_pt2>lead_jet_pt_cut).T[0]
    trig_decision_akt = np.argwhere(lead_iso_tar_jet_pt2>lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(lead_iso_pre_jet_pt2>lead_jet_pt_cut).T[0]

    # Set the x-axis and binning
    start,end = 20,550
    step = 10
    bins = np.arange(start, end, step)

    f,ax = plt.subplots(3,1,figsize=(8,15.5))
    n_tru,bins,_ = ax[0].hist(lead_iso_tru_jet_pt2,bins=bins,histtype='step',label='Truth jets',color='gold')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(lead_iso_tru_jet_pt2[trig_decision_akt],bins=bins,histtype='step',label='Antikt jetConstitScale',color="limegreen")
    n2_p,_,_ = ax[1].hist(lead_iso_tru_jet_pt2[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes',color="red")
    ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Events')
    ax[1].legend()
    ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Truth jet cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_tru)
        step_err = get_errorbars(n2_akt,n_tru)
        pred_eff = get_ratio(n2_p,n_tru)
        pred_err = get_errorbars(n2_p,n_tru)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt (jet constit. scale)',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred jets (jet constit. scale)',color='red')
    ax[2].set(xlabel="Leading Truth jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='lower right')
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/aktiso_central_leading_truth_{lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

    f,a = plt.subplots(1,1,figsize=(8,8))
    a.vlines(x=[lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"Truth $p_T$" + f" cut ({lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='AKT4EMTopo',color='limegreen')
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel="Leading truth jet " + r"$p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start,end),ylim=(-0.2,1.2))
    a.legend(bbox_to_anchor=(0.51,0.025), loc="lower left")
    plt.text(start+15,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+100,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+15,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start+15,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<0.8$",fontfamily='sans-serif',fontsize=12)
    plt.text(end-240,-0.15, f"AKT4EMTopo jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/aktiso_central_leading_truth{lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")







if subleading_central_akt_iso_jet:
    total_tar_pt    = load_object(metrics_folder+"/tarboxes_pt.pkl")
    total_tar_eta   = load_object(metrics_folder+"/tarboxes_eta.pkl")
    total_tar_phi      = load_object(metrics_folder+"/tarboxes_phi.pkl")
    total_tru_pt    = load_object(metrics_folder+"/truboxes_pt.pkl")
    total_tru_eta   = load_object(metrics_folder+"/truboxes_eta.pkl")
    total_p_pt      = load_object(metrics_folder+"/pboxes_pt.pkl")
    total_p_eta     = load_object(metrics_folder+"/pboxes_eta.pkl")
    print("=======================================================================================================")
    print(f"Making subleading jet trigger decision, but requiring AKT jets be dR > 0.6 away from each other")
    print("=======================================================================================================\n")

    print("We get rid of some events")
    keep_iso = np.zeros(len(total_tar_pt))
    for i in range(len(total_tar_pt)):
        event_i_jet_pt = total_tar_pt[i] 
        event_i_jet_eta = total_tar_eta[i] 
        event_i_jet_phi = total_tar_phi[i] 
        # filter, only count truth jets above 20 GeV
        event_i_jet_eta = event_i_jet_eta[event_i_jet_pt>20]
        event_i_jet_phi = event_i_jet_phi[event_i_jet_pt>20]
        event_i_jet_pt = event_i_jet_pt[event_i_jet_pt>20]

        if len(event_i_jet_pt)>3:
            ind = np.argpartition(event_i_jet_pt,-4)[-4:]
            top4_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top4_idx[3]]
            jet0_eta = event_i_jet_eta[top4_idx[3]]
            jet0_phi = event_i_jet_phi[top4_idx[3]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top4_idx[2]]
            jet1_eta = event_i_jet_eta[top4_idx[2]]
            jet1_phi = event_i_jet_phi[top4_idx[2]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top4_idx[1]]
            jet2_eta = event_i_jet_eta[top4_idx[1]]
            jet2_phi = event_i_jet_phi[top4_idx[1]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            jet3_pt = event_i_jet_pt[top4_idx[0]]
            jet3_eta = event_i_jet_eta[top4_idx[0]]
            jet3_phi = event_i_jet_phi[top4_idx[0]]
            jet3_ = np.array((jet3_eta,jet3_phi))

            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR03 = np.linalg.norm(jet0_-jet3_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            dR13 = np.linalg.norm(jet1_-jet3_)
            dR23 = np.linalg.norm(jet2_-jet3_)
            if (dR01 > 0.6) and (dR02 > 0.6) and (dR03 > 0.6) and (dR12 > 0.6) and (dR13 > 0.6) and (dR23 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>2:
            ind = np.argpartition(event_i_jet_pt,-3)[-3:]
            top3_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top3_idx[2]]
            jet0_eta = event_i_jet_eta[top3_idx[2]]
            jet0_phi = event_i_jet_phi[top3_idx[2]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top3_idx[1]]
            jet1_eta = event_i_jet_eta[top3_idx[1]]
            jet1_phi = event_i_jet_phi[top3_idx[1]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top3_idx[0]]
            jet2_eta = event_i_jet_eta[top3_idx[0]]
            jet2_phi = event_i_jet_phi[top3_idx[0]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            if (dR01 > 0.6) and (dR02 > 0.6) and (dR12 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>1:
            ind = np.argpartition(event_i_jet_pt,-2)[-2:]
            top2_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top2_idx[1]]
            jet0_eta = event_i_jet_eta[top2_idx[1]]
            jet0_phi = event_i_jet_phi[top2_idx[1]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top2_idx[0]]
            jet1_eta = event_i_jet_eta[top2_idx[0]]
            jet1_phi = event_i_jet_phi[top2_idx[0]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            dR01 = np.linalg.norm(jet0_-jet1_)
            if (dR01 > 0.6):
                keep_iso[i] = 1
        
        else:
            keep_iso[i] = 1

    print('--->',sum(keep_iso),len(keep_iso))
    print("now we can get rid of events where the leading akt jet is close to another top 4 jet (where possible)")
    iso_total_tar_pt,iso_total_tar_eta = [],[]
    iso_total_tru_pt,iso_total_tru_eta = [],[]
    iso_total_p_pt,iso_total_p_eta = [],[]
    for j in range(len(keep_iso)):
        should_keep = keep_iso[j]
        if should_keep:
            iso_total_tru_pt.append(total_tru_pt[j])
            iso_total_tru_eta.append(total_tru_eta[j])
            iso_total_tar_pt.append(total_tar_pt[j])
            iso_total_tar_eta.append(total_tar_eta[j])
            iso_total_p_pt.append(total_p_pt[j])
            iso_total_p_eta.append(total_p_eta[j])


    # find subleading jet & eta
    sublead_tar_jet_pt  = np.zeros(len(iso_total_tar_pt))
    sublead_tar_jet_eta = np.zeros(len(iso_total_tar_pt))
    for j in range(len(iso_total_tar_pt)):
        event_tar_pt = iso_total_tar_pt[j]
        event_tar_eta = iso_total_tar_eta[j]
        if len(event_tar_pt) < 3:
            continue
        else:
            _, idxs = torch.topk(event_tar_pt,3)
            sublead_tar_jet_idx = idxs[-1]
            sublead_tar_jet_pt[j]  = event_tar_pt[sublead_tar_jet_idx]
            sublead_tar_jet_eta[j] = event_tar_eta[sublead_tar_jet_idx]

    sublead_tru_jet_pt  = np.zeros(len(iso_total_tru_pt))
    sublead_tru_jet_eta = np.zeros(len(iso_total_tru_pt))
    for j in range(len(iso_total_tru_pt)):
        event_tru_pt = iso_total_tru_pt[j]
        event_tru_eta = iso_total_tru_eta[j]
        if len(event_tru_pt) < 3:
            continue
        else:
            _, idxs = torch.topk(event_tru_pt,3)
            sublead_tru_jet_idx = idxs[-1]
            sublead_tru_jet_pt[j]  = event_tru_pt[sublead_tru_jet_idx]
            sublead_tru_jet_eta[j] = event_tru_eta[sublead_tru_jet_idx]

    sublead_pre_jet_pt  = np.zeros(len(iso_total_p_pt))
    sublead_pre_jet_eta = np.zeros(len(iso_total_p_pt))
    for j in range(len(iso_total_p_pt)):
        event_pre_pt = iso_total_p_pt[j]
        event_pre_eta = iso_total_p_eta[j]
        if len(event_pre_pt) < 3:
            continue
        else:
            _, idxs = torch.topk(event_pre_pt,3)
            sublead_pre_jet_idx = idxs[-1]
            sublead_pre_jet_pt[j]  = event_pre_pt[sublead_pre_jet_idx]
            sublead_pre_jet_eta[j] = event_pre_eta[sublead_pre_jet_idx]
    
    # mask out events with truth leding jet outside central
    sublead_iso_tar_jet_pt1 = sublead_tar_jet_pt[np.abs(sublead_tar_jet_eta) < 0.8]
    sublead_iso_tru_jet_pt1 = sublead_tru_jet_pt[np.abs(sublead_tar_jet_eta) < 0.8]
    sublead_iso_pre_jet_pt1 = sublead_pre_jet_pt[np.abs(sublead_tar_jet_eta) < 0.8]


    # Make a trigger decision based on leading antikt jet pt
    nth_jet = 3
    nth_lead_jet_pt_cut = 60 
    trig_decision_akt = np.argwhere(sublead_iso_tar_jet_pt1>nth_lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(sublead_iso_pre_jet_pt1>nth_lead_jet_pt_cut).T[0]

    ###################################

    start,end = 20,160
    step = 5
    bins = np.arange(start, end, step)

    f,ax = plt.subplots(3,1,figsize=(6.5,12))
    n_akt,bins,_ = ax[0].hist(sublead_iso_tar_jet_pt1,bins=bins,histtype='step',label='Anti-kt jetConstitScale')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
    ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(sublead_iso_tar_jet_pt1[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="limegreen")
    n2_pb,_,_ = ax[1].hist(sublead_iso_tar_jet_pt1[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes Adj',color="red")
    ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Events')
    ax[1].legend()

    ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_akt)
        step_err = get_errorbars(n2_akt,n_akt)
        pred_eff = get_ratio(n2_pb,n_akt)
        pred_err = get_errorbars(n2_pb,n_akt)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes',color='red')
    ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
    ax[2].legend(loc='upper left')
    hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/aktiso_central_{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


    f,a = plt.subplots(1,1,figsize=(8,8))
    a.vlines(x=[nth_lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"$p_T$" + f" cut ({nth_lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel=f"${{{nth_jet}}}^{{rd}}$ leading AKT4EMTopo jet " + r" $p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start-10,end),ylim=(-0.2,1.2))
    a.legend(loc='lower right')
    plt.text(end-80,-0.175, f"AKT4EMTopo jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    plt.text(start-3,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+20,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start-3,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start-3,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<0.8$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/aktiso_central_{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



    print("=======================================================================================================")
    print(f"Making subleading central jet trigger decision with TRUTH jets")
    print("=======================================================================================================\n")
    # mask out events with truth leding jet outside central
    sublead_iso_tar_jet_pt2 = sublead_tar_jet_pt[np.abs(sublead_tru_jet_eta) < 0.8]
    sublead_iso_tru_jet_pt2 = sublead_tru_jet_pt[np.abs(sublead_tru_jet_eta) < 0.8]
    sublead_iso_pre_jet_pt2 = sublead_pre_jet_pt[np.abs(sublead_tru_jet_eta) < 0.8]

    # Make a trigger decision based on leading antikt jet pt
    lead_jet_pt_cut = 150 # GeV
    trig_decision_akt = np.argwhere(sublead_iso_tar_jet_pt2>nth_lead_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(sublead_iso_pre_jet_pt2>nth_lead_jet_pt_cut).T[0]


    f,ax = plt.subplots(3,1,figsize=(6.5,14))
    n_tru,bins,_ = ax[0].hist(sublead_iso_tru_jet_pt2,bins=bins,histtype='step',label='Truth jets', color='gold')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
    ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Events')
    ax[0].legend()

    n2_akt,bins,_ = ax[1].hist(sublead_iso_tru_jet_pt2[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="limegreen")
    n2_pb,_,_ = ax[1].hist(sublead_iso_tru_jet_pt2[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes Adj',color="red")
    ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
    ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
    ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Events')
    ax[1].legend()
    ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Truth jet cut')
    with np.errstate(divide='ignore', invalid='ignore'):
        step_eff = get_ratio(n2_akt,n_tru)
        step_err = get_errorbars(n2_akt,n_tru)
        pred_eff = get_ratio(n2_pb,n_tru)
        pred_err = get_errorbars(n2_pb,n_tru)

    ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt (jet constit. scale)',color='limegreen')
    ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred jets (jet constit. scale)',color='red')
    ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading Truth jet pT (GeV)",ylabel='Efficiency')
    hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.4)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    f.savefig(save_folder + f'/aktiso_central_{nth_jet}leading_truth_{nth_lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

    f,a = plt.subplots(1,1,figsize=(8,8))
    a.vlines(x=[nth_lead_jet_pt_cut], ymin=[-0.2], ymax=[1.02], colors='black', ls='--',alpha=0.6, label=r"Truth $p_T$" + f" cut ({nth_lead_jet_pt_cut} GeV)")
    a.errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='AKT4EMTopo',color='limegreen')
    a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='CaloJetSSD',color='red')
    a.set(xlabel=f"${{{nth_jet}}}^{{rd}}$ leading truth jet " + r" $p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start-10,end),ylim=(-0.2,1.2))
    a.legend(loc='lower right')
    plt.text(end-70,-0.175, f"AKT4EMTopo jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    plt.text(start-3,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+20,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start-3,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start-3,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<0.8$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/aktiso_central_{nth_jet}leading_truth{nth_lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")




if leading_jet_multi_cut:
    print("=======================================================================================================")
    print(f"Making leading jet trigger decision")
    print("=======================================================================================================\n")
    # Make a trigger decision based on leading antikt jet pt
    tar_lead_pt = np.array([leading_jet_pt(x) for x in total_tar_pt])
    p_lead_pt = np.array([leading_jet_pt(x) for x in total_p_pt])


    # Set the x-axis and binning
    start,end = 20,450
    step = 10
    bins = np.arange(start, end, step)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    

    # trigger decision 1
    lead_jet_pt_cut1 = 150 # GeV
    trig_decision_akt1 = np.argwhere(tar_lead_pt>lead_jet_pt_cut1).T[0]
    trig_decision_pred1 = np.argwhere(p_lead_pt>lead_jet_pt_cut1).T[0]
    h_akt1, bedges = np.histogram(tar_lead_pt,bins=bins)
    h2_akt1, bedges = np.histogram(tar_lead_pt[trig_decision_akt1],bins=bins)
    h2_pre1, bedges = np.histogram(tar_lead_pt[trig_decision_pred1],bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_akt_eff = get_ratio(h2_akt1,h_akt1)
        h_akt_err = get_errorbars(h2_akt1,h_akt1)
        h_pre_eff1 = get_ratio(h2_pre1,h_akt1)
        h_pre_err1 = get_errorbars(h2_pre1,h_akt1)

    # trigger decision 2
    lead_jet_pt_cut2 = 225 # GeV
    trig_decision_akt2  = np.argwhere(tar_lead_pt>lead_jet_pt_cut2).T[0]
    trig_decision_pred2 = np.argwhere(p_lead_pt>lead_jet_pt_cut2).T[0]
    h_akt2, bedges = np.histogram(tar_lead_pt,bins=bins)
    h2_akt2, bedges = np.histogram(tar_lead_pt[trig_decision_akt2],bins=bins)
    h2_pre2, bedges = np.histogram(tar_lead_pt[trig_decision_pred2],bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_pre_eff2 = get_ratio(h2_pre2,h_akt2)
        h_pre_err2 = get_errorbars(h2_pre2,h_akt2)

    # trigger decision 3
    lead_jet_pt_cut3 = 300 # GeV
    trig_decision_akt3  = np.argwhere(tar_lead_pt>lead_jet_pt_cut3).T[0]
    trig_decision_pred3 = np.argwhere(p_lead_pt>lead_jet_pt_cut3).T[0]
    h_akt3, bedges = np.histogram(tar_lead_pt,bins=bins)
    h2_akt2, bedges = np.histogram(tar_lead_pt[trig_decision_akt3],bins=bins)
    h2_pre3, bedges = np.histogram(tar_lead_pt[trig_decision_pred3],bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_pre_eff3 = get_ratio(h2_pre3,h_akt3)
        h_pre_err3 = get_errorbars(h2_pre3,h_akt3)


    f,a = plt.subplots(1,1,figsize=(8,8))
    a.vlines(x=[lead_jet_pt_cut1], ymin=[-0.2], ymax=[1.02], colors='lightsalmon', ls='--',alpha=0.7)
    a.errorbar(bin_centers,h_pre_eff1,xerr=bin_width/2,yerr=h_pre_err1,elinewidth=0.4,marker='.',ls='none',label=f'{lead_jet_pt_cut1} GeV',color='lightsalmon')
    a.vlines(x=[lead_jet_pt_cut2], ymin=[-0.2], ymax=[1.02], colors='red', ls='--',alpha=0.7)
    a.errorbar(bin_centers,h_pre_eff2,xerr=bin_width/2,yerr=h_pre_err2,elinewidth=0.4,marker='.',ls='none',label=f'{lead_jet_pt_cut2} GeV',color='red')
    a.vlines(x=[lead_jet_pt_cut3], ymin=[-0.2], ymax=[1.02], colors='maroon', ls='--',alpha=0.7)
    a.errorbar(bin_centers,h_pre_eff3,xerr=bin_width/2,yerr=h_pre_err3,elinewidth=0.4,marker='.',ls='none',label=f'{lead_jet_pt_cut3} GeV',color='maroon')
    a.set(xlabel="Leading AKT4EMTopo jet " + r"$p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start,end),ylim=(-0.2,1.2))
    a.legend(bbox_to_anchor=(0.7,0.025), loc="lower left")
    plt.text(start+15,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+98,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+15,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start+15,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/leading_multi_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")







if subleading_jet_multi_cut:
    print("=======================================================================================================")
    print(f"Making multiple subleading jet trigger decision")
    print("=======================================================================================================\n")

    start,end = 20,140
    step = 5 #2.5
    bins = np.arange(start, end, step)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]



    nth_jet = 3
    tru_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_tru_pt])
    tar_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_tar_pt])
    p_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_p_pt])

    # trigger decision 1
    nth_lead_jet_pt_cut1 = 40 # GeV
    trig_decision_akt1  = np.argwhere(tar_nlead_pt>nth_lead_jet_pt_cut1).T[0]
    trig_decision_pred1 = np.argwhere(p_nlead_pt>nth_lead_jet_pt_cut1).T[0]
    h_akt1, bedges = np.histogram(tar_nlead_pt,bins=bins)
    h2_akt1, bedges = np.histogram(tar_nlead_pt[trig_decision_akt1],bins=bins)
    h2_pre1, bedges = np.histogram(tar_nlead_pt[trig_decision_pred1],bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_akt_eff = get_ratio(h2_akt1,h_akt1)
        h_akt_err = get_errorbars(h2_akt1,h_akt1)
        h_pre_eff1 = get_ratio(h2_pre1,h_akt1)
        h_pre_err1 = get_errorbars(h2_pre1,h_akt1)

    # trigger decision 2
    nth_lead_jet_pt_cut2 = 60 # GeV
    trig_decision_akt2  = np.argwhere(tar_nlead_pt>nth_lead_jet_pt_cut2).T[0]
    trig_decision_pred2 = np.argwhere(p_nlead_pt>nth_lead_jet_pt_cut2).T[0]
    h_akt2, bedges = np.histogram(tar_nlead_pt,bins=bins)
    h2_akt2, bedges = np.histogram(tar_nlead_pt[trig_decision_akt2],bins=bins)
    h2_pre2, bedges = np.histogram(tar_nlead_pt[trig_decision_pred2],bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_pre_eff2 = get_ratio(h2_pre2,h_akt2)
        h_pre_err2 = get_errorbars(h2_pre2,h_akt2)

    # trigger decision 2
    nth_lead_jet_pt_cut3 = 80 # GeV
    trig_decision_akt3  = np.argwhere(tar_nlead_pt>nth_lead_jet_pt_cut3).T[0]
    trig_decision_pred3 = np.argwhere(p_nlead_pt>nth_lead_jet_pt_cut3).T[0]
    h_akt3, bedges = np.histogram(tar_nlead_pt,bins=bins)
    h2_akt2, bedges = np.histogram(tar_nlead_pt[trig_decision_akt3],bins=bins)
    h2_pre3, bedges = np.histogram(tar_nlead_pt[trig_decision_pred3],bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_pre_eff3 = get_ratio(h2_pre3,h_akt3)
        h_pre_err3 = get_errorbars(h2_pre3,h_akt3)

    f,a = plt.subplots(1,1,figsize=(8,8))
    a.vlines(x=[nth_lead_jet_pt_cut1], ymin=[-0.2], ymax=[1.02], colors='lightsalmon', ls='--',alpha=0.9)
    a.errorbar(bin_centers,h_pre_eff1,xerr=bin_width/2,yerr=h_pre_err1,elinewidth=0.4,marker='.',ls='none',label=f'{nth_lead_jet_pt_cut1} GeV',color='lightsalmon')
    a.vlines(x=[nth_lead_jet_pt_cut2], ymin=[-0.2], ymax=[1.02], colors='red', ls='--',alpha=0.9)
    a.errorbar(bin_centers,h_pre_eff2,xerr=bin_width/2,yerr=h_pre_err2,elinewidth=0.4,marker='.',ls='none',label=f'{nth_lead_jet_pt_cut2} GeV',color='red')
    a.vlines(x=[nth_lead_jet_pt_cut3], ymin=[-0.2], ymax=[1.02], colors='maroon', ls='--',alpha=0.9)
    a.errorbar(bin_centers,h_pre_eff3,xerr=bin_width/2,yerr=h_pre_err3,elinewidth=0.4,marker='.',ls='none',label=f'{nth_lead_jet_pt_cut3} GeV',color='maroon')
    a.set(xlabel=f"${{{nth_jet}}}^{{rd}}$ leading AKT4EMTopo jet " + r" $p_T$ [GeV]",ylabel='Trigger Efficiency',xlim=(start-10,end),ylim=(-0.2,1.2))
    a.legend(loc='lower right')
    plt.text(start-3,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(start+20,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start-3,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start-3,1.03, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/{nth_jet}leading_multi_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")







if leading_akt_iso_jet_multi_cut:
    total_tar_eta      = load_object(metrics_folder+"/tarboxes_eta.pkl")
    total_tar_phi      = load_object(metrics_folder+"/tarboxes_phi.pkl")
    print("=======================================================================================================")
    print(f"Making leading jet trigger decision, but requiring AKT jets be dR > 0.6 away from each other")
    print("=======================================================================================================\n")

    print("We get rid of some events")
    keep_iso = np.zeros(len(total_tru_pt))
    for i in range(len(total_tru_pt)):
        event_i_jet_pt = total_tar_pt[i] 
        event_i_jet_eta = total_tar_eta[i] 
        event_i_jet_phi = total_tar_phi[i] 
        # print(len(event_i_jet_pt),len(event_i_jet_pt[event_i_jet_pt>20]))
        # filter, only count truth jets above 20 GeV
        event_i_jet_eta = event_i_jet_eta[event_i_jet_pt>20]
        event_i_jet_phi = event_i_jet_phi[event_i_jet_pt>20]
        event_i_jet_pt = event_i_jet_pt[event_i_jet_pt>20]

        if len(event_i_jet_pt)>3:
            ind = np.argpartition(event_i_jet_pt,-4)[-4:]
            top4_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top4_idx[3]]
            jet0_eta = event_i_jet_eta[top4_idx[3]]
            jet0_phi = event_i_jet_phi[top4_idx[3]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top4_idx[2]]
            jet1_eta = event_i_jet_eta[top4_idx[2]]
            jet1_phi = event_i_jet_phi[top4_idx[2]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top4_idx[1]]
            jet2_eta = event_i_jet_eta[top4_idx[1]]
            jet2_phi = event_i_jet_phi[top4_idx[1]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            jet3_pt = event_i_jet_pt[top4_idx[0]]
            jet3_eta = event_i_jet_eta[top4_idx[0]]
            jet3_phi = event_i_jet_phi[top4_idx[0]]
            jet3_ = np.array((jet3_eta,jet3_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR03 = np.linalg.norm(jet0_-jet3_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            dR13 = np.linalg.norm(jet1_-jet3_)
            dR23 = np.linalg.norm(jet2_-jet3_)
            # print(jet0_pt,jet1_pt,jet2_pt,jet3_pt)
            # print(dR01,dR02,dR12)
            if (dR01 > 0.6) and (dR02 > 0.6) and (dR03 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>2:
            ind = np.argpartition(event_i_jet_pt,-3)[-3:]
            top3_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top3_idx[2]]
            jet0_eta = event_i_jet_eta[top3_idx[2]]
            jet0_phi = event_i_jet_phi[top3_idx[2]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top3_idx[1]]
            jet1_eta = event_i_jet_eta[top3_idx[1]]
            jet1_phi = event_i_jet_phi[top3_idx[1]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top3_idx[0]]
            jet2_eta = event_i_jet_eta[top3_idx[0]]
            jet2_phi = event_i_jet_phi[top3_idx[0]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            if (dR01 > 0.6) and (dR02 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>1:
            ind = np.argpartition(event_i_jet_pt,-2)[-2:]
            top2_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top2_idx[1]]
            jet0_eta = event_i_jet_eta[top2_idx[1]]
            jet0_phi = event_i_jet_phi[top2_idx[1]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top2_idx[0]]
            jet1_eta = event_i_jet_eta[top2_idx[0]]
            jet1_phi = event_i_jet_phi[top2_idx[0]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            if (dR01 > 0.6):
                keep_iso[i] = 1
        
        else:
            keep_iso[i] = 1

    print('--->',sum(keep_iso),len(keep_iso))
    print("now we can get rid of events where the leading akt jet is close to another top 4 jet (where possible)")
    iso_total_tar_pt = []
    iso_total_tru_pt = []
    iso_total_p_pt = []
    for j in range(len(keep_iso)):
        should_keep = keep_iso[j]
        if should_keep:
            iso_total_tru_pt.append(total_tru_pt[j])
            iso_total_tar_pt.append(total_tar_pt[j])
            iso_total_p_pt.append(total_p_pt[j])

    # Make a trigger decision based on leading antikt jet pt
    tar_lead_pt = np.array([leading_jet_pt(x) for x in iso_total_tar_pt])
    p_lead_pt = np.array([leading_jet_pt(x) for x in iso_total_p_pt])

    # Set the x-axis and binning
    start,end = 20,450
    step = 10
    bins = np.arange(start, end, step)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    
    # trigger decision 1
    lead_jet_pt_cut1 = 150 # GeV
    trig_decision_akt1 = np.argwhere(tar_lead_pt>lead_jet_pt_cut1).T[0]
    trig_decision_pred1 = np.argwhere(p_lead_pt>lead_jet_pt_cut1).T[0]
    h_akt1, bedges = np.histogram(tar_lead_pt,bins=bins)
    h2_akt1, bedges = np.histogram(tar_lead_pt[trig_decision_akt1],bins=bins)
    h2_pre1, bedges = np.histogram(tar_lead_pt[trig_decision_pred1],bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_akt_eff = get_ratio(h2_akt1,h_akt1)
        h_akt_err = get_errorbars(h2_akt1,h_akt1)
        h_pre_eff1 = get_ratio(h2_pre1,h_akt1)
        h_pre_err1 = get_errorbars(h2_pre1,h_akt1)

    # trigger decision 2
    lead_jet_pt_cut2 = 225 # GeV
    trig_decision_akt2  = np.argwhere(tar_lead_pt>lead_jet_pt_cut2).T[0]
    trig_decision_pred2 = np.argwhere(p_lead_pt>lead_jet_pt_cut2).T[0]
    h_akt2, bedges = np.histogram(tar_lead_pt,bins=bins)
    h2_akt2, bedges = np.histogram(tar_lead_pt[trig_decision_akt2],bins=bins)
    h2_pre2, bedges = np.histogram(tar_lead_pt[trig_decision_pred2],bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_pre_eff2 = get_ratio(h2_pre2,h_akt2)
        h_pre_err2 = get_errorbars(h2_pre2,h_akt2)

    # trigger decision 3
    lead_jet_pt_cut3 = 300 # GeV
    trig_decision_akt3  = np.argwhere(tar_lead_pt>lead_jet_pt_cut3).T[0]
    trig_decision_pred3 = np.argwhere(p_lead_pt>lead_jet_pt_cut3).T[0]
    h_akt3, bedges = np.histogram(tar_lead_pt,bins=bins)
    h2_akt2, bedges = np.histogram(tar_lead_pt[trig_decision_akt3],bins=bins)
    h2_pre3, bedges = np.histogram(tar_lead_pt[trig_decision_pred3],bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_pre_eff3 = get_ratio(h2_pre3,h_akt3)
        h_pre_err3 = get_errorbars(h2_pre3,h_akt3)


    f,a = plt.subplots(1,1,figsize=(8,8))
    a.vlines(x=[lead_jet_pt_cut1], ymin=[-0.2], ymax=[1.02], colors='lightsalmon', ls='--',alpha=0.7)
    a.errorbar(bin_centers,h_pre_eff1,xerr=bin_width/2,yerr=h_pre_err1,elinewidth=0.4,marker='.',ls='none',label=f'{lead_jet_pt_cut1} GeV',color='lightsalmon')
    a.vlines(x=[lead_jet_pt_cut2], ymin=[-0.2], ymax=[1.02], colors='red', ls='--',alpha=0.7)
    a.errorbar(bin_centers,h_pre_eff2,xerr=bin_width/2,yerr=h_pre_err2,elinewidth=0.4,marker='.',ls='none',label=f'{lead_jet_pt_cut2} GeV',color='red')
    a.vlines(x=[lead_jet_pt_cut3], ymin=[-0.2], ymax=[1.02], colors='maroon', ls='--',alpha=0.7)
    a.errorbar(bin_centers,h_pre_eff3,xerr=bin_width/2,yerr=h_pre_err3,elinewidth=0.4,marker='.',ls='none',label=f'{lead_jet_pt_cut3} GeV',color='maroon')
    a.set(xlabel="Leading AKT4EMTopo jet " + r"$p_T$ [GeV]",ylabel='Preselection Efficiency',xlim=(start,end),ylim=(-0.2,1.2))
    a.legend(bbox_to_anchor=(0.7,0.025), loc="lower left")
    plt.text(start+15,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    # plt.text(start+80,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+80,1.12, "Simulation Preliminary",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+15,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    # plt.text(start+15,1.03, f"MC {proc[:2]} " +  r"$p^{uncalib}_{T} > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    plt.text(start+15,1.03, f"MC dijet " +  r"$p^{uncalib}_{T} > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    plt.text(end-200,-0.175, f"AKT4EMTopo jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/aktiso_leading_multi_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")







if subleading_iso_jet_multi_cut:
    total_tar_pt    = load_object(metrics_folder+"/tarboxes_pt.pkl")
    total_tar_eta   = load_object(metrics_folder+"/tarboxes_eta.pkl")
    total_tar_phi      = load_object(metrics_folder+"/tarboxes_phi.pkl")
    total_tru_pt    = load_object(metrics_folder+"/truboxes_pt.pkl")
    total_tru_eta   = load_object(metrics_folder+"/truboxes_eta.pkl")
    total_p_pt      = load_object(metrics_folder+"/pboxes_pt.pkl")
    total_p_eta     = load_object(metrics_folder+"/pboxes_eta.pkl")
    print("We get rid of some events")
    keep_iso = np.zeros(len(total_tar_pt))
    for i in range(len(total_tar_pt)):
        event_i_jet_pt = total_tar_pt[i] 
        event_i_jet_eta = total_tar_eta[i] 
        event_i_jet_phi = total_tar_phi[i] 
        # filter, only count truth jets above 20 GeV
        event_i_jet_eta = event_i_jet_eta[event_i_jet_pt>20]
        event_i_jet_phi = event_i_jet_phi[event_i_jet_pt>20]
        event_i_jet_pt = event_i_jet_pt[event_i_jet_pt>20]

        if len(event_i_jet_pt)>3:
            ind = np.argpartition(event_i_jet_pt,-4)[-4:]
            top4_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top4_idx[3]]
            jet0_eta = event_i_jet_eta[top4_idx[3]]
            jet0_phi = event_i_jet_phi[top4_idx[3]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top4_idx[2]]
            jet1_eta = event_i_jet_eta[top4_idx[2]]
            jet1_phi = event_i_jet_phi[top4_idx[2]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top4_idx[1]]
            jet2_eta = event_i_jet_eta[top4_idx[1]]
            jet2_phi = event_i_jet_phi[top4_idx[1]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            jet3_pt = event_i_jet_pt[top4_idx[0]]
            jet3_eta = event_i_jet_eta[top4_idx[0]]
            jet3_phi = event_i_jet_phi[top4_idx[0]]
            jet3_ = np.array((jet3_eta,jet3_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR03 = np.linalg.norm(jet0_-jet3_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            dR13 = np.linalg.norm(jet1_-jet3_)
            dR23 = np.linalg.norm(jet2_-jet3_)
            # print(jet0_pt,jet1_pt,jet2_pt,jet3_pt)
            # print(dR01,dR02,dR12)
            if (dR01 > 0.6) and (dR02 > 0.6) and (dR03 > 0.6) and (dR12 > 0.6) and (dR13 > 0.6) and (dR23 > 0.6):
            # if (dR01 > 0.6) and (dR02 > 0.6) and (dR12 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>2:
            ind = np.argpartition(event_i_jet_pt,-3)[-3:]
            top3_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top3_idx[2]]
            jet0_eta = event_i_jet_eta[top3_idx[2]]
            jet0_phi = event_i_jet_phi[top3_idx[2]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top3_idx[1]]
            jet1_eta = event_i_jet_eta[top3_idx[1]]
            jet1_phi = event_i_jet_phi[top3_idx[1]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            jet2_pt = event_i_jet_pt[top3_idx[0]]
            jet2_eta = event_i_jet_eta[top3_idx[0]]
            jet2_phi = event_i_jet_phi[top3_idx[0]]
            jet2_ = np.array((jet2_eta,jet2_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            dR02 = np.linalg.norm(jet0_-jet2_)
            dR12 = np.linalg.norm(jet1_-jet2_)
            if (dR01 > 0.6) and (dR02 > 0.6) and (dR12 > 0.6):
                keep_iso[i] = 1

        elif len(event_i_jet_pt)>1:
            ind = np.argpartition(event_i_jet_pt,-2)[-2:]
            top2_idx = ind[np.argsort(event_i_jet_pt[ind])]

            jet0_pt = event_i_jet_pt[top2_idx[1]]
            jet0_eta = event_i_jet_eta[top2_idx[1]]
            jet0_phi = event_i_jet_phi[top2_idx[1]]
            jet0_ = np.array((jet0_eta,jet0_phi))

            jet1_pt = event_i_jet_pt[top2_idx[0]]
            jet1_eta = event_i_jet_eta[top2_idx[0]]
            jet1_phi = event_i_jet_phi[top2_idx[0]]
            jet1_ = np.array((jet1_eta,jet1_phi))

            # np.sqrt((jet0_eta-jet1_eta)**2 + (jet0_phi - jet1_phi)**2)
            dR01 = np.linalg.norm(jet0_-jet1_)
            if (dR01 > 0.6):
                keep_iso[i] = 1
        
        else:
            keep_iso[i] = 1

    print('-->',sum(keep_iso),len(keep_iso))
    print("now we can get rid of events where the leading truth jet is close to another top 4 jet (where possible)")
    iso_total_tar_pt = []
    iso_total_tru_pt = []
    iso_total_p_pt = []
    for j in range(len(keep_iso)):
        should_keep = keep_iso[j]
        if should_keep:
            iso_total_tru_pt.append(total_tru_pt[j])
            iso_total_tar_pt.append(total_tar_pt[j])
            iso_total_p_pt.append(total_p_pt[j])


    start,end = 20,140
    step = 2.5
    bins = np.arange(start, end, step)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    nth_jet = 3
    tru_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_tru_pt])
    tar_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_tar_pt])
    p_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in iso_total_p_pt])

    # trigger decision 1
    nth_lead_jet_pt_cut1 = 40 # GeV
    trig_decision_akt1  = np.argwhere(tar_nlead_pt>nth_lead_jet_pt_cut1).T[0]
    trig_decision_pred1 = np.argwhere(p_nlead_pt>nth_lead_jet_pt_cut1).T[0]
    h_akt1, bedges = np.histogram(tar_nlead_pt,bins=bins)
    h2_akt1, bedges = np.histogram(tar_nlead_pt[trig_decision_akt1],bins=bins)
    h2_pre1, bedges = np.histogram(tar_nlead_pt[trig_decision_pred1],bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_akt_eff = get_ratio(h2_akt1,h_akt1)
        h_akt_err = get_errorbars(h2_akt1,h_akt1)
        h_pre_eff1 = get_ratio(h2_pre1,h_akt1)
        h_pre_err1 = get_errorbars(h2_pre1,h_akt1)

    # trigger decision 2
    nth_lead_jet_pt_cut2 = 60 # GeV
    trig_decision_akt2  = np.argwhere(tar_nlead_pt>nth_lead_jet_pt_cut2).T[0]
    trig_decision_pred2 = np.argwhere(p_nlead_pt>nth_lead_jet_pt_cut2).T[0]
    h_akt2, bedges = np.histogram(tar_nlead_pt,bins=bins)
    h2_akt2, bedges = np.histogram(tar_nlead_pt[trig_decision_akt2],bins=bins)
    h2_pre2, bedges = np.histogram(tar_nlead_pt[trig_decision_pred2],bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_pre_eff2 = get_ratio(h2_pre2,h_akt2)
        h_pre_err2 = get_errorbars(h2_pre2,h_akt2)

    # trigger decision 2
    nth_lead_jet_pt_cut3 = 80 # GeV
    trig_decision_akt3  = np.argwhere(tar_nlead_pt>nth_lead_jet_pt_cut3).T[0]
    trig_decision_pred3 = np.argwhere(p_nlead_pt>nth_lead_jet_pt_cut3).T[0]
    h_akt3, bedges = np.histogram(tar_nlead_pt,bins=bins)
    h2_akt2, bedges = np.histogram(tar_nlead_pt[trig_decision_akt3],bins=bins)
    h2_pre3, bedges = np.histogram(tar_nlead_pt[trig_decision_pred3],bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_pre_eff3 = get_ratio(h2_pre3,h_akt3)
        h_pre_err3 = get_errorbars(h2_pre3,h_akt3)

    f,a = plt.subplots(1,1,figsize=(8,8))
    a.vlines(x=[nth_lead_jet_pt_cut1], ymin=[-0.2], ymax=[1.02], colors='lightsalmon', ls='--',alpha=0.9)
    a.errorbar(bin_centers,h_pre_eff1,xerr=bin_width/2,yerr=h_pre_err1,elinewidth=0.4,marker='.',ls='none',label=f'{nth_lead_jet_pt_cut1} GeV',color='lightsalmon')
    a.vlines(x=[nth_lead_jet_pt_cut2], ymin=[-0.2], ymax=[1.02], colors='red', ls='--',alpha=0.9)
    a.errorbar(bin_centers,h_pre_eff2,xerr=bin_width/2,yerr=h_pre_err2,elinewidth=0.4,marker='.',ls='none',label=f'{nth_lead_jet_pt_cut2} GeV',color='red')
    a.vlines(x=[nth_lead_jet_pt_cut3], ymin=[-0.2], ymax=[1.02], colors='maroon', ls='--',alpha=0.9)
    a.errorbar(bin_centers,h_pre_eff3,xerr=bin_width/2,yerr=h_pre_err3,elinewidth=0.4,marker='.',ls='none',label=f'{nth_lead_jet_pt_cut3} GeV',color='maroon')
    a.set(xlabel=f"${{{nth_jet}}}^{{rd}}$ leading AKT4EMTopo jet " + r" $p_T$ [GeV]",ylabel='Preselection Efficiency',xlim=(start-10,end),ylim=(-0.2,1.2))
    a.legend(loc='lower right')
    plt.text(start-3,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    # plt.text(start+18,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start+18,1.12, "Simulation Preliminary",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(start-3,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(start-3,1.03, f"MC {proc[:2]} " +  r"$p^{uncalib}_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
    plt.text(end-58,-0.175, f"AKT4EMTopo jet dR > 0.6 isolated ",fontfamily='sans-serif',fontweight='bold',fontsize=12)
    f.subplots_adjust(hspace=0.4)
    f.savefig(save_folder + f'/aktiso_{nth_jet}leading_multi_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")




