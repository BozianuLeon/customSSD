import numpy as np 
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


model_name = "jetSSD_di_uconvnext_central_11e"
proc       = "JZ4"
date       = "20250211-13"

metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/{proc}/{date}/box_metrics"
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/{proc}/{date}/trig/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

image_format = "png"

print("=======================================================================================================")
print(f"Loading jets from\n{metrics_folder}")
print("=======================================================================================================\n")


total_t_pt      = load_object(metrics_folder+"/tboxes_pt.pkl")
total_p_pt      = load_object(metrics_folder+"/pboxes_pt.pkl")
total_p_scr     = load_object(metrics_folder+"/pboxes_scores.pkl")



print("=======================================================================================================")
print(f"Making leading jet trigger decision")
print("=======================================================================================================\n")



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




# Set the x-axis and binning
start,end = 350,750
step = 10
bins = np.arange(start, end, step)

# Make a trigger decision based on leading antikt jet pt
t_lead_pt = np.array([leading_jet_pt(x) for x in total_t_pt])
p_lead_pt = np.array([leading_jet_pt(x) for x in total_p_pt])

lead_jet_pt_cut = 225 # GeV
trig_decision_akt = np.argwhere(t_lead_pt>lead_jet_pt_cut).T[0]
trig_decision_pred = np.argwhere(p_lead_pt>lead_jet_pt_cut).T[0]



f,ax = plt.subplots(3,1,figsize=(8,14))
n_akt,bins,_ = ax[0].hist(t_lead_pt,bins=bins,histtype='step',label='AntiKt4EMTopo jetConstitScale')
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]
ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
ax[0].set(xlabel="Leading EMTopo (Offline) jet pT (GeV)",ylabel='Events')
ax[0].legend()

n2_akt,bins,_ = ax[1].hist(t_lead_pt[trig_decision_akt],bins=bins,histtype='step',label='Antikt jetConstitScale',color="gold")
n2_p,_,_ = ax[1].hist(t_lead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes',color="red")
ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
ax[1].set_title(f'After {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
ax[1].set(xlabel="Leading jet pT (GeV)",ylabel='Events')
ax[1].legend()

ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
with np.errstate(divide='ignore', invalid='ignore'):
    step_eff = get_ratio(n2_akt,n_akt)
    step_err = get_errorbars(n2_akt,n_akt)

    pred_eff = get_ratio(n2_p,n_akt)
    pred_err = get_errorbars(n2_p,n_akt)


ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='gold')
ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes.',color='red')
ax[2].set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency')
ax[2].legend(loc='lower right')
# hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
f.savefig(save_folder + f'/leading_{lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



f,a = plt.subplots(1,1,figsize=(8,8))
a.axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3)
a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes',color='red')
a.set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency')
a.legend(loc='upper left')
hep.atlas.label(ax=a,label='Work in Progress',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(save_folder + f'/leading{lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")





print("=======================================================================================================")
print(f"Making subleading jet trigger decision")
print("=======================================================================================================\n")

start,end = 20,350
step = 10
bins = np.arange(start, end, step)

nth_jet = 2
nth_lead_jet_pt_cut = 200 # 400GeV
t_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_t_pt])
p_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in total_p_pt])

trig_decision_akt  = np.argwhere(t_nlead_pt>nth_lead_jet_pt_cut).T[0]
trig_decision_pred = np.argwhere(p_nlead_pt>nth_lead_jet_pt_cut).T[0]



f,ax = plt.subplots(3,1,figsize=(6.5,12))
n_akt,bins,_ = ax[0].hist(t_nlead_pt,bins=bins,histtype='step',label='Anti-kt jetConstitScale')
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]
ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Events')
ax[0].legend()

n2_akt,bins,_ = ax[1].hist(t_nlead_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="gold")
n2_pb,_,_ = ax[1].hist(t_nlead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes Adj',color="red")
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

ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='gold')
ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes',color='red')
ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
ax[2].legend(loc='lower right')
hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
f.savefig(save_folder + f'/{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


f,a = plt.subplots(1,1,figsize=(8,8))
a.axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3)
a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes',color='red')
a.grid()
a.set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
a.legend(loc='upper left')
hep.atlas.label(ax=a,label='Work in Progress',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(save_folder + f'/{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_efficiency.{image_format}',dpi=400,format=image_format,bbox_inches="tight")




