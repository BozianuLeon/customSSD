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
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/{proc}/{date}/matching/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

image_format = "png"



print("=======================================================================================================")
print(f"Loading jets from\n{metrics_folder}")
print("=======================================================================================================\n")

# IOU matched
match_t_pt      = np.concatenate(load_object(metrics_folder+"/tboxes_matched_pt.pkl"))
match_p_pt      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_pt.pkl"))
match_t_eta     = np.concatenate(load_object(metrics_folder+"/tboxes_matched_eta.pkl"))
match_p_eta     = np.concatenate(load_object(metrics_folder+"/pboxes_matched_eta.pkl"))
match_p_scr     = np.concatenate(load_object(metrics_folder+"/pboxes_matched_scr.pkl"))
unmatch_t_pt    = np.concatenate(load_object(metrics_folder+"/tboxes_unmatched_pt.pkl"))
unmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_pt.pkl"))
unmatch_t_eta   = np.concatenate(load_object(metrics_folder+"/tboxes_unmatched_eta.pkl"))
unmatch_p_eta   = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_eta.pkl"))
unmatch_p_scr   = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_scr.pkl"))
# dR matched
dRmatch_t_pt    = np.concatenate(load_object(metrics_folder+"/tboxes_dRmatched_pt.pkl"))
dRmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_pt.pkl"))
dRmatch_t_eta   = np.concatenate(load_object(metrics_folder+"/tboxes_dRmatched_eta.pkl"))
dRmatch_p_eta   = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_eta.pkl"))
dRmatch_p_scr   = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_scr.pkl"))
dRunmatch_t_pt  = np.concatenate(load_object(metrics_folder+"/tboxes_dRunmatched_pt.pkl"))
dRunmatch_p_pt  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_pt.pkl"))
dRunmatch_t_eta = np.concatenate(load_object(metrics_folder+"/tboxes_dRunmatched_eta.pkl"))
dRunmatch_p_eta = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_eta.pkl"))
dRunmatch_p_scr = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_scr.pkl"))

print("Implementing a temporary, post-hoc confidence threshold:")
print(f"Number      matched targets: {len(match_t_pt)}\nNumber      matched predictions: {len(match_p_pt)}")
print(f"Number    unmatched targets: {len(unmatch_t_pt)}\nNumber    unmatched predictions: {len(unmatch_p_pt)}")
print(f"Number dR   matched targets: {len(dRmatch_t_pt)}\nNumber dR   matched predictions: {len(dRmatch_p_pt)}")
print(f"Number dR unmatched targets: {len(dRunmatch_t_pt)}\nNumber dR unmatched predictions: {len(dRunmatch_p_pt)}")

scr_threshold = 0.5

match_scr_mask = match_p_scr > scr_threshold
match_t_pt = match_t_pt[match_scr_mask]
match_p_pt = match_p_pt[match_scr_mask]
match_t_eta = match_t_eta[match_scr_mask]
match_p_eta = match_p_eta[match_scr_mask]

dRmatch_scr_mask = dRmatch_p_scr > scr_threshold
dRmatch_t_pt = dRmatch_t_pt[dRmatch_scr_mask]
dRmatch_p_pt = dRmatch_p_pt[dRmatch_scr_mask]
dRmatch_t_eta = dRmatch_t_eta[dRmatch_scr_mask]
dRmatch_p_eta = dRmatch_p_eta[dRmatch_scr_mask]

unmatch_scr_mask = unmatch_p_scr > scr_threshold
unmatch_p_pt = unmatch_p_pt[unmatch_scr_mask]
unmatch_p_eta = unmatch_p_eta[unmatch_scr_mask]

dRunmatch_scr_mask = dRunmatch_p_scr > scr_threshold
dRunmatch_p_pt = dRunmatch_p_pt[dRunmatch_scr_mask]
dRunmatch_p_eta = dRunmatch_p_eta[dRunmatch_scr_mask]




print(f"In the unseen test data we have {len(load_object(metrics_folder+'/tboxes_matched_pt.pkl'))}  {len(load_object(metrics_folder+'/tboxes_dRmatched_pt.pkl'))} events,")
print(f"with a total {len(unmatch_t_pt)+len(match_t_pt)} AntiKt4EMTopo Jets and {len(unmatch_p_pt)+len(match_p_pt)}  jets predicted by the CNN")
print(f"Of the {len(unmatch_p_pt)+len(match_p_pt)} predicted boxes {len(match_p_pt)} are matched, (therefore {len(unmatch_p_pt)} are unmatched)")
print(f"Of the {len(unmatch_t_pt)+len(match_t_pt)} target boxes {len(match_t_pt)} are matched, (therefore {len(unmatch_t_pt)} are unmatched)\n\n")


pt_thresholds = [0,10,15,20,25,30,35,40,60,100,200,250,450,650]
cumulative_t_match_frac = []
for threshold in pt_thresholds:
    matched_above_thresh   = match_t_pt[match_t_pt > threshold]
    unmatched_above_thresh = unmatch_t_pt[unmatch_t_pt > threshold]
    print(f'For jets above {threshold} GeV: {len(matched_above_thresh) / (len(matched_above_thresh)+len(unmatched_above_thresh)):.4f} of target jets are matched')
    print(f'Note that there are {len(matched_above_thresh)+len(unmatched_above_thresh)} target jets above {threshold} GeV\n')
    cumulative_t_match_frac.append(len(matched_above_thresh) / (len(matched_above_thresh)+len(unmatched_above_thresh)))
    if threshold==40:
        print(f'\t\t\tFOR ALL JETS ABOVE {threshold} WE MATCH {len(matched_above_thresh) / (len(matched_above_thresh)+len(unmatched_above_thresh)):.4f} targets! NOTICE!')
print()
print()
cumulative_p_match_frac = []
for threshold in pt_thresholds:
    matched_above_thresh   = match_p_pt[match_p_pt > threshold]
    unmatched_above_thresh = unmatch_p_pt[unmatch_p_pt > threshold]
    print(f'For jets above {threshold} GeV: {len(matched_above_thresh) / (len(matched_above_thresh)+len(unmatched_above_thresh)):.4f} of predicted jets are matched')
    print(f'Note that there are {len(matched_above_thresh)+len(unmatched_above_thresh)} predicted jets above {threshold} GeV\n')
    cumulative_p_match_frac.append(len(matched_above_thresh) / (len(matched_above_thresh)+len(unmatched_above_thresh)))
    if threshold==40:
        print(f'\t\t\tFOR ALL JETS ABOVE {threshold} WE MATCH {len(matched_above_thresh) / (len(matched_above_thresh)+len(unmatched_above_thresh)):.4f} predictions! NOTICE!')


print()
print("Lists of match fraction ALL jets above thresholds")
print("Thresholds: \t",pt_thresholds)
print("Targets:    \t",cumulative_t_match_frac)
print("Predictions:\t",cumulative_p_match_frac)
print()




print("=======================================================================================================")
print(f"Plotting (un)matched fraction, saving to {save_folder}")
print("=======================================================================================================\n")



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





# bin_edges = [0, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, max(total_pred_pt)]
# bin_edges = [20, 30, 40, 50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200, 225, 250]
bin_edges = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 850]
bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
bin_width = np.diff(bin_edges)

print("=======================================================================================================")
print("Calculating fraction of targets matched, and predictions unmatched for IOU method!")
frac_p_match,frac_p_unmatch,num_p = [],[],[]
frac_t_match,frac_t_unmatch,num_t = [],[],[]
for bin_idx in range(len(bin_edges)-1):
    # First, get PREDICTED jets in this pT bin
    match_p_pt_mask   = (bin_edges[bin_idx] < match_p_pt) & (match_p_pt < bin_edges[bin_idx+1])
    unmatch_p_pt_mask = (bin_edges[bin_idx] < unmatch_p_pt) & (unmatch_p_pt < bin_edges[bin_idx+1])

    matched_p_bin_i   = match_p_pt[match_p_pt_mask]
    unmatched_p_bin_i = unmatch_p_pt[unmatch_p_pt_mask]

    # Next, get TARGET jets in this pT bin
    match_t_pt_mask   = (bin_edges[bin_idx] < match_t_pt) & (match_t_pt < bin_edges[bin_idx+1])
    unmatch_t_pt_mask = (bin_edges[bin_idx] < unmatch_t_pt) & (unmatch_t_pt < bin_edges[bin_idx+1])

    matched_t_bin_i   = match_t_pt[match_t_pt_mask]
    unmatched_t_bin_i = unmatch_t_pt[unmatch_t_pt_mask]
    print(f"There are {len(matched_p_bin_i)+len(unmatched_p_bin_i)} predicted jets and {len(matched_t_bin_i)+len(unmatched_t_bin_i)} target jets in bin {bin_idx} ([{bin_edges[bin_idx]}, {bin_edges[bin_idx+1]}])")

    # Finally, get the fraction (un)matched in each bin
    print(f"Fraction of predictions that are matched: {len(matched_p_bin_i)/(len(matched_p_bin_i)+len(unmatched_p_bin_i)):.3f}\t unmatched: {len(unmatched_p_bin_i)/(len(matched_p_bin_i)+len(unmatched_p_bin_i)):.3f}")
    print(f"Fraction of targets that are matched: {len(matched_t_bin_i)/(len(matched_t_bin_i)+len(unmatched_t_bin_i)):.3f}\t unmatched: {len(unmatched_t_bin_i)/(len(matched_t_bin_i)+len(unmatched_t_bin_i)):.3f}\n")

    frac_p_match.append(len(matched_p_bin_i)/(len(matched_p_bin_i)+len(unmatched_p_bin_i)))
    frac_t_match.append(len(matched_t_bin_i)/(len(matched_t_bin_i)+len(unmatched_t_bin_i)))
    frac_p_unmatch.append(len(unmatched_p_bin_i)/(len(matched_p_bin_i)+len(unmatched_p_bin_i)))
    frac_t_unmatch.append(len(unmatched_t_bin_i)/(len(matched_t_bin_i)+len(unmatched_t_bin_i)))
    num_p.append(len(matched_p_bin_i)+len(unmatched_p_bin_i))
    num_t.append(len(matched_t_bin_i)+len(unmatched_t_bin_i))


# calculate error bars by giving the absolute number of (un)matched and the total number in each bin
match_t_error = get_errorbars(np.array(frac_t_match)*np.array(num_t), np.array(num_t))
match_p_error = get_errorbars(np.array(frac_p_match)*np.array(num_p), np.array(num_p))
unmatch_p_error = get_errorbars(np.array(frac_p_unmatch)*np.array(num_p), np.array(num_p))

f,ax = plt.subplots(1,1,figsize=(8, 6))
# ax.plot(bin_centers,percentage_matched_in_pt,marker='x',color='cyan',label=f'% Matched Truth Boxes (Accuracy)')
# ax.plot(bin_centers,percentage_unmatched_in_pt,marker='+',color='coral',label=f'% Unmatched Prediction Boxes (Fake rate)')
ax.errorbar(bin_centers,frac_t_match,xerr=bin_width/2,yerr=match_t_error,color='cyan',marker='.',ms=5.5,elinewidth=1.2,ls='none',label=f'% Matched Truth Boxes (Accuracy)')
ax.errorbar(bin_centers,frac_p_unmatch,xerr=bin_width/2,yerr=unmatch_p_error,color='coral',marker='x',ms=5.5,elinewidth=1.2,ls='none',label=f'% Unmatched Prediction Boxes (Fake rate)')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum (GeV)',ylabel=f'Fraction of boxes')
ax.set_ylim((-0.2,1.2))
ax.legend(loc='lower left',bbox_to_anchor=(0.005, 0.005),fontsize="x-small")
hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
f.savefig(save_folder + f'/match_frac_pT.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


print()
print()


print("=======================================================================================================")
print("Calculating fraction of targets matched, and predictions unmatched for deltaR method!")
frac_p_dRmatch,frac_p_dRunmatch,num_dR_p = [],[],[]
frac_t_dRmatch,frac_t_dRunmatch,num_dR_t = [],[],[]
for bin_idx in range(len(bin_edges)-1):
    # First, get PREDICTED jets in this pT bin
    dRmatch_p_pt_mask   = (bin_edges[bin_idx] < dRmatch_p_pt) & (dRmatch_p_pt < bin_edges[bin_idx+1])
    dRunmatch_p_pt_mask = (bin_edges[bin_idx] < dRunmatch_p_pt) & (dRunmatch_p_pt < bin_edges[bin_idx+1])

    dRmatched_p_bin_i   = dRmatch_p_pt[dRmatch_p_pt_mask]
    dRunmatched_p_bin_i = dRunmatch_p_pt[dRunmatch_p_pt_mask]

    # Next, get TARGET jets in this pT bin
    dRmatch_t_pt_mask   = (bin_edges[bin_idx] < dRmatch_t_pt) & (dRmatch_t_pt < bin_edges[bin_idx+1])
    dRunmatch_t_pt_mask = (bin_edges[bin_idx] < dRunmatch_t_pt) & (dRunmatch_t_pt < bin_edges[bin_idx+1])

    dRmatched_t_bin_i   = dRmatch_t_pt[dRmatch_t_pt_mask]
    dRunmatched_t_bin_i = dRunmatch_t_pt[dRunmatch_t_pt_mask]
    print(f"There are {len(dRmatched_p_bin_i)+len(dRunmatched_p_bin_i)} predicted jets and {len(dRmatched_t_bin_i)+len(dRunmatched_t_bin_i)} target jets in bin {bin_idx} ([{bin_edges[bin_idx]}, {bin_edges[bin_idx+1]}])")

    # Finally, get the fraction (un)matched in each bin
    print(f"Fraction of predictions that are matched: {len(dRmatched_p_bin_i)/(len(dRmatched_p_bin_i)+len(dRunmatched_p_bin_i)):.3f}\t unmatched: {len(dRunmatched_p_bin_i)/(len(dRmatched_p_bin_i)+len(dRunmatched_p_bin_i)):.3f}")
    print(f"Fraction of targets that are matched: {len(dRmatched_t_bin_i)/(len(dRmatched_t_bin_i)+len(dRunmatched_t_bin_i)):.3f}\t unmatched: {len(dRunmatched_t_bin_i)/(len(dRmatched_t_bin_i)+len(dRunmatched_t_bin_i)):.3f}\n")

    frac_p_dRmatch.append(len(dRmatched_p_bin_i)/(len(dRmatched_p_bin_i)+len(dRunmatched_p_bin_i)))
    frac_t_dRmatch.append(len(dRmatched_t_bin_i)/(len(dRmatched_t_bin_i)+len(dRunmatched_t_bin_i)))
    frac_p_dRunmatch.append(len(dRunmatched_p_bin_i)/(len(dRmatched_p_bin_i)+len(dRunmatched_p_bin_i)))
    frac_t_dRunmatch.append(len(dRunmatched_t_bin_i)/(len(dRmatched_t_bin_i)+len(dRunmatched_t_bin_i)))
    num_dR_p.append(len(dRmatched_p_bin_i)+len(dRunmatched_p_bin_i))
    num_dR_t.append(len(dRmatched_t_bin_i)+len(dRunmatched_t_bin_i))


# calculate error bars by giving the absolute number of (un)matched and the total number in each bin
dRmatch_t_error = get_errorbars(np.array(frac_t_dRmatch)*np.array(num_dR_t), np.array(num_dR_t))
dRmatch_p_error = get_errorbars(np.array(frac_p_dRmatch)*np.array(num_dR_p), np.array(num_dR_p))
dRunmatch_p_error = get_errorbars(np.array(frac_p_dRunmatch)*np.array(num_dR_p), np.array(num_dR_p))

f,ax = plt.subplots(1,1,figsize=(8, 6))
ax.errorbar(bin_centers,frac_t_dRmatch,xerr=bin_width/2,yerr=dRmatch_t_error,color='cyan',marker='.',ms=5.5,elinewidth=1.2,ls='none',label=f'% Matched Truth Boxes (Accuracy)')
ax.errorbar(bin_centers,frac_p_dRunmatch,xerr=bin_width/2,yerr=dRunmatch_p_error,color='coral',marker='x',ms=5.5,elinewidth=1.2,ls='none',label=f'% Unmatched Prediction Boxes (Fake rate)')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum (GeV)',ylabel=f'Fraction of boxes')
ax.set_ylim((-0.2,1.2))
ax.legend(loc='lower left',bbox_to_anchor=(0.005, 0.005),fontsize="x-small")
hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
f.savefig(save_folder + f'/dRmatch_frac_pT.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


