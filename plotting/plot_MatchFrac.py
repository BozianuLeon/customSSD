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


model_name = "jetSSD_custom_convnext_central_32e"
# proc = "JZcomb0_test"
# date = "20250313-06"
# # date = "20250406-23"
proc = "ttbar_test"
date = "20250407-14"
# # date = "20250306-16"

metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/{proc}/{date}/box_metrics"
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/{proc}/{date}/matching/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

image_format = "png"



print("=======================================================================================================")
print(f"Loading jets from\n{metrics_folder}")
print("=======================================================================================================\n")
#total
total_tar_pt      = np.concatenate(load_object(metrics_folder+"/tarboxes_pt.pkl"))
total_tru_pt      = np.concatenate(load_object(metrics_folder+"/truboxes_pt.pkl"))
total_p_pt        = np.concatenate(load_object(metrics_folder+"/pboxes_pt.pkl"))
total_jet_weight  = np.concatenate(load_object(metrics_folder+"/jet_evt_weight.pkl"))
total_evt_weight  = load_object(metrics_folder+"/evt_weight.pkl")


# IOU matched
match_tar_pt      = np.concatenate(load_object(metrics_folder+"/tarboxes_matched_pt.pkl"))
match_p_pt      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_pt.pkl"))
match_tar_eta     = np.concatenate(load_object(metrics_folder+"/tarboxes_matched_eta.pkl"))
match_p_eta     = np.concatenate(load_object(metrics_folder+"/pboxes_matched_eta.pkl"))
match_p_scr     = np.concatenate(load_object(metrics_folder+"/pboxes_matched_scr.pkl"))
unmatch_tar_pt    = np.concatenate(load_object(metrics_folder+"/tarboxes_unmatched_pt.pkl"))
unmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_pt.pkl"))
unmatch_tar_eta   = np.concatenate(load_object(metrics_folder+"/tarboxes_unmatched_eta.pkl"))
unmatch_p_eta   = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_eta.pkl"))
unmatch_p_scr   = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_scr.pkl"))
# dR matched
dRmatch_tar_pt    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRmatched_pt.pkl"))
dRmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_pt.pkl"))
dRmatch_tar_eta   = np.concatenate(load_object(metrics_folder+"/tarboxes_dRmatched_eta.pkl"))
dRmatch_p_eta   = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_eta.pkl"))
dRmatch_p_scr   = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_scr.pkl"))
dRunmatch_tar_pt  = np.concatenate(load_object(metrics_folder+"/tarboxes_dRunmatched_pt.pkl"))
dRunmatch_p_pt  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_pt.pkl"))
dRunmatch_tar_eta = np.concatenate(load_object(metrics_folder+"/tarboxes_dRunmatched_eta.pkl"))
dRunmatch_p_eta = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_eta.pkl"))
dRunmatch_p_scr = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_scr.pkl"))
# dR truth matched
dRtruthmatch_tru_pt    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthmatched_pt.pkl"))
dRtruthmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_pt.pkl"))
dRtruthmatch_p_scr   = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_scr.pkl"))
dRtruthunmatch_tru_pt  = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthunmatched_pt.pkl"))
dRtruthunmatch_p_pt  = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthunmatched_pt.pkl"))
dRtruthunmatch_p_scr = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthunmatched_scr.pkl"))
# dR target matched to truth
dRtruthtarmatch_tru_pt    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthtarmatched_pt.pkl"))
dRtruthtarmatch_tar_pt    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRtruthtarmatched_pt.pkl"))
dRtruthtarunmatch_tru_pt    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthtarunmatched_pt.pkl"))
dRtruthtarunmatch_tar_pt    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRtruthtarunmatched_pt.pkl"))



print("Implementing a temporary, post-hoc confidence threshold:")
print(f"Number of events {len(total_evt_weight)}")
print(f"Number      matched targets: {len(match_tar_pt)}\nNumber      matched predictions: {len(match_p_pt)}")
print(f"Number    unmatched targets: {len(unmatch_tar_pt)}\nNumber    unmatched predictions: {len(unmatch_p_pt)}")
print(f"Number dR   matched targets: {len(dRmatch_tar_pt)}\nNumber dR   matched predictions: {len(dRmatch_p_pt)}")
print(f"Number dR unmatched targets: {len(dRunmatch_tar_pt)}\nNumber dR unmatched predictions: {len(dRunmatch_p_pt)}")

scr_threshold = 0.5

match_scr_mask = match_p_scr > scr_threshold
match_tar_pt = match_tar_pt[match_scr_mask]
match_p_pt = match_p_pt[match_scr_mask]
match_tar_eta = match_tar_eta[match_scr_mask]
match_p_eta = match_p_eta[match_scr_mask]

dRmatch_scr_mask = dRmatch_p_scr > scr_threshold
dRmatch_tar_pt = dRmatch_tar_pt[dRmatch_scr_mask]
dRmatch_p_pt = dRmatch_p_pt[dRmatch_scr_mask]
dRmatch_tar_eta = dRmatch_tar_eta[dRmatch_scr_mask]
dRmatch_p_eta = dRmatch_p_eta[dRmatch_scr_mask]

unmatch_scr_mask = unmatch_p_scr > scr_threshold
unmatch_p_pt = unmatch_p_pt[unmatch_scr_mask]
unmatch_p_eta = unmatch_p_eta[unmatch_scr_mask]

dRunmatch_scr_mask = dRunmatch_p_scr > scr_threshold
dRunmatch_p_pt = dRunmatch_p_pt[dRunmatch_scr_mask]
dRunmatch_p_eta = dRunmatch_p_eta[dRunmatch_scr_mask]

# truth matching
dRtruthmatch_scr_mask = dRtruthmatch_p_scr > scr_threshold
dRtruthmatch_tru_pt = dRtruthmatch_tru_pt[dRtruthmatch_scr_mask]
dRtruthmatch_p_pt = dRtruthmatch_p_pt[dRtruthmatch_scr_mask]
dRtruthunmatch_scr_mask = dRtruthunmatch_p_scr > scr_threshold
dRtruthunmatch_p_pt = dRtruthunmatch_p_pt[dRtruthunmatch_scr_mask]


print(f"In the unseen test data we have {len(load_object(metrics_folder+'/tarboxes_matched_pt.pkl'))}  {len(load_object(metrics_folder+'/tarboxes_dRmatched_pt.pkl'))} events,")
print(f"with a total {len(unmatch_tar_pt)+len(match_tar_pt)} AntiKt4EMTopo Jets and {len(unmatch_p_pt)+len(match_p_pt)}  jets predicted by the CNN")
print(f"Of the {len(unmatch_p_pt)+len(match_p_pt)} predicted boxes {len(match_p_pt)} are matched, (therefore {len(unmatch_p_pt)} are unmatched)")
print(f"Of the {len(unmatch_tar_pt)+len(match_tar_pt)} target boxes {len(match_tar_pt)} are matched, (therefore {len(unmatch_tar_pt)} are unmatched)\n\n")


pt_thresholds = [0,10,15,20,25,30,35,40,60,100,200,250,450,650]
cumulative_tar_match_frac = []
for threshold in pt_thresholds:
    matched_above_thresh   = match_tar_pt[match_tar_pt > threshold]
    unmatched_above_thresh = unmatch_tar_pt[unmatch_tar_pt > threshold]
    print(f'For jets above {threshold} GeV: {len(matched_above_thresh) / (len(matched_above_thresh)+len(unmatched_above_thresh)):.4f} of target jets are matched')
    print(f'Note that there are {len(matched_above_thresh)+len(unmatched_above_thresh)} target jets above {threshold} GeV\n')
    cumulative_tar_match_frac.append(len(matched_above_thresh) / (len(matched_above_thresh)+len(unmatched_above_thresh)))
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
print("Targets:    \t",cumulative_tar_match_frac)
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
frac_tar_match,frac_tar_unmatch,num_tar = [],[],[]
for bin_idx in range(len(bin_edges)-1):
    # First, get PREDICTED jets in this pT bin
    match_p_pt_mask   = (bin_edges[bin_idx] < match_p_pt) & (match_p_pt < bin_edges[bin_idx+1])
    unmatch_p_pt_mask = (bin_edges[bin_idx] < unmatch_p_pt) & (unmatch_p_pt < bin_edges[bin_idx+1])

    matched_p_bin_i   = match_p_pt[match_p_pt_mask]
    unmatched_p_bin_i = unmatch_p_pt[unmatch_p_pt_mask]

    # Next, get TARGET jets in this pT bin
    match_tar_pt_mask   = (bin_edges[bin_idx] < match_tar_pt) & (match_tar_pt < bin_edges[bin_idx+1])
    unmatch_tar_pt_mask = (bin_edges[bin_idx] < unmatch_tar_pt) & (unmatch_tar_pt < bin_edges[bin_idx+1])

    matched_tar_bin_i   = match_tar_pt[match_tar_pt_mask]
    unmatched_tar_bin_i = unmatch_tar_pt[unmatch_tar_pt_mask]
    print(f"There are {len(matched_p_bin_i)+len(unmatched_p_bin_i)} predicted jets and {len(matched_tar_bin_i)+len(unmatched_tar_bin_i)} target jets in bin {bin_idx} ([{bin_edges[bin_idx]}, {bin_edges[bin_idx+1]}])")

    # Finally, get the fraction (un)matched in each bin
    print(f"Fraction of predictions that are matched: {len(matched_p_bin_i)/(len(matched_p_bin_i)+len(unmatched_p_bin_i)):.3f}\t unmatched: {len(unmatched_p_bin_i)/(len(matched_p_bin_i)+len(unmatched_p_bin_i)):.3f}")
    print(f"Fraction of targets that are matched: {len(matched_tar_bin_i)/(len(matched_tar_bin_i)+len(unmatched_tar_bin_i)):.3f}\t unmatched: {len(unmatched_tar_bin_i)/(len(matched_tar_bin_i)+len(unmatched_tar_bin_i)):.3f}\n")

    frac_p_match.append(len(matched_p_bin_i)/(len(matched_p_bin_i)+len(unmatched_p_bin_i)))
    frac_tar_match.append(len(matched_tar_bin_i)/(len(matched_tar_bin_i)+len(unmatched_tar_bin_i)))
    frac_p_unmatch.append(len(unmatched_p_bin_i)/(len(matched_p_bin_i)+len(unmatched_p_bin_i)))
    frac_tar_unmatch.append(len(unmatched_tar_bin_i)/(len(matched_tar_bin_i)+len(unmatched_tar_bin_i)))
    num_p.append(len(matched_p_bin_i)+len(unmatched_p_bin_i))
    num_tar.append(len(matched_tar_bin_i)+len(unmatched_tar_bin_i))


# calculate error bars by giving the absolute number of (un)matched and the total number in each bin
match_tar_error = get_errorbars(np.array(frac_tar_match)*np.array(num_tar), np.array(num_tar))
match_p_error = get_errorbars(np.array(frac_p_match)*np.array(num_p), np.array(num_p))
unmatch_p_error = get_errorbars(np.array(frac_p_unmatch)*np.array(num_p), np.array(num_p))

f,ax = plt.subplots(1,1,figsize=(8, 6))
# ax.plot(bin_centers,percentage_matched_in_pt,marker='x',color='cyan',label=f'% Matched Truth Boxes (Accuracy)')
# ax.plot(bin_centers,percentage_unmatched_in_pt,marker='+',color='coral',label=f'% Unmatched Prediction Boxes (Fake rate)')
ax.errorbar(bin_centers,frac_tar_match,xerr=bin_width/2,yerr=match_tar_error,color='cyan',marker='.',ms=5.5,elinewidth=1.2,ls='-',label=f'% Matched Target AKT Jets (Accuracy)')
ax.errorbar(bin_centers,frac_p_unmatch,xerr=bin_width/2,yerr=unmatch_p_error,color='coral',marker='x',ms=5.5,elinewidth=1.2,ls='-',label=f'% Unmatched Prediction Boxes (Fake rate)')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum (GeV)',ylabel=f'Fraction of Jets')
ax.set_ylim((0.0,1.2))
ax.legend(loc='lower left',bbox_to_anchor=(0.45, 0.4),fontsize="x-small")
# hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
f.savefig(save_folder + f'/match_frac_pT.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

f,ax = plt.subplots(1,1,figsize=(8, 6))
# ax.plot(bin_centers,percentage_matched_in_pt,marker='x',color='cyan',label=f'% Matched Truth Boxes (Accuracy)')
# ax.plot(bin_centers,percentage_unmatched_in_pt,marker='+',color='coral',label=f'% Unmatched Prediction Boxes (Fake rate)')
ax.errorbar(bin_centers,frac_tar_match,xerr=bin_width/2,yerr=match_tar_error,color='cyan',marker='.',ms=5.5,elinewidth=1.2,ls='-',label=f'% Matched Target AKT Jets (Accuracy)')
ax.errorbar(bin_centers,frac_p_unmatch,xerr=bin_width/2,yerr=unmatch_p_error,color='coral',marker='x',ms=5.5,elinewidth=1.2,ls='-',label=f'% Unmatched Prediction Boxes (Fake rate)')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum (GeV)',ylabel=f'Fraction of boxes')
ax.set_ylim((0.0,1.2))
ax.legend(loc='lower left',bbox_to_anchor=(0.45, 0.5),fontsize="x-small")

ax2 = ax.twinx()
freq_tar, bins, _    = ax2.hist(total_tar_pt,bins=bin_edges,weights=total_jet_weight,histtype='stepfilled',color='grey',alpha=0.2,lw=1.5,label='AntiKT4EMTopo Target Jets')
ax2.set_ylabel('Number of jets', color='grey')
ax2.set_yscale('log')
hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
f.savefig(save_folder + f'/match_frac_pT_2.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


print()
print()


print("=======================================================================================================")
print("Calculating fraction of targets matched, and predictions unmatched for deltaR method!")
frac_p_dRmatch,frac_p_dRunmatch,num_dR_p = [],[],[]
frac_tar_dRmatch,frac_tar_dRunmatch,num_dR_tar = [],[],[]
for bin_idx in range(len(bin_edges)-1):
    # First, get PREDICTED jets in this pT bin
    dRmatch_p_pt_mask   = (bin_edges[bin_idx] < dRmatch_p_pt) & (dRmatch_p_pt < bin_edges[bin_idx+1])
    dRunmatch_p_pt_mask = (bin_edges[bin_idx] < dRunmatch_p_pt) & (dRunmatch_p_pt < bin_edges[bin_idx+1])

    dRmatched_p_bin_i   = dRmatch_p_pt[dRmatch_p_pt_mask]
    dRunmatched_p_bin_i = dRunmatch_p_pt[dRunmatch_p_pt_mask]

    # Next, get TARGET jets in this pT bin
    dRmatch_tar_pt_mask   = (bin_edges[bin_idx] < dRmatch_tar_pt) & (dRmatch_tar_pt < bin_edges[bin_idx+1])
    dRunmatch_tar_pt_mask = (bin_edges[bin_idx] < dRunmatch_tar_pt) & (dRunmatch_tar_pt < bin_edges[bin_idx+1])

    dRmatched_tar_bin_i   = dRmatch_tar_pt[dRmatch_tar_pt_mask]
    dRunmatched_tar_bin_i = dRunmatch_tar_pt[dRunmatch_tar_pt_mask]
    print(f"There are {len(dRmatched_p_bin_i)+len(dRunmatched_p_bin_i)} predicted jets and {len(dRmatched_tar_bin_i)+len(dRunmatched_tar_bin_i)} target jets in bin {bin_idx} ([{bin_edges[bin_idx]}, {bin_edges[bin_idx+1]}])")

    # Finally, get the fraction (un)matched in each bin
    print(f"Fraction of predictions that are matched: {len(dRmatched_p_bin_i)/(len(dRmatched_p_bin_i)+len(dRunmatched_p_bin_i)):.3f}\t unmatched: {len(dRunmatched_p_bin_i)/(len(dRmatched_p_bin_i)+len(dRunmatched_p_bin_i)):.3f}")
    print(f"Fraction of targets that are matched: {len(dRmatched_tar_bin_i)/(len(dRmatched_tar_bin_i)+len(dRunmatched_tar_bin_i)):.3f}\t unmatched: {len(dRunmatched_tar_bin_i)/(len(dRmatched_tar_bin_i)+len(dRunmatched_tar_bin_i)):.3f}\n")

    frac_p_dRmatch.append(len(dRmatched_p_bin_i)/(len(dRmatched_p_bin_i)+len(dRunmatched_p_bin_i)))
    frac_tar_dRmatch.append(len(dRmatched_tar_bin_i)/(len(dRmatched_tar_bin_i)+len(dRunmatched_tar_bin_i)))
    frac_p_dRunmatch.append(len(dRunmatched_p_bin_i)/(len(dRmatched_p_bin_i)+len(dRunmatched_p_bin_i)))
    frac_tar_dRunmatch.append(len(dRunmatched_tar_bin_i)/(len(dRmatched_tar_bin_i)+len(dRunmatched_tar_bin_i)))
    num_dR_p.append(len(dRmatched_p_bin_i)+len(dRunmatched_p_bin_i))
    num_dR_tar.append(len(dRmatched_tar_bin_i)+len(dRunmatched_tar_bin_i))


# calculate error bars by giving the absolute number of (un)matched and the total number in each bin
dRmatch_tar_error = get_errorbars(np.array(frac_tar_dRmatch)*np.array(num_dR_tar), np.array(num_dR_tar))
dRmatch_p_error = get_errorbars(np.array(frac_p_dRmatch)*np.array(num_dR_p), np.array(num_dR_p))
dRunmatch_p_error = get_errorbars(np.array(frac_p_dRunmatch)*np.array(num_dR_p), np.array(num_dR_p))




f,ax = plt.subplots(1,1,figsize=(8, 6))
ax.errorbar(bin_centers,frac_tar_dRmatch,xerr=bin_width/2,yerr=dRmatch_tar_error,color='cyan',marker='.',ms=5.5,elinewidth=1.2,ls='-',label=f'% deltaR Matched Anti-kt jets (Accuracy)')
ax.errorbar(bin_centers,frac_p_dRunmatch,xerr=bin_width/2,yerr=dRunmatch_p_error,color='coral',marker='x',ms=5.5,elinewidth=1.2,ls='-',label=f'% deltaR Unmatched Pred. jets (Fake rate)')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum [GeV]',ylabel=f'Fraction of jets')
ax.set_ylim((0.0,1.2))
ax.legend(loc='lower left',bbox_to_anchor=(0.3, 0.37),fontsize="small")
# hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
# ax.text(45,1.08, r"MC21, $\sqrt{s}=14\,$TeV $<\mu >=200$")
# ax.text(45,1.02, r"Dijet JZ1-4",fontsize='small')
f.savefig(save_folder + f'/dRmatch_frac_pT.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

f,ax = plt.subplots(1,1,figsize=(8, 6))
ax.errorbar(bin_centers,frac_tar_dRmatch,xerr=bin_width/2,yerr=dRmatch_tar_error,color='steelblue',marker='.',ms=7,elinewidth=1.5,ls='none',label=r'$\Delta$R matched targets')
ax.errorbar(bin_centers,frac_p_dRunmatch,xerr=bin_width/2,yerr=dRunmatch_p_error,color='orangered',marker='x',ms=7,elinewidth=1.5,ls='none',label=r'$\Delta$R unmatched predictions')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel=r'Uncalibrated jet $p_T$ [GeV]',ylabel=f'Fraction of jets')
ax.set_ylim((0.0,1.2))
ax.set_xlim((0.0,400))
ax.legend(loc='lower left',bbox_to_anchor=(0.56, 0.37),fontsize="small",handletextpad=0.1)
# hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
# ax.text(45,1.08, r"MC21, $\sqrt{s}=14\,$TeV $<\mu >=200$")
# ax.text(45,1.02, r"Dijet JZ1-4",fontsize='small')
ax.text(10,1.12, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
# ax.text(72,1.12, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
ax.text(72,1.12, "Simulation Preliminary",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
ax.text(10,1.07, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
named_proc = r"$t\bar{t}$" if str(proc[:2])=="tt" else "dijet"
ax.text(10,1.02, f"MC {named_proc} " +  r"$p^{uncalib}_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
f.savefig(save_folder + f'/dRmatch_frac_pT_b.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


f,ax = plt.subplots(1,1,figsize=(8, 6))
ax.errorbar(bin_centers,frac_tar_dRmatch,xerr=bin_width/2,yerr=dRmatch_tar_error,color='cyan',marker='.',ms=5.5,elinewidth=1.2,ls='none',label=f'% Matched Target AKT Jets (Accuracy)')
ax.errorbar(bin_centers,frac_p_dRunmatch,xerr=bin_width/2,yerr=dRunmatch_p_error,color='coral',marker='x',ms=5.5,elinewidth=1.2,ls='none',label=f'% Unmatched Prediction Boxes (Fake rate)')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum (GeV)',ylabel=f'Fraction of jets')
ax.set_ylim((0.0,1.3))
ax.legend(loc='lower left',bbox_to_anchor=(0.35, 0.45),fontsize="small")

ax2 = ax.twinx()
freq_tar, bins, _    = ax2.hist(total_tar_pt,bins=bin_edges,weights=total_jet_weight,histtype='stepfilled',color='grey',alpha=0.2,lw=1.5,label='AntiKT4EMTopo Target Jets')
ax2.set_ylabel('Number of AntiKt4EMTopo jets', color='grey')
ax2.set_yscale('log')
hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
ax.text(45,1.08, r"MC21, $\sqrt{s}=14\,$TeV $<\mu >=200$")
ax.text(45,1.02, r"Dijet JZ1-4",fontsize='small')
f.savefig(save_folder + f'/dRmatch_frac_pT_2.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()






f,ax = plt.subplots(1,1,figsize=(8, 6))
ax.errorbar(bin_centers,frac_tar_dRmatch,xerr=bin_width/2,yerr=dRmatch_tar_error,color='cyan',marker='.',ms=5.5,elinewidth=1.2,ls='-',label=f'% Matched Target AKT Jets (Accuracy)')
ax.errorbar(bin_centers,frac_p_dRunmatch,xerr=bin_width/2,yerr=dRunmatch_p_error,color='coral',marker='x',ms=5.5,elinewidth=1.2,ls='-',label=f'% Unmatched Prediction Boxes (Fake rate)')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum (GeV)',ylabel=f'Fraction of jets')
ax.set_ylim((0.0,1.3))
ax.set_xlim((0.0,400))
ax.legend(loc='lower left',bbox_to_anchor=(0.4, 0.45),fontsize="small")

ax2 = ax.twinx()
freq_tar, bins, _    = ax2.hist(total_tar_pt,bins=bin_edges,weights=total_jet_weight,histtype='stepfilled',color='grey',alpha=0.2,lw=1.5,label='AntiKT4EMTopo Target Jets')
ax2.set_ylabel('Number of AntiKt4EMTopo jets', color='grey')
ax2.set_yscale('log')
hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
ax.text(45,1.08, r"MC21, $\sqrt{s}=14\,$TeV $<\mu >=200$")
ax.text(45,1.02, r"Dijet JZ1-4",fontsize='small')
f.savefig(save_folder + f'/dRmatch_frac_pT_2b.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()





print()
print()
print()
print()
print()
print("=======================================================================================================")
print("Calculating fraction of targets matched, and predictions unmatched for deltaR method with TRUTH jets!")










frac_p_dRtruthmatch,frac_p_dRtruthunmatch,num_dR_p = [],[],[]
frac_tru_dRtruthmatch,frac_tru_dRtruthunmatch,num_dR_tru = [],[],[]
for bin_idx in range(len(bin_edges)-1):
    # First, get PREDICTED jets in this pT bin
    dRtruthmatch_p_pt_mask   = (bin_edges[bin_idx] < dRtruthmatch_p_pt) & (dRtruthmatch_p_pt < bin_edges[bin_idx+1])
    dRtruthunmatch_p_pt_mask = (bin_edges[bin_idx] < dRtruthunmatch_p_pt) & (dRtruthunmatch_p_pt < bin_edges[bin_idx+1])

    dRtruthmatched_p_bin_i   = dRtruthmatch_p_pt[dRtruthmatch_p_pt_mask]
    dRtruthunmatched_p_bin_i = dRtruthunmatch_p_pt[dRtruthunmatch_p_pt_mask]

    # Next, get TRUTH jets in this pT bin
    dRtruthmatch_tru_pt_mask   = (bin_edges[bin_idx] < dRtruthmatch_tru_pt) & (dRtruthmatch_tru_pt < bin_edges[bin_idx+1])
    dRtruthunmatch_tru_pt_mask = (bin_edges[bin_idx] < dRtruthunmatch_tru_pt) & (dRtruthunmatch_tru_pt < bin_edges[bin_idx+1])

    dRtruthmatched_tru_bin_i   = dRtruthmatch_tru_pt[dRtruthmatch_tru_pt_mask]
    dRtruthunmatched_tru_bin_i = dRtruthunmatch_tru_pt[dRtruthunmatch_tru_pt_mask]
    print(f"There are {len(dRtruthmatched_p_bin_i)+len(dRtruthunmatched_p_bin_i)} predicted jets and {len(dRtruthmatched_tru_bin_i)+len(dRtruthunmatched_tru_bin_i)} truth jets in bin {bin_idx} ([{bin_edges[bin_idx]}, {bin_edges[bin_idx+1]}])")

    # Finally, get the fraction (un)matched in each bin
    print(f"Fraction of predictions that are matched: {len(dRtruthmatched_p_bin_i)/(len(dRtruthmatched_p_bin_i)+len(dRtruthunmatched_p_bin_i)):.3f}\t unmatched: {len(dRtruthunmatched_p_bin_i)/(len(dRtruthmatched_p_bin_i)+len(dRtruthunmatched_p_bin_i)):.3f}\t{len(dRtruthmatched_p_bin_i):.3f} / {(len(dRtruthmatched_p_bin_i)+len(dRtruthunmatched_p_bin_i)):.3f}")
    print(f"Fraction of truth that are matched: {len(dRtruthmatched_tru_bin_i)/(len(dRtruthmatched_tru_bin_i)+len(dRtruthunmatched_tru_bin_i)):.3f}\t unmatched: {len(dRtruthunmatched_tru_bin_i)/(len(dRtruthmatched_tru_bin_i)+len(dRtruthunmatched_tru_bin_i)):.3f}\t{len(dRtruthmatched_tru_bin_i):.3f} / {(len(dRtruthmatched_tru_bin_i)+len(dRtruthunmatched_tru_bin_i)):.3f}\n")

    frac_p_dRtruthmatch.append(len(dRtruthmatched_p_bin_i)/(len(dRtruthmatched_p_bin_i)+len(dRtruthunmatched_p_bin_i)))
    frac_tru_dRtruthmatch.append(len(dRtruthmatched_tru_bin_i)/(len(dRtruthmatched_tru_bin_i)+len(dRtruthunmatched_tru_bin_i)))
    frac_p_dRtruthunmatch.append(len(dRtruthunmatched_p_bin_i)/(len(dRtruthmatched_p_bin_i)+len(dRtruthunmatched_p_bin_i)))
    frac_tru_dRtruthunmatch.append(len(dRtruthunmatched_tru_bin_i)/(len(dRtruthmatched_tru_bin_i)+len(dRtruthunmatched_tru_bin_i)))
    num_dR_p.append(len(dRtruthmatched_p_bin_i)+len(dRtruthunmatched_p_bin_i))
    num_dR_tru.append(len(dRtruthmatched_tru_bin_i)+len(dRtruthunmatched_tru_bin_i))


# calculate error bars by giving the absolute number of (un)matched and the total number in each bin
dRtruthmatch_tru_error = get_errorbars(np.array(frac_tru_dRtruthmatch)*np.array(num_dR_tru), np.array(num_dR_tru))
dRtruthmatch_p_error = get_errorbars(np.array(frac_p_dRtruthmatch)*np.array(num_dR_p), np.array(num_dR_p))
dRtruthunmatch_p_error = get_errorbars(np.array(frac_p_dRtruthunmatch)*np.array(num_dR_p), np.array(num_dR_p))

f,ax = plt.subplots(1,1,figsize=(8, 6))
ax.errorbar(bin_centers,frac_tru_dRtruthmatch,xerr=bin_width/2,yerr=dRtruthmatch_tru_error,color='gold',marker='.',ms=5.5,elinewidth=1.2,ls='none',label=f'% Matched Truth Jets (Accuracy)')
ax.errorbar(bin_centers,frac_p_dRtruthunmatch,xerr=bin_width/2,yerr=dRtruthunmatch_p_error,color='chocolate',marker='x',ms=5.5,elinewidth=1.2,ls='none',label=f'% Unmatched Prediction Boxes (Fake rate)')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum (GeV)',ylabel=f'Fraction of boxes')
ax.set_ylim((-0.2,1.2))
ax.legend(loc='lower left',bbox_to_anchor=(0.005, 0.005),fontsize="x-small")
hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
f.savefig(save_folder + f'/dRtruthmatch_frac_pT.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

f,ax = plt.subplots(1,1,figsize=(8, 6))
ax.errorbar(bin_centers,frac_tru_dRtruthmatch,xerr=bin_width/2,yerr=dRtruthmatch_tru_error,color='gold',marker='.',ms=5.5,elinewidth=1.2,ls='none',label=f'% Matched Truth Jets (Accuracy)')
ax.errorbar(bin_centers,frac_p_dRtruthunmatch,xerr=bin_width/2,yerr=dRtruthunmatch_p_error,color='chocolate',marker='x',ms=5.5,elinewidth=1.2,ls='none',label=f'% Unmatched Prediction Boxes (Fake rate)')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum (GeV)',ylabel=f'Fraction of boxes')
ax.set_ylim((0.0,1.2))
ax.legend(loc='center right',bbox_to_anchor=(0.005, 0.005),fontsize="x-small")

ax2 = ax.twinx()
freq_tar, bins, _    = ax2.hist(total_tru_pt,bins=bin_edges,histtype='stepfilled',color='grey',alpha=0.2,lw=1.5,label='AntiKT4 Truth Jets')
ax2.set_ylabel('Number of jets', color='grey')
ax2.set_yscale('log')
hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
f.savefig(save_folder + f'/dRtruthmatch_frac_pT_2.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()



print()
print()
print()
print()
print()
print("=======================================================================================================")
print("Calculating fraction of TARGETS matched to TRUTH, with deltaR method!")

frac_tar_dRtruthtarmatch,frac_tar_dRtruthtarunmatch,num_dR_tar = [],[],[]
frac_tru_dRtruthtarmatch,frac_tru_dRtruthtarunmatch,num_dR_tru = [],[],[]
for bin_idx in range(len(bin_edges)-1):
    # First, get TARGET jets in this pT bin
    dRtruthtarmatch_tar_pt_mask   = (bin_edges[bin_idx] < dRtruthtarmatch_tar_pt) & (dRtruthtarmatch_tar_pt < bin_edges[bin_idx+1])
    dRtruthtarunmatch_tar_pt_mask = (bin_edges[bin_idx] < dRtruthtarunmatch_tar_pt) & (dRtruthtarunmatch_tar_pt < bin_edges[bin_idx+1])

    dRtruthtarmatched_tar_bin_i   = dRtruthtarmatch_tar_pt[dRtruthtarmatch_tar_pt_mask]
    dRtruthtarunmatched_tar_bin_i = dRtruthtarunmatch_tar_pt[dRtruthtarunmatch_tar_pt_mask]

    # Next, get TRUTH jets in this pT bin
    dRtruthtarmatch_tru_pt_mask   = (bin_edges[bin_idx] < dRtruthtarmatch_tru_pt) & (dRtruthtarmatch_tru_pt < bin_edges[bin_idx+1])
    dRtruthtarunmatch_tru_pt_mask = (bin_edges[bin_idx] < dRtruthtarunmatch_tru_pt) & (dRtruthtarunmatch_tru_pt < bin_edges[bin_idx+1])

    dRtruthtarmatched_tru_bin_i   = dRtruthtarmatch_tru_pt[dRtruthtarmatch_tru_pt_mask]
    dRtruthtarunmatched_tru_bin_i = dRtruthtarunmatch_tru_pt[dRtruthtarunmatch_tru_pt_mask]
    print(f"There are {len(dRtruthtarmatched_tar_bin_i)+len(dRtruthtarunmatched_tar_bin_i)} target jets and {len(dRtruthtarmatched_tru_bin_i)+len(dRtruthtarunmatched_tru_bin_i)} truth jets in bin {bin_idx} ([{bin_edges[bin_idx]}, {bin_edges[bin_idx+1]}])")

    # Finally, get the fraction (un)matched in each bin
    print(f"Fraction of target (AKT) jets that are matched: {len(dRtruthtarmatched_tar_bin_i)/(len(dRtruthtarmatched_tar_bin_i)+len(dRtruthtarunmatched_tar_bin_i)):.3f}\t unmatched: {len(dRtruthtarunmatched_tar_bin_i)/(len(dRtruthtarmatched_tar_bin_i)+len(dRtruthtarunmatched_tar_bin_i)):.3f}\t {len(dRtruthtarmatched_tar_bin_i):.3f} / {(len(dRtruthtarmatched_tar_bin_i)+len(dRtruthtarunmatched_tar_bin_i)):.3f}")
    print(f"Fraction of truth jets that are matched: {len(dRtruthtarmatched_tru_bin_i)/(len(dRtruthtarmatched_tru_bin_i)+len(dRtruthtarunmatched_tru_bin_i)):.3f}\t unmatched: {len(dRtruthtarunmatched_tru_bin_i)/(len(dRtruthtarmatched_tru_bin_i)+len(dRtruthtarunmatched_tru_bin_i)):.3f}\t{len(dRtruthtarmatched_tru_bin_i):.3f} / {(len(dRtruthtarmatched_tru_bin_i)+len(dRtruthtarunmatched_tru_bin_i)):.3f}\n")

    frac_tar_dRtruthtarmatch.append(len(dRtruthtarmatched_tar_bin_i)/(len(dRtruthtarmatched_tar_bin_i)+len(dRtruthtarunmatched_tar_bin_i)))
    frac_tru_dRtruthtarmatch.append(len(dRtruthtarmatched_tru_bin_i)/(len(dRtruthtarmatched_tru_bin_i)+len(dRtruthtarunmatched_tru_bin_i)))
    frac_tar_dRtruthtarunmatch.append(len(dRtruthtarunmatched_tar_bin_i)/(len(dRtruthtarmatched_tar_bin_i)+len(dRtruthtarunmatched_tar_bin_i)))
    frac_tru_dRtruthtarunmatch.append(len(dRtruthtarunmatched_tru_bin_i)/(len(dRtruthtarmatched_tru_bin_i)+len(dRtruthtarunmatched_tru_bin_i)))
    num_dR_tar.append(len(dRtruthtarmatched_tar_bin_i)+len(dRtruthtarunmatched_tar_bin_i))
    num_dR_tru.append(len(dRtruthtarmatched_tru_bin_i)+len(dRtruthtarunmatched_tru_bin_i))


# calculate error bars by giving the absolute number of (un)matched and the total number in each bin
dRtruthtarmatch_tru_error = get_errorbars(np.array(frac_tru_dRtruthtarmatch)*np.array(num_dR_tru), np.array(num_dR_tru))
dRtruthtarmatch_tar_error = get_errorbars(np.array(frac_tar_dRtruthtarmatch)*np.array(num_dR_tar), np.array(num_dR_tar))
dRtruthtarunmatch_tar_error = get_errorbars(np.array(frac_tar_dRtruthtarunmatch)*np.array(num_dR_tar), np.array(num_dR_tar))

f,ax = plt.subplots(1,1,figsize=(8, 6))
ax.errorbar(bin_centers,frac_tru_dRtruthtarmatch,xerr=bin_width/2,yerr=dRtruthmatch_tru_error,color='gold',marker='.',ms=5.5,elinewidth=1.2,ls='none',label=f'% Matched Truth Jets (Accuracy)')
ax.errorbar(bin_centers,frac_tar_dRtruthtarunmatch,xerr=bin_width/2,yerr=dRtruthtarunmatch_tar_error,color='lightseagreen',marker='x',ms=5.5,elinewidth=1.2,ls='none',label=f'% Unmatched Target (AKT) Jets (Fake rate)')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum (GeV)',ylabel=f'Fraction of jets')
ax.set_ylim((-0.2,1.2))
ax.legend(loc='lower left',bbox_to_anchor=(0.005, 0.005),fontsize="x-small")
hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
f.savefig(save_folder + f'/dRtruthtarmatch_frac_pT.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


f,ax = plt.subplots(1,1,figsize=(8, 6))
ax.errorbar(bin_centers,frac_tru_dRtruthtarmatch,xerr=bin_width/2,yerr=dRtruthmatch_tru_error,color='gold',marker='o',ms=5.5,elinewidth=1.2,ls='-',label=f'% Matched Truth-AKT')
ax.errorbar(bin_centers,frac_tar_dRtruthtarunmatch,xerr=bin_width/2,yerr=dRtruthtarunmatch_tar_error,color='lightseagreen',marker='o',ms=5.5,elinewidth=1.2,ls='-',label=f'% Unmatched AKT Jets')
ax.errorbar(bin_centers,frac_tru_dRtruthmatch,xerr=bin_width/2,yerr=dRtruthmatch_tru_error,color='maroon',marker='s',ms=5.5,elinewidth=1.2,ls='-',label=f'% Matched Truth-Pred')
ax.errorbar(bin_centers,frac_p_dRtruthunmatch,xerr=bin_width/2,yerr=dRtruthunmatch_p_error,color='chocolate',marker='s',ms=5.5,elinewidth=1.2,ls='-',label=f'% Unmatched Pred Boxes')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum (GeV)',ylabel=f'Fraction of jets')
ax.set_ylim((0.0,1.2))
ax.legend(loc='lower left',bbox_to_anchor=(0.55, 0.37),fontsize="small")
hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
f.savefig(save_folder + f'/comp_dRtruthtarmatch_frac_pT.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()





# total_tar_pt      = np.concatenate(load_object(metrics_folder+"/tarboxes_pt.pkl"))
# total_tru_pt      = np.concatenate(load_object(metrics_folder+"/truboxes_pt.pkl"))
# total_p_pt        = np.concatenate(load_object(metrics_folder+"/pboxes_pt.pkl"))

print()
print()
for bin_idx in range(len(bin_edges)-1):
    # First, get PREDICTED jets in this pT bin
    total_p_pt_mask = (bin_edges[bin_idx] < total_p_pt) & (total_p_pt < bin_edges[bin_idx+1])
    total_p_bin_i   = total_p_pt[total_p_pt_mask]
    dRtruthmatch_p_pt_mask   = (bin_edges[bin_idx] < dRtruthmatch_p_pt) & (dRtruthmatch_p_pt < bin_edges[bin_idx+1])
    dRtruthunmatch_p_pt_mask = (bin_edges[bin_idx] < dRtruthunmatch_p_pt) & (dRtruthunmatch_p_pt < bin_edges[bin_idx+1])
    dRtruthmatched_p_bin_i   = dRtruthmatch_p_pt[dRtruthmatch_p_pt_mask]
    dRtruthunmatched_p_bin_i = dRtruthunmatch_p_pt[dRtruthunmatch_p_pt_mask]

    # Then, get TARGET jets in this pT bin
    total_tar_pt_mask = (bin_edges[bin_idx] < total_tar_pt) & (total_tar_pt < bin_edges[bin_idx+1])
    total_tar_bin_i   = total_tar_pt[total_tar_pt_mask]
    dRmatch_tar_pt_mask   = (bin_edges[bin_idx] < dRmatch_tar_pt) & (dRmatch_tar_pt < bin_edges[bin_idx+1])
    dRunmatch_tar_pt_mask = (bin_edges[bin_idx] < dRunmatch_tar_pt) & (dRunmatch_tar_pt < bin_edges[bin_idx+1])
    dRmatched_tar_bin_i   = dRmatch_tar_pt[dRmatch_tar_pt_mask]
    dRunmatched_tar_bin_i = dRunmatch_tar_pt[dRunmatch_tar_pt_mask]

    # Next, get TRUTH jets in this pT bin
    total_tru_pt_mask = (bin_edges[bin_idx] < total_tru_pt) & (total_tru_pt < bin_edges[bin_idx+1])
    total_tru_bin_i   = total_tru_pt[total_tru_pt_mask]
    dRtruthmatch_tru_pt_mask   = (bin_edges[bin_idx] < dRtruthmatch_tru_pt) & (dRtruthmatch_tru_pt < bin_edges[bin_idx+1])
    dRtruthunmatch_tru_pt_mask = (bin_edges[bin_idx] < dRtruthunmatch_tru_pt) & (dRtruthunmatch_tru_pt < bin_edges[bin_idx+1])
    dRtruthmatched_tru_bin_i   = dRtruthmatch_tru_pt[dRtruthmatch_tru_pt_mask]
    dRtruthunmatched_tru_bin_i = dRtruthunmatch_tru_pt[dRtruthunmatch_tru_pt_mask]

    print(f"There are {len(total_p_bin_i)} ({len(dRtruthmatched_p_bin_i)}, {len(dRtruthunmatched_p_bin_i)}) predicted jets there are {len(total_tar_bin_i)} ({len(dRmatched_tar_bin_i)}, {len(dRunmatched_tar_bin_i)}) target jets and {len(total_tru_bin_i)} ({len(dRtruthmatched_tru_bin_i)}, {len(dRtruthunmatched_tru_bin_i)}) truth jets in bin {bin_idx} ([{bin_edges[bin_idx]}, {bin_edges[bin_idx+1]}])")
    print()
print("To be investigated further... small discrepancies!")


