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

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496


def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)



model_name = "jetSSD_smallconvnext_central_32e"
metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/JZ4/20250124-13/box_metrics"
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/JZ4/20250124-13/jet_match/"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)



print("==================================================================================================")
print(f"Loading matched jets from\n{metrics_folder}")
print("==================================================================================================\n")

#by event
total_n_truth = load_object(metrics_folder+"/n_truth.pkl")
total_n_preds = load_object(metrics_folder+"/n_preds.pkl")

total_n_tru_match = load_object(metrics_folder+"/n_matched_truth.pkl")
total_n_tru_unmatch = load_object(metrics_folder+"/n_unmatched_truth.pkl")
total_n_pre_match = load_object(metrics_folder+"/n_matched_preds.pkl")
total_n_pre_unmatch = load_object(metrics_folder+"/n_unmatched_preds.pkl")

#by jet
total_tru_pt = np.concatenate(load_object(metrics_folder+"/tboxes_pt.pkl"))
total_tru_eta = np.concatenate(load_object(metrics_folder+"/tboxes_eta.pkl"))
total_tru_phi = np.concatenate(load_object(metrics_folder+"/tboxes_phi.pkl"))
total_tru_matched = np.concatenate(load_object(metrics_folder+"/tboxes_matched.pkl"))

total_pred_matched = np.concatenate(load_object(metrics_folder+"/pboxes_matched.pkl"))
total_pred_scores = np.concatenate(load_object(metrics_folder+"/pboxes_scores.pkl"))
total_pred_pt = np.concatenate(load_object(metrics_folder+"/pboxes_pt.pkl"))
total_pred_eta = np.concatenate(load_object(metrics_folder+"/pboxes_eta.pkl"))
total_pred_phi = np.concatenate(load_object(metrics_folder+"/pboxes_phi.pkl"))

total_matched_tru_pt = np.concatenate(load_object(metrics_folder+"/tboxes_matched_pt.pkl"))
total_matched_pred_pt = np.concatenate(load_object(metrics_folder+"/pboxes_matched_pt.pkl"))


print(f"In the unseen test data we have {len(total_n_truth)} events, with a total {len(total_tru_pt)} AntiKt4EMTopo Jets and {len(total_pred_pt)} jets predicted by the CNN")
print(f"Of the {len(total_pred_eta)} predicted boxes {sum(total_pred_matched)} are matched, {len(total_matched_tru_pt)} or {len(total_matched_pred_pt)}")
print(f"Of the {len(total_tru_pt)} target boxes {sum(total_tru_matched)} are matched.")
print("total_n_tru_match",sum(total_n_tru_match))
print("total_n_tru_unmatch",sum(total_n_tru_unmatch))
print("total_n_pre_match",sum(total_n_pre_match))
print("total_n_pre_unmatch",sum(total_n_pre_unmatch))
quit()







pt_thresholds = [0,10,15,20,25,30,35,40,60,100,200,250,450,625]
tru_matched_percentage = []
for threshold in pt_thresholds:
    mask = total_tru_pt > threshold
    total_tru_matched_mask = total_tru_matched[mask]
    tru_matched_percentage.append(np.round(sum(total_tru_matched_mask) / len(total_tru_matched_mask),4))
    print(f'For jets above {threshold}: {sum(total_tru_matched_mask)/len(total_tru_matched_mask)} of truth jets are matched')
    print(f'There are {len(total_tru_pt[mask])} jets above this threshold\n')

print()
pred_matched_percentage = []
for threshold in pt_thresholds:
    mask = total_pred_pt > threshold
    total_pred_matched_mask = total_pred_matched[mask]
    pred_matched_percentage.append(np.round(sum(total_pred_matched_mask) / len(total_pred_matched_mask),4))
    print(f'For jets above {threshold}: {sum(total_pred_matched_mask) / len(total_pred_matched_mask)} of predicted jets are matched')
    print(f'There are {len(total_pred_pt[mask])} jets above this threshold\n')

print("lists")
print(tru_matched_percentage)
print(pred_matched_percentage)













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
bin_edges = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, max(total_pred_pt)]
bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
bin_width = np.diff(bin_edges)




percentage_matched_in_pt,percentage_unmatched_in_pt = list(), list()
n_matched_preds,n_unmatched_preds = list(), list()
n_preds,n_truth = list(), list()
for bin_idx in range(len(bin_edges)-1):
    # print(bin_edges[bin_idx],bin_edges[bin_idx+1])
    bin_mask = (bin_edges[bin_idx]<total_pred_pt) & (total_pred_pt<bin_edges[bin_idx+1])
    num_predictions = len(total_pred_matched[bin_mask])
    num_matched_predictions = sum(total_pred_matched[bin_mask])
    
    bin_mask_tru = (bin_edges[bin_idx]<total_tru_pt) & (total_tru_pt<bin_edges[bin_idx+1])
    print('num_predictions',num_predictions,'num_matched_predictions',num_matched_predictions, num_matched_predictions/num_predictions)
    num_truth = len(total_tru_matched[bin_mask_tru])
    num_matched_truth = sum(total_tru_matched[bin_mask_tru])
    num_unmatched_predictions = num_predictions - num_matched_predictions #np.count_nonzero(total_pred_matched[(bin_edges[bin_idx]<total_pred_eT) & (total_pred_eT<bin_edges[bin_idx+1])]==0)
    try:
        print('num_truth',num_truth,'num_matched_truth',num_matched_truth, num_matched_truth/num_truth)
    except ZeroDivisionError:
        pass
    percentage_matched_in_pt.append(num_matched_truth/num_truth)
    # percentage_matched_in_pt.append(num_matched_predictions/num_predictions)
    percentage_unmatched_in_pt.append(num_unmatched_predictions/num_predictions)
    n_matched_preds.append(num_matched_predictions)
    n_unmatched_preds.append(num_unmatched_predictions)
    n_preds.append(num_predictions)
    n_truth.append(num_truth)
print("--->")
print(n_truth)
print(n_preds)
match_pred_errro = get_errorbars(np.array(n_matched_preds),np.array(n_truth))
unmatch_pred_errro = get_errorbars(np.array(n_unmatched_preds),np.array(n_preds))





f,ax = plt.subplots(1,1,figsize=(8, 6))
ax.plot(bin_centers,percentage_matched_in_pt,marker='x',color='cyan',label=f'% Matched Truth Boxes (Accuracy)')
ax.plot(bin_centers,percentage_unmatched_in_pt,marker='+',color='coral',label=f'% Unmatched Prediction Boxes (Fake rate)')
# ax.errorbar(bin_centers,percentage_matched_in_pt,xerr=bin_width/2,yerr=match_pred_errro,color='black',marker='.',ms=5.5,elinewidth=1.2,ls='none',label=f'% Matched Truth Boxes (Accuracy)')
# ax.errorbar(bin_centers,percentage_unmatched_in_pt,xerr=bin_width/2,yerr=unmatch_pred_errro,color='black',marker='x',ms=5.5,elinewidth=1.2,ls='none',label=f'% Unmatched Prediction Boxes (Fake rate)')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum (GeV)',ylabel=f'Fraction of boxes',title='Box Matching Test Set')
ax.set_ylim((-0.2,1.2))
ax.legend(loc='lower left',bbox_to_anchor=(0.005, 0.005),fontsize="x-small")
print('Bin centres',bin_centers)
print(percentage_matched_in_pt)
print(percentage_unmatched_in_pt)

hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
f.savefig(eff_save_loc + f'/match_frac_boxes_pT.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()









