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




def make_poster_plots(
    poster_folder,
    save_folder,
    log=True,
    image_format='png',
):
    save_loc = save_folder + f"/poster/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    #####################################################################################################################################
    #Plot 1

    total_pred_matched = np.concatenate(load_object(poster_folder + '/pboxes_matched.pkl'))
    total_tru_matched = np.concatenate(load_object(poster_folder + '/tboxes_matched.pkl'))
    total_pred_eT = np.concatenate(load_object(poster_folder + '/pboxes_eT.pkl'))
    total_tru_eT = np.concatenate(load_object(poster_folder + '/tboxes_eT.pkl'))
    total_pred_eta = np.concatenate(load_object(poster_folder + '/pboxes_eta.pkl'))
    total_tru_eta = np.concatenate(load_object(poster_folder + '/tboxes_eta.pkl'))
    # print(len(total_pred_eT),len(total_pred_matched))
    # print(len(total_tru_eT),len(total_tru_matched))
    
    bin_edges = [0,20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, max(max(total_pred_eT),max(total_pred_eT))/1000]
    bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
    bin_width = np.diff(bin_edges)
    percentage_matched_in_et,percentage_unmatched_in_et = [],[]
    n_matched_preds,n_unmatched_preds = [], []
    n_preds,n_truth = [],[]
    for bin_idx in range(len(bin_edges)-1):
        # print(bin_edges[bin_idx],bin_edges[bin_idx+1])
        num_predictions = len(total_pred_matched[(bin_edges[bin_idx]<total_pred_eT/1000) & (total_pred_eT/1000<bin_edges[bin_idx+1])])
        num_matched_predictions = sum(total_pred_matched[(bin_edges[bin_idx]<total_pred_eT/1000) & (total_pred_eT/1000<bin_edges[bin_idx+1])])
        num_matched_truth = sum(total_tru_matched[(bin_edges[bin_idx]<total_tru_eT/1000) & (total_tru_eT/1000<bin_edges[bin_idx+1])])
        num_unmatched_predictions = num_predictions - num_matched_predictions #np.count_nonzero(total_pred_matched[(bin_edges[bin_idx]<total_pred_eT) & (total_pred_eT<bin_edges[bin_idx+1])]==0)
        num_truth = len(total_tru_matched[(bin_edges[bin_idx]<total_tru_eT/1000) & (total_tru_eT/1000<bin_edges[bin_idx+1])])
        percentage_matched_in_et.append(num_matched_predictions/num_truth)
        percentage_unmatched_in_et.append(num_unmatched_predictions/num_predictions)
        n_matched_preds.append(num_matched_predictions)
        n_unmatched_preds.append(num_unmatched_predictions)
        n_preds.append(num_predictions)
        n_truth.append(num_truth)

    # print(percentage_unmatched_in_et)
    # print(percentage_matched_in_et)
    # print(len(percentage_matched_in_et),len(bin_centers))
    # print(len(percentage_unmatched_in_et),len(bin_centers))


    match_pred_errro = get_errorbars(np.array(n_matched_preds),np.array(n_truth))
    unmatch_pred_errro = get_errorbars(np.array(n_unmatched_preds),np.array(n_preds))
    # print(match_pred_errro)
    # print(unmatch_pred_errro)
    # print('\n\n\n')
    f,ax = plt.subplots(1,1,figsize=(8, 6))
    # ax.scatter(bin_centers,percentage_matched_in_et,marker='x',s=50,color='black',label=f'% Matched Truth Boxes (Accuracy)')
    # ax.scatter(bin_centers,percentage_unmatched_in_et,marker='+',s=64,color='grey',label=f'% Unmatched Prediction Boxes (Fake rate)')
    ax.errorbar(bin_centers,percentage_matched_in_et,xerr=bin_width/2,yerr=match_pred_errro,color='black',marker='.',ms=5.5,elinewidth=1.2,ls='none',label=f'% Matched Truth Boxes (Accuracy)')
    ax.errorbar(bin_centers,percentage_unmatched_in_et,xerr=bin_width/2,yerr=unmatch_pred_errro,color='black',marker='x',ms=5.5,elinewidth=1.2,ls='none',label=f'% Unmatched Prediction Boxes (Fake rate)')
    ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
    ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
    ax.set(xlabel='Transverse Energy',ylabel=f'Fraction of boxes',title='Box Matching Test Set')
    ax.set_ylim((-0.2,1.2))
    ax.legend(loc='lower left',bbox_to_anchor=(0.005, 0.005),fontsize="x-small")
    
    ax2 = ax.twinx()
    n_pbox, bins, _ = ax2.hist(total_pred_eT/1000,bins=bin_edges,histtype='step',alpha=0.35,color='red',lw=1.5,label='Predicted Boxes ({})'.format(len(total_pred_eT)))
    n_tbox, _, _ = ax2.hist(total_tru_eT/1000,bins=bin_edges,histtype='step',alpha=0.35,color='green',lw=1.5,label='Truth Boxes ({})'.format(len(total_tru_eT)))

    ax2.set(yscale='log',xlabel='Number of boxes')
    ax2.legend(loc='lower left',bbox_to_anchor=(0.65, 0.85),fontsize="x-small")
    ax2.set_xticks(ax2.get_xticks())
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylabel("Number of Boxes",color="tab:green",fontsize=17,alpha=0.55)
    plt.setp(ax2.get_yticklabels(), alpha=0.4)
    hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
    f.savefig(save_loc + f'/total_boxes_eT_log_25.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()











    ####################################################################################
    #Plot 2

    bin_edges = [-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5]
    bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
    bin_width = np.diff(bin_edges)
    percentage_matched_in_eta,percentage_unmatched_in_eta = [],[]
    n_matched_preds,n_unmatched_preds = [], []
    n_preds,n_truth = [],[]
    for bin_idx in range(len(bin_edges)-1):
        # print(bin_edges[bin_idx],bin_edges[bin_idx+1])
        num_predictions = len(total_pred_matched[(bin_edges[bin_idx]<total_pred_eta) & (total_pred_eta<bin_edges[bin_idx+1])])
        num_matched_predictions = sum(total_pred_matched[(bin_edges[bin_idx]<total_pred_eta) & (total_pred_eta<bin_edges[bin_idx+1])])
        num_matched_truth = sum(total_tru_matched[(bin_edges[bin_idx]<total_tru_eta) & (total_tru_eta<bin_edges[bin_idx+1])])
        num_unmatched_predictions = num_predictions - num_matched_predictions #np.count_nonzero(total_pred_matched[(bin_edges[bin_idx]<total_pred_eta) & (total_pred_eta<bin_edges[bin_idx+1])]==0)
        num_truth = len(total_tru_matched[(bin_edges[bin_idx]<total_tru_eta) & (total_tru_eta<bin_edges[bin_idx+1])])
        percentage_matched_in_eta.append(num_matched_predictions/num_truth)
        percentage_unmatched_in_eta.append(num_unmatched_predictions/num_predictions)
        n_matched_preds.append(num_matched_predictions)
        n_unmatched_preds.append(num_unmatched_predictions)
        n_preds.append(num_predictions)
        n_truth.append(num_truth)
    # print('-------------------------------------------')
    # print(percentage_matched_in_eta)
    # print(percentage_unmatched_in_eta)
    # print(len(percentage_matched_in_eta),len(bin_centers))
    # print(len(percentage_unmatched_in_eta),len(bin_centers))


    match_pred_error = get_errorbars(np.array(n_matched_preds),np.array(n_truth))
    unmatch_pred_error = get_errorbars(np.array(n_unmatched_preds),np.array(n_preds))
    # print('-------------------------------------------')
    # print(match_pred_errro)
    # print(unmatch_pred_errro)

    f,ax = plt.subplots(1,1,figsize=(8, 6))
    # ax.scatter(bin_centers,percentage_matched_in_eta,marker='x',s=50,color='black',label=f'% Matched Truth Boxes (Accuracy)')
    # ax.scatter(bin_centers,percentage_unmatched_in_eta,marker='+',s=64,color='grey',label=f'% Unmatched Prediction Boxes (Fake rate)')
    ax.errorbar(bin_centers,percentage_matched_in_eta,xerr=bin_width/2,yerr=match_pred_error,color='black',marker='.',ms=5.5,elinewidth=1.2,ls='none',label=f'% Matched Truth Boxes (Accuracy)')
    ax.errorbar(bin_centers,percentage_unmatched_in_eta,xerr=bin_width/2,yerr=unmatch_pred_error,color='grey',marker='x',ms=5.5,elinewidth=1.2,ls='none',label=f'% Unmatched Prediction Boxes (Fake rate)')
    ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
    ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
    ax.set(xlabel='Transverse Energy',ylabel=f'Fraction of boxes',title='Box Matching Test Set')
    ax.set_ylim((-0.2,1.2))
    ax.set_xlim((-4.5,4.5))
    ax.legend(loc='lower left',bbox_to_anchor=(0.005, 0.005),fontsize="x-small")
    
    ax2 = ax.twinx()
    n_pbox, bins, _ = ax2.hist(total_pred_eta,bins=bin_edges,histtype='step',alpha=0.35,color='red',lw=1.5,label='Predicted Boxes ({})'.format(len(total_pred_eta)))
    n_tbox, _, _ = ax2.hist(total_tru_eta,bins=bin_edges,histtype='step',alpha=0.35,color='green',lw=1.5,label='Truth Boxes ({})'.format(len(total_tru_eta)))

    ax2.set(yscale='log',xlabel='Number of boxes')
    ax2.legend(loc='lower left',bbox_to_anchor=(0.65, 0.85),fontsize="x-small")
    ax2.set_xticks(ax2.get_xticks())
    hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
    f.savefig(save_loc + f'/total_boxes_eta_log_25.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()




    ####################################################################################
    #Plot 3
    total_pred_matched = np.concatenate(load_object(poster_folder + '/pboxes_matched.pkl'))
    total_tru_matched = np.concatenate(load_object(poster_folder + '/tboxes_matched.pkl'))
    total_pred_eta = np.concatenate(load_object(poster_folder + '/pboxes_eta.pkl'))
    total_tru_eta = np.concatenate(load_object(poster_folder + '/tboxes_eta.pkl'))
    total_pred_phi = np.concatenate(load_object(poster_folder + '/pboxes_phi.pkl'))
    total_tru_phi = np.concatenate(load_object(poster_folder + '/tboxes_phi.pkl'))
    print(len(total_pred_matched),len(total_pred_eta),len(total_pred_phi))
    matched_pred_eta = total_pred_eta[np.nonzero(total_pred_matched!=0)]
    unmatched_pred_eta = total_pred_eta[np.nonzero(total_pred_matched==0)]
    matched_pred_phi = total_pred_phi[np.nonzero(total_pred_matched!=0)]
    unmatched_pred_phi = total_pred_phi[np.nonzero(total_pred_matched==0)]

    matched_truth_eta = total_tru_eta[np.nonzero(total_tru_matched!=0)]
    unmatched_truth_eta = total_tru_eta[np.nonzero(total_tru_matched==0)]
    matched_truth_phi = total_tru_phi[np.nonzero(total_tru_matched!=0)]
    unmatched_truth_phi = total_tru_phi[np.nonzero(total_tru_matched==0)]
    print(len(matched_pred_eta),sum(total_pred_matched))
    print(len(unmatched_pred_eta),len(total_pred_matched)-sum(total_pred_matched))
    print(len(unmatched_pred_phi),len(total_pred_matched)-sum(total_pred_matched))

    #np.linspace(start, stop, int((stop - start) / step + 1))
    bins_x = np.linspace(MIN_CELLS_ETA, MAX_CELLS_ETA, (int((MAX_CELLS_ETA - MIN_CELLS_ETA) / (0.1)) + 1))
    bins_y = np.linspace(MIN_CELLS_PHI, MAX_CELLS_PHI, (int((MAX_CELLS_PHI - MIN_CELLS_PHI) / ((2*np.pi)/64)) + 1))
    extent = (MIN_CELLS_ETA,MAX_CELLS_ETA,MIN_CELLS_PHI,MAX_CELLS_PHI)

    H_match,_,_ = np.histogram2d(matched_pred_eta,matched_pred_phi,bins=(bins_x,bins_y),weights=np.ones_like(matched_pred_eta)/10_000)
    H_unmatch,_,_ = np.histogram2d(unmatched_pred_eta,unmatched_pred_phi,bins=(bins_x,bins_y),weights=np.ones_like(unmatched_pred_phi)/10_000)
    H_match = H_match.T
    H_unmatch = H_unmatch.T

    H_matcht,_,_ = np.histogram2d(matched_truth_eta,matched_truth_phi,bins=(bins_x,bins_y),weights=np.ones_like(matched_truth_eta)/10_000)
    H_unmatcht,_,_ = np.histogram2d(unmatched_truth_eta,unmatched_truth_phi,bins=(bins_x,bins_y),weights=np.ones_like(unmatched_truth_phi)/10_000)
    H_matcht = H_matcht.T
    H_unmatcht = H_unmatcht.T

    f,ax = plt.subplots()
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])
    cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","green","lime"])
    ii = ax.imshow(H_match,extent=extent,cmap=cmap1)
    cbar = f.colorbar(ii,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('Num. Matched Predictions', rotation=90)
    cbar.ax.set_yticklabels(['{:.3f}'.format(x) for x in cbar.ax.get_yticks()])
    ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
    ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
    ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
    ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
    ax.set(xlabel='eta',ylabel='phi')
    # f.savefig(save_loc + f'/matched_predictions_2d.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    f.savefig(save_loc + f'/matched_predictions_2d.{image_format}',dpi=400,format=image_format)
    plt.close()

    f,ax = plt.subplots()
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","violet","red"])
    i = ax.imshow(H_unmatch,extent=extent,cmap=cmap2)
    cbar = f.colorbar(i,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    # cbar.set_label('Num. Unmatched Predictions per event', rotation=90)
    cbar.set_label('Fake Prediction Rate', rotation=90)
    # cbar.ax.set_yticks(cbar.ax.get_yticks()) 
    cbar.ax.set_yticklabels(['{:.3f}'.format(x) for x in cbar.ax.get_yticks()])
    ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
    ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
    ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
    ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
    ax.set(xlabel='eta',ylabel='phi')
    f.savefig(save_loc + f'/unmatched_predictions_2d.{image_format}',dpi=400,format=image_format)
    plt.close()


    # truth boxes:
    f,ax = plt.subplots()
    ii = ax.imshow(H_matcht,extent=extent,cmap=cmap1)
    cbar = f.colorbar(ii,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('Num. Matched Truth', rotation=90)
    cbar.ax.set_yticklabels(['{:.3f}'.format(x) for x in cbar.ax.get_yticks()])
    ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
    ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
    ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
    ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
    ax.set(xlabel='eta',ylabel='phi')
    # f.savefig(save_loc + f'/matched_predictions_2d.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    f.savefig(save_loc + f'/matched_truth_2d.{image_format}',dpi=400,format=image_format)
    plt.close()

    f,ax = plt.subplots()
    i = ax.imshow(H_unmatcht,extent=extent,cmap=cmap2)
    cbar = f.colorbar(i,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    # cbar.set_label('Num. Unmatched Predictions per event', rotation=90)
    cbar.set_label('Missed truth box rate', rotation=90)
    # cbar.ax.set_yticks(cbar.ax.get_yticks()) 
    cbar.ax.set_yticklabels(['{:.3f}'.format(x) for x in cbar.ax.get_yticks()])
    ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
    ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
    ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
    ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
    ax.set(xlabel='eta',ylabel='phi')
    f.savefig(save_loc + f'/unmatched_truth_2d.{image_format}',dpi=400,format=image_format)
    plt.close()


if __name__=="__main__":
    poster_folder = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD1_50k5_mu_15e/new_phys_metrics/poster25"
    save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_plots/SSD1_50k5_mu_15e/"
    print('Making poster plots')
    make_poster_plots(poster_folder,save_at)
    print(1)
    print('Completed poster plots\n')

