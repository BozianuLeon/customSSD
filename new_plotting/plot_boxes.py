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

def is_nested_list(metric):
    #check if we have list or list of list of lists
    if isinstance(metric, list):
        return isinstance(metric[0], list) if metric else False

def get_ratio(numer,denom):
    result = np.zeros_like(numer, dtype=float)
    non_zero_indices = denom != 0
    # Perform element-wise division, handling zeros in the denominator
    result[non_zero_indices] = numer[non_zero_indices] / denom[non_zero_indices]
    return result

list_metric_names = ['delta_n',
                     'n_matched_preds',
                     'n_matched_truth',
                     'n_preds',
                     'n_truth',
                     'n_unmatched_preds',
                     'n_unmatched_truth',
                     'percentage_total_area_covered_preds',
                     'percentage_total_area_covered_truth',
                     'percentage_truth_area_covered'
                     ]

list_x_labels = ['Number Truth Boxes - Number Predicted',
                 'Number of matched prediction boxes',
                 'Number of matched truth boxes (TP)',
                 'Number of predicted boxes',
                 'Number of true boxes',
                 'Number of Truth Boxes Missed (FN)',
                 'Number of unmatched/fake prediction boxes (FP)',
                 'Percentage of calorimeter included in predicted boxes (%)',   
                 'Percentage of calorimeter included in truth boxes (%)',   
                 'Percentage of matched truth boxes covered by predictions (%)'   
                 ]

def make_box_metric_plots(
    folder_containing_lists,
    save_folder,
):
    save_loc = save_folder + "/boxes/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    
    for j in range(len(list_metric_names)):
        event_level_metric = load_object(folder_containing_lists + "/{}.pkl".format(list_metric_names[j]))
        itle = f"{len(event_level_metric)} events"
        # if is_nested_list(event_level_metric):
        #     event_level_metric = np.concatenate(event_level_metric)
        #     itle = f"{len(event_level_metric)} boxes"

        label = "{:.2f}$\pm${:.1f}".format(np.mean(event_level_metric),np.std(event_level_metric))
        fig, ax = plt.subplots()
        ax.hist(event_level_metric,density=True,histtype='step',bins=50,label=label) 
        ax.set(xlabel=list_x_labels[j],ylabel='Freq. Density',title=itle)
        ax.grid(True)
        ax.legend()
        fig.savefig(save_loc+"{}.png".format(list_metric_names[j]))
        plt.close() 



    comp_save_loc = save_loc + "comp"
    if not os.path.exists(comp_save_loc):
        os.makedirs(comp_save_loc)
    #####################################################################################################################################
    #Plot 1, the number of boxes predicted and true
    n_truth = load_object(folder_containing_lists + '/n_truth.pkl')
    n_preds = load_object(folder_containing_lists + '/n_preds.pkl')

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    freq_tru, bins, _ = ax[0].hist(n_truth,bins=50,range=(0,max(n_truth)),histtype='step',color='green',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(n_truth),np.std(n_truth)))
    freq_pred, _, _ = ax[0].hist(n_preds,bins=bins,histtype='step',color='red',label='Predictions {:.2f}$\pm${:.1f}'.format(np.mean(n_preds),np.std(n_preds)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='{} Events'.format(len(n_truth)),yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Total Number of Boxes (per event)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(comp_save_loc + '/total_n_boxes.png')
    plt.close()

    #####################################################################################################################################
    #Plot 2, the number of matched predicted and true boxes
    n_unmatch_truth = load_object(folder_containing_lists + '/n_unmatched_truth.pkl')
    n_unmatch_preds = load_object(folder_containing_lists + '/n_unmatched_preds.pkl')

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    freq_tru, bins, _ = ax[0].hist(n_unmatch_truth,bins=50,range=(0,max(n_unmatch_truth)),histtype='step',color='green',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(n_unmatch_truth),np.std(n_unmatch_truth)))
    freq_pred, _, _ = ax[0].hist(n_unmatch_preds,bins=bins,histtype='step',color='red',label='Predictions {:.2f}$\pm${:.1f}'.format(np.mean(n_unmatch_preds),np.std(n_unmatch_preds)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='{} Events'.format(len(n_unmatch_truth)),yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Number of Unmatched Boxes (per event)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(comp_save_loc + '/unmatched_n_boxes.png')
    plt.close()

    #####################################################################################################################################
    #Plot 3, the number of unmatched predicted and true boxes
    n_match_truth = load_object(folder_containing_lists + '/n_matched_truth.pkl')
    n_match_preds = load_object(folder_containing_lists + '/n_matched_preds.pkl')

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    freq_tru, bins, _ = ax[0].hist(n_match_truth,bins=50,range=(0,max(n_match_truth)),histtype='step',color='green',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(n_match_truth),np.std(n_match_truth)))
    freq_pred, _, _ = ax[0].hist(n_match_preds,bins=bins,histtype='step',color='red',label='Predictions {:.2f}$\pm${:.1f}'.format(np.mean(n_match_preds),np.std(n_match_preds)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='{} Events'.format(len(n_match_truth)),yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Number of Matched Boxes (per event)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(comp_save_loc + '/matched_n_boxes.png')
    plt.close()







    print('Finished making metric plots.')
    return 




#make a plot of the number of truth boxes, number predicted boxes
#jaccard index


folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD_50k5_mu_20e/box_metrics/"
save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_plots/SSD_50k5_mu_20e/"
if __name__=="__main__":
    print('Making plots about boxes')
    make_box_metric_plots(folder_to_look_in,save_at)
    print('Completed plots about boxes\n')