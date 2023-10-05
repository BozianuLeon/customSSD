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
        label = f"{len(event_level_metric)} events"
        if is_nested_list(event_level_metric):
            event_level_metric = np.concatenate(event_level_metric)
            label = f"{len(event_level_metric)} boxes"

        fig, ax = plt.subplots()
        ax.hist(event_level_metric,density=True,histtype='step',bins=50,label=label) 
        ax.set(xlabel=list_x_labels[j],ylabel='Freq. Density')
        ax.grid(True)
        ax.legend()
        fig.savefig(save_loc+"{}.png".format(list_metric_names[j]))
        plt.close() 



    #####################################################################################################################################
    #Plot 1, the energies of our boxes and the true clusters
    #total:










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