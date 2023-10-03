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

list_metric_names = ['total_delta_n',
                     'total_n_unmatch_truth',
                     'total_n_unmatch_pred',
                     'total_area_cov',
                     'total_centre_diff']

list_x_labels = ['Number Truth Boxes - Number Predicted',
                 'Number Truth Boxes Missed',
                 'Number of fake predictions',
                 'Percentage of Matched Truth Boxes Area Covered',
                 'L2 Distance between Truth and Prediction Box Centres']


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
    print('Finished making metric plots.')
    return 




#make a plot of the number of truth boxes, number predicted boxes
