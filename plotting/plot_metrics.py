print('Plotting script. WARNING: THIS SCRIPT IS OLD AND NEEDS UPDATING')
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


save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD_model_25_large_mu/"
if not os.path.exists(save_loc):
    os.makedirs(save_loc)

# files = ["/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp1_SSD_model_15_real/20230526-05/",
#          "/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp1EM_SSD_model_15_real/20230526-05/",
#          "/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp1HAD_SSD_model_15_real/20230526-05/",
#          "/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp2_SSD_model_15_real/20230526-05/",
#          "/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp3_SSD_model_15_real/20230526-05/"]
# names = ["combined", "EM", "HAD", "EM+HAD", "EM+HAD+combi"]
files = ["/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp3_SSD_model_15_real/20230526-05/",
         "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD_model_15_real_PU/20230821-12/",
         "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD_model_20_real_PU/20230831-05/",
         "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD_model_25_large_mu/20230908-12/"]
names = ["$\mu$=0","$\mu$=32, 3","$\mu$=32, 8","$\mu$=32, 8, 50k" ]

f1,ax1 = plt.subplots()
f2,ax2 = plt.subplots()
f3,ax3 = plt.subplots()
f4,ax4 = plt.subplots()
f5,ax5 = plt.subplots()
f6,ax6 = plt.subplots()
f7,ax7 = plt.subplots()
f8,ax8 = plt.subplots()
f9,ax9 = plt.subplots()
f10,ax10 = plt.subplots()
f11,ax11= plt.subplots()
f12,ax12 = plt.subplots()
f13,ax13 = plt.subplots()
f14,ax14 = plt.subplots()
f15,ax15= plt.subplots()

for file,name in zip(files,names):
    print(name)
    total_delta_n = load_object(file + "/total_delta_n.pkl")
    total_n_unmatch_truth = load_object(file + "/total_n_unmatch_truth.pkl")
    total_n_unmatch_pred = load_object(file + "/total_n_unmatch_pred.pkl")
    total_centre_diff = np.concatenate(load_object(file + "/total_centre_diff.pkl"))
    total_h_diff = np.concatenate(load_object(file + "/total_h_diff.pkl"))
    total_w_diff = np.concatenate(load_object(file + "/total_w_diff.pkl"))
    total_area_cov = load_object(file + "/total_area_cov.pkl")

    total_centre_diff_filter = np.concatenate(load_object(file + "/total_centre_diff_filter.pkl"))
    total_h_diff_filter = np.concatenate(load_object(file + "/total_h_diff_filter.pkl"))
    total_w_diff_filter = np.concatenate(load_object(file + "/total_w_diff_filter.pkl"))
    total_area_cov_filter = load_object(file + "/total_area_cov_filter.pkl")

    total_centre_diff_filter_rad = np.concatenate(load_object(file + "/total_centre_diff_filter_rad.pkl"))
    total_h_diff_filter_rad = np.concatenate(load_object(file + "/total_h_diff_filter_rad.pkl"))
    total_w_diff_filter_rad = np.concatenate(load_object(file + "/total_w_diff_filter_rad.pkl"))
    total_area_cov_filter_rad = load_object(file + "/total_area_cov_filter_rad.pkl")

    ax1.hist(total_delta_n,histtype='step',density=True,bins=np.arange(-5,20,step=1),label=name+f" {len(total_delta_n)} events")
    ax2.hist(total_n_unmatch_truth,histtype='step',density=True,bins=np.arange(0,20,step=1),label=name+f" {len(total_n_unmatch_truth)} events")
    ax3.hist(total_n_unmatch_pred,histtype='step',density=True,bins=np.arange(0,20,step=1),label=name+f" {len(total_n_unmatch_pred)} events")
    ax4.hist(total_area_cov,histtype='step',density=True,bins=np.arange(0,1,step=0.025),label=name+f" {len(total_area_cov)} events")
    ax5.hist(total_centre_diff,histtype='step',density=True,bins=np.arange(0,10,step=0.2),label=name+f" {len(total_centre_diff)} clusters")
    ax6.hist(total_h_diff,histtype='step',density=True,bins=np.arange(-2,2,step=0.1),label=name+f" {len(total_h_diff)} clusters")
    ax7.hist(total_w_diff,histtype='step',density=True,bins=np.arange(-2,2,step=0.1),label=name+f" {len(total_w_diff)} clusters")

    ax8.hist(total_area_cov_filter,histtype='step',density=True,bins=np.arange(0,1,step=0.025),label=name+f" {len(total_area_cov_filter)} events")
    ax9.hist(total_centre_diff_filter,histtype='step',density=True,bins=np.arange(0,10,step=0.2),label=name+f" {len(total_centre_diff_filter)} clusters")
    ax10.hist(total_h_diff_filter,histtype='step',density=True,bins=np.arange(-2,2,step=0.1),label=name+f" {len(total_h_diff_filter)} clusters")
    ax11.hist(total_w_diff_filter,histtype='step',density=True,bins=np.arange(-2,2,step=0.1),label=name+f" {len(total_w_diff_filter)} clusters")

    ax12.hist(total_area_cov_filter_rad,histtype='step',density=True,bins=np.arange(0,1,step=0.025),label=name+f" {len(total_area_cov_filter_rad)} events")
    ax13.hist(total_centre_diff_filter_rad,histtype='step',density=True,bins=np.arange(0,10,step=0.2),label=name+f" {len(total_centre_diff_filter_rad)} clusters")
    ax14.hist(total_h_diff_filter_rad,histtype='step',density=True,bins=np.arange(-2,2,step=0.1),label=name+f" {len(total_h_diff_filter_rad)} clusters")
    ax15.hist(total_w_diff_filter_rad,histtype='step',density=True,bins=np.arange(-2,2,step=0.1),label=name+f" {len(total_w_diff_filter_rad)} clusters")


ax1.set(xlabel='total_delta_n')
ax2.set(xlabel='total_n_unmatch truth')
ax3.set(xlabel='total_n_unmatch predictions')
ax4.set(xlabel='total_area_cov')
ax5.set(xlabel='total_centre_diff',title='{} Objects in {} images'.format(len(total_centre_diff),len(total_delta_n)))#,yscale='log'
ax6.set(xlabel='total_h_diff',title='{} Objects in {} images'.format(len(total_h_diff),len(total_delta_n)))
ax7.set(xlabel='total_w_diff',title='{} Objects in {} images'.format(len(total_w_diff),len(total_delta_n)))

ax8.set(xlabel='total_area_cov_filter')
ax9.set(xlabel='total_centre_diff_filter',title='{} Filtered Objects in {} images'.format(len(total_centre_diff_filter),len(total_delta_n)))#,yscale='log'
ax10.set(xlabel='total_h_diff_filter',title='{} Filtered Objects in {} images'.format(len(total_h_diff_filter),len(total_delta_n)))
ax11.set(xlabel='total_w_diff_filter',title='{} FilteredObjects in {} images'.format(len(total_w_diff_filter),len(total_delta_n)))

ax12.set(xlabel='total_area_cov_filter')
ax13.set(xlabel='total_centre_diff_filter_rad',title='{} Filtered (0.4) Objects in {} images'.format(len(total_centre_diff_filter_rad),len(total_delta_n)))#,yscale='log'
ax14.set(xlabel='total_h_diff_filter_rad',title='{} Filtered (0.4) Objects in {} images'.format(len(total_h_diff_filter_rad),len(total_delta_n)))
ax15.set(xlabel='total_w_diff_filter_rad',title='{} Filtered (0.4) Objects in {} images'.format(len(total_w_diff_filter_rad),len(total_delta_n)))



ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
ax6.legend()
ax7.legend()
ax8.legend()
ax9.legend()
ax10.legend()
ax11.legend()
ax12.legend()
ax13.legend()
ax14.legend()
ax15.legend()

f1.savefig(save_loc + 'total_delta_n.png')
f2.savefig(save_loc + 'total_n_unmatch_truth.png')
f3.savefig(save_loc + 'total_n_unmatch_pred.png')
f4.savefig(save_loc + 'total_area_cov.png')
f5.savefig(save_loc + 'total_centre_diff.png')
f6.savefig(save_loc + 'total_h_diff.png')
f7.savefig(save_loc + 'total_w_diff.png')
f8.savefig(save_loc + 'total_area_cov_filter.png')
f9.savefig(save_loc + 'total_centre_diff_filter.png')
f10.savefig(save_loc + 'total_h_diff_filter.png')
f11.savefig(save_loc + 'total_w_diff_filter.png')
f12.savefig(save_loc + 'total_area_cov_filter_rad.png')
f13.savefig(save_loc + 'total_centre_diff_filter_rad.png')
f14.savefig(save_loc + 'total_h_diff_filter_rad.png')
f15.savefig(save_loc + 'total_w_diff_filter_rad.png')

