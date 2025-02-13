import numpy as np 
import scipy
import math
import os

import itertools
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
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/{proc}/{date}/2D/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

image_format = "png"



print("=======================================================================================================")
print(f"Loading all jets from\n{metrics_folder}")
print("=======================================================================================================\n")

n_test_events = len(load_object(metrics_folder+"/tboxes_pt.pkl"))
print("N test events:", n_test_events, len(load_object(metrics_folder+"/tboxes_matched_pt.pkl")), len(load_object(metrics_folder+"/tboxes_dRmatched_pt.pkl")))
total_t_pt      = np.concatenate(load_object(metrics_folder+"/tboxes_pt.pkl"))
total_p_pt      = np.concatenate(load_object(metrics_folder+"/pboxes_pt.pkl"))
total_t_eta      = np.concatenate(load_object(metrics_folder+"/tboxes_eta.pkl"))
total_p_eta      = np.concatenate(load_object(metrics_folder+"/pboxes_eta.pkl"))
total_t_phi      = np.concatenate(load_object(metrics_folder+"/tboxes_phi.pkl"))
total_p_phi      = np.concatenate(load_object(metrics_folder+"/pboxes_phi.pkl"))
total_p_scr      = np.concatenate(load_object(metrics_folder+"/pboxes_scores.pkl"))

# IOU matched
match_t_pt      = np.concatenate(load_object(metrics_folder+"/tboxes_matched_pt.pkl"))
match_p_pt      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_pt.pkl"))
match_t_eta      = np.concatenate(load_object(metrics_folder+"/tboxes_matched_eta.pkl"))
match_p_eta      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_eta.pkl"))
match_t_phi      = np.concatenate(load_object(metrics_folder+"/tboxes_matched_phi.pkl"))
match_p_phi      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_phi.pkl"))
match_p_scr      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_scr.pkl"))
unmatch_t_pt    = np.concatenate(load_object(metrics_folder+"/tboxes_unmatched_pt.pkl"))
unmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_pt.pkl"))
unmatch_t_eta    = np.concatenate(load_object(metrics_folder+"/tboxes_unmatched_eta.pkl"))
unmatch_p_eta    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_eta.pkl"))
unmatch_t_phi    = np.concatenate(load_object(metrics_folder+"/tboxes_unmatched_phi.pkl"))
unmatch_p_phi    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_phi.pkl"))
unmatch_p_scr    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_scr.pkl"))

# dR matched
dRmatch_t_pt    = np.concatenate(load_object(metrics_folder+"/tboxes_dRmatched_pt.pkl"))
dRmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_pt.pkl"))
dRmatch_t_eta    = np.concatenate(load_object(metrics_folder+"/tboxes_dRmatched_eta.pkl"))
dRmatch_p_eta    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_eta.pkl"))
dRmatch_t_phi    = np.concatenate(load_object(metrics_folder+"/tboxes_dRmatched_phi.pkl"))
dRmatch_p_phi    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_phi.pkl"))
dRmatch_p_scr    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_scr.pkl"))
dRunmatch_t_pt  = np.concatenate(load_object(metrics_folder+"/tboxes_dRunmatched_pt.pkl"))
dRunmatch_p_pt  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_pt.pkl"))
dRunmatch_t_eta  = np.concatenate(load_object(metrics_folder+"/tboxes_dRunmatched_eta.pkl"))
dRunmatch_p_eta  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_eta.pkl"))
dRunmatch_t_phi  = np.concatenate(load_object(metrics_folder+"/tboxes_dRunmatched_phi.pkl"))
dRunmatch_p_phi  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_phi.pkl"))
dRunmatch_p_scr  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_scr.pkl"))



print("=======================================================================================================")
print(f"Making 2D plots and saving in \n{save_folder}")
print("=======================================================================================================\n")


# 2d plots of all/matched/unmatched

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -2.5, 2.5
# MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496

bins_x = np.linspace(MIN_CELLS_ETA, MAX_CELLS_ETA, (int((MAX_CELLS_ETA - MIN_CELLS_ETA) / (0.1)) + 1))
bins_y = np.linspace(MIN_CELLS_PHI, MAX_CELLS_PHI, (int((MAX_CELLS_PHI - MIN_CELLS_PHI) / ((2*np.pi)/64)) + 1))
extent = (MIN_CELLS_ETA,MAX_CELLS_ETA,MIN_CELLS_PHI,MAX_CELLS_PHI)



H_tot_tru,_,_ = np.histogram2d(total_t_eta,total_t_phi,bins=(bins_x,bins_y),weights=np.ones_like(total_t_eta)/n_test_events)
H_tot_tru = H_tot_tru.T
f,ax = plt.subplots()
cmap_t = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","green","lime"])
ii = ax.imshow(H_tot_tru,extent=extent,cmap=cmap_t)
cbar = f.colorbar(ii,ax=ax)
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Number of target jets per event', rotation=90)
cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
# ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
ax.set(xlabel=f"eta",ylabel=f"phi")
f.savefig(save_folder + f'/2d_t.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()



H_tot_pre,_,_ = np.histogram2d(total_p_eta,total_p_phi,bins=(bins_x,bins_y),weights=np.ones_like(total_p_eta)/n_test_events)
H_tot_pre = H_tot_pre.T
f,ax = plt.subplots()
cmap_p = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","blue","violet","red"])
ii = ax.imshow(H_tot_pre,extent=extent,cmap=cmap_p)
cbar = f.colorbar(ii,ax=ax)
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Num. jet predicted per event', rotation=90)
cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
# ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
ax.set(xlabel=f"eta",ylabel=f"phi")
f.savefig(save_folder + f'/2d_p.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()




print("=======================================================================================================")
print(f"Making 2D matched plots")
print("=======================================================================================================")

H_match_tot_tru, _, _ = np.histogram2d(match_t_eta,match_t_phi,bins=(bins_x,bins_y),weights=np.ones_like(match_t_eta)/n_test_events)
H_match_tot_tru = H_match_tot_tru.T
f,ax = plt.subplots()
ii = ax.imshow(H_match_tot_tru,extent=extent,cmap=cmap_t)
cbar = f.colorbar(ii,ax=ax)
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Number matched target jets per event', rotation=90)
cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
# ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
ax.set(xlabel=f"eta",ylabel=f"phi")
f.savefig(save_folder + f'/2d_match_t.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


H_match_tot_pre,_,_ = np.histogram2d(match_p_eta,match_p_phi,bins=(bins_x,bins_y),weights=np.ones_like(match_p_eta)/n_test_events)
H_match_tot_pre = H_match_tot_pre.T
f,ax = plt.subplots()
ii = ax.imshow(H_match_tot_pre,extent=extent,cmap=cmap_p)
cbar = f.colorbar(ii,ax=ax)
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Number matched jet predicted per event', rotation=90)
cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
# ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
ax.set(xlabel=f"eta",ylabel=f"phi")
f.savefig(save_folder + f'/2d_match_p.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()




print("=======================================================================================================")
print(f"Making 2D unmatched plots")
print("=======================================================================================================")


H_unmatch_tot_tru,_,_ = np.histogram2d(unmatch_t_eta,unmatch_t_phi,bins=(bins_x,bins_y),weights=np.ones_like(unmatch_t_eta)/n_test_events)
H_unmatch_tot_tru = H_unmatch_tot_tru.T
f,ax = plt.subplots()
ii = ax.imshow(H_unmatch_tot_tru,extent=extent,cmap=cmap_t)
cbar = f.colorbar(ii,ax=ax)
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Number unmatched target jets per event', rotation=90)
cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
# ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
ax.set(xlabel=f"eta",ylabel=f"phi")
f.savefig(save_folder + f'/2d_unmatch_t.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


H_unmatch_tot_pre,_,_ = np.histogram2d(unmatch_p_eta,unmatch_p_phi,bins=(bins_x,bins_y),weights=np.ones_like(unmatch_p_eta)/n_test_events)
H_unmatch_tot_pre = H_unmatch_tot_pre.T
f,ax = plt.subplots()
ii = ax.imshow(H_unmatch_tot_pre,extent=extent,cmap=cmap_p)
cbar = f.colorbar(ii,ax=ax)
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Number unmatched jet predicted per event', rotation=90)
cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
# ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
ax.set(xlabel=f"eta",ylabel=f"phi")
f.savefig(save_folder + f'/2d_unmatch_p.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()



