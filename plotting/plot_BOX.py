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
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/{proc}/{date}/box_stat/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

image_format = "png"

print("=======================================================================================================")
print(f"Loading box statistics from\n{metrics_folder}")
print("=======================================================================================================\n")
     
     


# Event-level vars
n_truth = load_object(metrics_folder+"/n_truth.pkl")    
n_preds = load_object(metrics_folder+"/n_preds.pkl")    
delta_n = load_object(metrics_folder+"/delta_n.pkl")    

n_matched_truth   = load_object(metrics_folder+"/n_matched_truth.pkl")    
n_unmatched_truth = load_object(metrics_folder+"/n_unmatched_truth.pkl")    
n_matched_preds   = load_object(metrics_folder+"/n_matched_preds.pkl")    
n_unmatched_preds = load_object(metrics_folder+"/n_unmatched_preds.pkl")    

n_dRmatched_truth   = load_object(metrics_folder+"/n_dRmatched_truth.pkl")    
n_dRunmatched_truth = load_object(metrics_folder+"/n_dRunmatched_truth.pkl")    
n_dRmatched_preds   = load_object(metrics_folder+"/n_dRmatched_preds.pkl")    
n_dRunmatched_preds = load_object(metrics_folder+"/n_dRunmatched_preds.pkl")    
 
 
# total_t_pt      = load_object(metrics_folder+"/tboxes_pt.pkl") #num_truth = [len(x) for x in total_t_pt]
# total_p_pt      = load_object(metrics_folder+"/pboxes_pt.pkl")
# total_p_scr     = load_object(metrics_folder+"/pboxes_scores.pkl")
# IOU matched
# match_t_pt      = load_object(metrics_folder+"/tboxes_matched_pt.pkl")
# match_p_pt      = load_object(metrics_folder+"/pboxes_matched_pt.pkl")
# match_p_scr     = load_object(metrics_folder+"/pboxes_matched_scr.pkl")
# unmatch_t_pt    = load_object(metrics_folder+"/tboxes_unmatched_pt.pkl")
# unmatch_p_pt    = load_object(metrics_folder+"/pboxes_unmatched_pt.pkl")
# unmatch_p_scr   = load_object(metrics_folder+"/pboxes_unmatched_scr.pkl")
# # dR matched
# dRmatch_t_pt    = load_object(metrics_folder+"/tboxes_dRmatched_pt.pkl")
# dRmatch_p_pt    = load_object(metrics_folder+"/pboxes_dRmatched_pt.pkl")
# dRmatch_p_scr   = load_object(metrics_folder+"/pboxes_dRmatched_scr.pkl")
# dRunmatch_t_pt  = load_object(metrics_folder+"/tboxes_dRunmatched_pt.pkl")
# dRunmatch_p_pt  = load_object(metrics_folder+"/pboxes_dRunmatched_pt.pkl")
# dRunmatch_p_scr = load_object(metrics_folder+"/pboxes_dRunmatched_scr.pkl")
 

print("=======================================================================================================")
print(f"Plotting box statistics, saving to {save_folder}")
print("=======================================================================================================\n")



print(f"Plotting total number of jets per event: avg {np.mean(n_preds):.3f} predictions, avg {np.mean(n_truth):.3f} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(n_preds,bins=max(n_preds),histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(n_truth,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
# ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
ax0.legend(loc='lower left',bbox_to_anchor=(0.6, 0.75),fontsize="small")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet multiplicity per event')
f.savefig(save_folder + f'/n_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

print(f"Plotting number of matched jets per event: avg {np.mean(n_matched_preds):.3f} predictions, avg {np.mean(n_matched_truth):.3f} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(n_matched_preds,bins=max(n_matched_preds),histtype='step',color='red',lw=1.5,ls='--',zorder=10,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(n_matched_truth,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
# ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
ax0.legend(loc='lower left',bbox_to_anchor=(0.6, 0.75),fontsize="small")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Matched jets per event')
f.savefig(save_folder + f'/n_match_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

print(f"Plotting number of unmatched jets per event: avg {np.mean(n_unmatched_preds):.3f} predictions, avg {np.mean(n_unmatched_truth):.3f} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(n_unmatched_preds,bins=max(n_unmatched_preds),histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(n_unmatched_truth,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
# ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
ax0.legend(loc='lower left',bbox_to_anchor=(0.6, 0.75),fontsize="small")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Unmatched jets per event')
f.savefig(save_folder + f'/n_unmatch_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

print(f"Plotting number of dR matched jets per event: avg {np.mean(n_dRmatched_preds):.3f} predictions, avg {np.mean(n_dRmatched_truth):.3f} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(n_dRmatched_preds,bins=max(n_dRmatched_preds),histtype='step',color='red',ls='--',zorder=10,lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(n_dRmatched_truth,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
# ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
ax0.legend(loc='lower left',bbox_to_anchor=(0.6, 0.75),fontsize="small")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='dR Matched jets per event')
f.savefig(save_folder + f'/n_dRmatch_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

print(f"Plotting number of dR unmatched jets per event: avg {np.mean(n_dRunmatched_preds):.3f} predictions, avg {np.mean(n_dRunmatched_truth):.3f} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(n_dRunmatched_preds,bins=max(n_dRunmatched_preds),histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(n_dRunmatched_truth,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
# ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
ax0.legend(loc='lower left',bbox_to_anchor=(0.6, 0.75),fontsize="small")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='dR Unmatched jets per event')
f.savefig(save_folder + f'/n_dRunmatch_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()






