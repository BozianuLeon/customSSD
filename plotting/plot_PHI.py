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
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/{proc}/{date}/jet_kin/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

image_format = "png"

print("=======================================================================================================")
print(f"Loading all jets from\n{metrics_folder}")
print("=======================================================================================================\n")

# total_t_pt      = np.concatenate(load_object(metrics_folder+"/tboxes_pt.pkl"))
# total_p_pt      = np.concatenate(load_object(metrics_folder+"/pboxes_pt.pkl"))
total_t_phi      = np.concatenate(load_object(metrics_folder+"/tboxes_phi.pkl"))
total_p_phi      = np.concatenate(load_object(metrics_folder+"/pboxes_phi.pkl"))
total_p_scr      = np.concatenate(load_object(metrics_folder+"/pboxes_scores.pkl"))
# IOU matched
# match_t_pt      = np.concatenate(load_object(metrics_folder+"/tboxes_matched_pt.pkl"))
# match_p_pt      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_pt.pkl"))
match_t_phi      = np.concatenate(load_object(metrics_folder+"/tboxes_matched_phi.pkl"))
match_p_phi      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_phi.pkl"))
match_p_scr      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_scr.pkl"))
# unmatch_t_pt    = np.concatenate(load_object(metrics_folder+"/tboxes_unmatched_pt.pkl"))
# unmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_pt.pkl"))
unmatch_t_phi    = np.concatenate(load_object(metrics_folder+"/tboxes_unmatched_phi.pkl"))
unmatch_p_phi    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_phi.pkl"))
unmatch_p_scr    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_scr.pkl"))
# dR matched
# dRmatch_t_pt    = np.concatenate(load_object(metrics_folder+"/tboxes_dRmatched_pt.pkl"))
# dRmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_pt.pkl"))
dRmatch_t_phi    = np.concatenate(load_object(metrics_folder+"/tboxes_dRmatched_phi.pkl"))
dRmatch_p_phi    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_phi.pkl"))
dRmatch_p_scr    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_scr.pkl"))
# dRunmatch_t_pt  = np.concatenate(load_object(metrics_folder+"/tboxes_dRunmatched_pt.pkl"))
# dRunmatch_p_pt  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_pt.pkl"))
dRunmatch_t_phi  = np.concatenate(load_object(metrics_folder+"/tboxes_dRunmatched_phi.pkl"))
dRunmatch_p_phi  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_phi.pkl"))
dRunmatch_p_scr  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_scr.pkl"))

print("Implementing a temporary, post-hoc confidence threshold:")
print(f"Number        total targets: {len(total_t_phi)}\nNumber        total predictions: {len(total_p_phi)}")
print(f"Number      matched targets: {len(match_t_phi)}\nNumber      matched predictions: {len(match_p_phi)}")
print(f"Number    unmatched targets: {len(unmatch_t_phi)}\nNumber    unmatched predictions: {len(unmatch_p_phi)}")
print(f"Number dR   matched targets: {len(dRmatch_t_phi)}\nNumber dR   matched predictions: {len(dRmatch_p_phi)}")
print(f"Number dR unmatched targets: {len(dRunmatch_t_phi)}\nNumber dR unmatched predictions: {len(dRunmatch_p_phi)}")

scr_threshold = 0.5
total_scr_mask = total_p_scr > scr_threshold
total_p_phi = total_p_phi[total_scr_mask]

match_scr_mask = match_p_scr > scr_threshold
match_t_phi = match_t_phi[match_scr_mask]
match_p_phi = match_p_phi[match_scr_mask]

dRmatch_scr_mask = dRmatch_p_scr > scr_threshold
dRmatch_t_phi = dRmatch_t_phi[dRmatch_scr_mask]
dRmatch_p_phi = dRmatch_p_phi[dRmatch_scr_mask]

unmatch_scr_mask = unmatch_p_scr > scr_threshold
unmatch_p_phi = unmatch_p_phi[unmatch_scr_mask]

dRunmatch_scr_mask = dRunmatch_p_scr > scr_threshold
dRunmatch_p_phi = dRunmatch_p_phi[dRunmatch_scr_mask]

print("\n\nNew score threshold applied")
print(f"Number        total targets: {len(total_t_phi)}\nNumber        total predictions: {len(total_p_phi)}")
print(f"Number      matched targets: {len(match_t_phi)}\nNumber      matched predictions: {len(match_p_phi)}")
print(f"Number    unmatched targets: {len(unmatch_t_phi)}\nNumber    unmatched predictions: {len(unmatch_p_phi)}")
print(f"Number dR   matched targets: {len(dRmatch_t_phi)}\nNumber dR   matched predictions: {len(dRmatch_p_phi)}")
print(f"Number dR unmatched targets: {len(dRunmatch_t_phi)}\nNumber dR unmatched predictions: {len(dRunmatch_p_phi)}")


################################################################
print("=======================================================================================================")
print(f"Plotting jet phi, saving to {save_folder}")
print("=======================================================================================================\n")


################################################################

print(f"Plotting total jet phi: {len(total_p_phi)} predictions, {len(total_t_phi)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_p_phi,bins=50,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(total_t_phi,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.set_title('Pseudorapidity Total', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $\phi$')
f.savefig(save_folder + f'/jet_phi_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


print(f"Plotting matched jet phi: {len(match_p_phi)} predictions, {len(match_t_phi)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(match_p_phi,bins=50,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(match_t_phi,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.set_title('Pseudorapidity IoU Matched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $\phi$')
f.savefig(save_folder + f'/jet_phi_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

print(f"Plotting dR matched jet phi: {len(dRmatch_p_phi)} predictions, {len(dRmatch_t_phi)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(dRmatch_p_phi,bins=50,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(dRmatch_t_phi,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.set_title('Pseudorapidity deltaR Matched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $\phi$')
f.savefig(save_folder + f'/jet_phi_dRmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


print(f"Plotting unmatched jet phi: {len(unmatch_p_phi)} predictions, {len(unmatch_t_phi)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(unmatch_p_phi,bins=50,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(unmatch_t_phi,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.set_title('Pseudorapidity IoU Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $\phi$')
f.savefig(save_folder + f'/jet_phi_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


print(f"Plotting dR unmatched jet phi: {len(dRunmatch_p_phi)} predictions, {len(dRunmatch_t_phi)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(dRunmatch_p_phi,bins=50,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(dRunmatch_t_phi,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.set_title('Pseudorapidity deltaR Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $\phi$')
f.savefig(save_folder + f'/jet_phi_dRunmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()





