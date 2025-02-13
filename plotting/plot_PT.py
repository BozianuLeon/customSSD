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
# metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/ttbar/20250124-12/box_metrics"
# save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/ttbar/20250124-12/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

square_comp = True
image_format = "png"

print("=======================================================================================================")
print(f"Loading all jets from\n{metrics_folder}")
print("=======================================================================================================\n")


total_t_pt      = np.concatenate(load_object(metrics_folder+"/tboxes_pt.pkl"))
total_p_pt      = np.concatenate(load_object(metrics_folder+"/pboxes_pt.pkl"))
total_p_scr     = np.concatenate(load_object(metrics_folder+"/pboxes_scores.pkl"))
# IOU matched
match_t_pt      = np.concatenate(load_object(metrics_folder+"/tboxes_matched_pt.pkl"))
match_p_pt      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_pt.pkl"))
match_p_scr     = np.concatenate(load_object(metrics_folder+"/pboxes_matched_scr.pkl"))
unmatch_t_pt    = np.concatenate(load_object(metrics_folder+"/tboxes_unmatched_pt.pkl"))
unmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_pt.pkl"))
unmatch_p_scr   = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_scr.pkl"))
# dR matched
dRmatch_t_pt    = np.concatenate(load_object(metrics_folder+"/tboxes_dRmatched_pt.pkl"))
dRmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_pt.pkl"))
dRmatch_p_scr   = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_scr.pkl"))
dRunmatch_t_pt  = np.concatenate(load_object(metrics_folder+"/tboxes_dRunmatched_pt.pkl"))
dRunmatch_p_pt  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_pt.pkl"))
dRunmatch_p_scr = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_scr.pkl"))

print("Implementing a temporary, post-hoc confidence threshold:")
print(f"Number        total targets: {len(total_t_pt)}\nNumber        total predictions: {len(total_p_pt)}")
print(f"Number      matched targets: {len(match_t_pt)}\nNumber      matched predictions: {len(match_p_pt)}")
print(f"Number    unmatched targets: {len(unmatch_t_pt)}\nNumber    unmatched predictions: {len(unmatch_p_pt)}")
print(f"Number dR   matched targets: {len(dRmatch_t_pt)}\nNumber dR   matched predictions: {len(dRmatch_p_pt)}")
print(f"Number dR unmatched targets: {len(dRunmatch_t_pt)}\nNumber dR unmatched predictions: {len(dRunmatch_p_pt)}")

scr_threshold = 0.5
total_scr_mask = total_p_scr > scr_threshold
total_p_pt = total_p_pt[total_scr_mask]

match_scr_mask = match_p_scr > scr_threshold
match_t_pt = match_t_pt[match_scr_mask]
match_p_pt = match_p_pt[match_scr_mask]

dRmatch_scr_mask = dRmatch_p_scr > scr_threshold
dRmatch_t_pt = dRmatch_t_pt[dRmatch_scr_mask]
dRmatch_p_pt = dRmatch_p_pt[dRmatch_scr_mask]

unmatch_scr_mask = unmatch_p_scr > scr_threshold
unmatch_p_pt = unmatch_p_pt[unmatch_scr_mask]

dRunmatch_scr_mask = dRunmatch_p_scr > scr_threshold
dRunmatch_p_pt = dRunmatch_p_pt[dRunmatch_scr_mask]

print("\n\nNew score threshold applied")
print(f"Number        total targets: {len(total_t_pt)}\nNumber        total predictions: {len(total_p_pt)}")
print(f"Number      matched targets: {len(match_t_pt)}\nNumber      matched predictions: {len(match_p_pt)}")
print(f"Number    unmatched targets: {len(unmatch_t_pt)}\nNumber    unmatched predictions: {len(unmatch_p_pt)}")
print(f"Number dR   matched targets: {len(dRmatch_t_pt)}\nNumber dR   matched predictions: {len(dRmatch_p_pt)}")
print(f"Number dR unmatched targets: {len(dRunmatch_t_pt)}\nNumber dR unmatched predictions: {len(dRunmatch_p_pt)}")


################################################################
print("=======================================================================================================")
print(f"Plotting jet pT, saving to {save_folder}")
print("=======================================================================================================\n")


print(f"Plotting total jet pT: {len(total_p_pt)} predictions, {len(total_t_pt)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(total_t_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.set_title('Transverse Momentum', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
f.savefig(save_folder + f'/jet_pt_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


print(f"Plotting matched jet pT: {len(match_p_pt)} predictions, {len(match_t_pt)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(match_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(match_t_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.set_title('Transverse Momentum IoU Matched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
f.savefig(save_folder + f'/jet_pt_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

print(f"Plotting dR matched jet pT: {len(dRmatch_p_pt)} predictions, {len(dRmatch_t_pt)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(dRmatch_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(dRmatch_t_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.set_title('Transverse Momentum deltaR Matched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
f.savefig(save_folder + f'/jet_pt_dRmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


print(f"Plotting unmatched jet pT: {len(unmatch_p_pt)} predictions, {len(unmatch_t_pt)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(unmatch_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(unmatch_t_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.set_title('Transverse Momentum IoU Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
f.savefig(save_folder + f'/jet_pt_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


print(f"Plotting dR unmatched jet pT: {len(dRunmatch_p_pt)} predictions, {len(dRunmatch_t_pt)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(dRunmatch_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tru, bins, _    = ax0.hist(dRunmatch_t_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.set_title('Transverse Momentum deltaR Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
f.savefig(save_folder + f'/jet_pt_dRunmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()









if square_comp:

    # square metrics folder
    square_model_name = "jetSSD_sq_uconvnext_central_11e"
    square_date       = "20250207-13"
    square_metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{square_model_name}/{proc}/{square_date}/box_metrics"

    print("\n=======================================================================================================")
    print(f"Getting inference and metrics from a model trained with square(!) sum-pool layer")
    print(f"Loading all jets from\n{square_metrics_folder}")
    print("=======================================================================================================\n")

    total_sq_t_pt = np.concatenate(load_object(square_metrics_folder+"/tboxes_pt.pkl"))
    total_sq_p_pt = np.concatenate(load_object(square_metrics_folder+"/pboxes_pt.pkl"))
    total_sq_p_scr = np.concatenate(load_object(square_metrics_folder+"/pboxes_scores.pkl"))
    # IOU matched
    match_sq_t_pt = np.concatenate(load_object(square_metrics_folder+"/tboxes_matched_pt.pkl"))
    match_sq_p_pt = np.concatenate(load_object(square_metrics_folder+"/pboxes_matched_pt.pkl"))
    unmatch_sq_t_pt = np.concatenate(load_object(square_metrics_folder+"/tboxes_unmatched_pt.pkl"))
    unmatch_sq_p_pt = np.concatenate(load_object(square_metrics_folder+"/pboxes_unmatched_pt.pkl"))
    # dR matched
    dRmatch_sq_t_pt = np.concatenate(load_object(square_metrics_folder+"/tboxes_dRmatched_pt.pkl"))
    dRmatch_sq_p_pt = np.concatenate(load_object(square_metrics_folder+"/pboxes_dRmatched_pt.pkl"))
    dRunmatch_sq_t_pt = np.concatenate(load_object(square_metrics_folder+"/tboxes_dRunmatched_pt.pkl"))
    dRunmatch_sq_p_pt = np.concatenate(load_object(square_metrics_folder+"/pboxes_dRunmatched_pt.pkl"))


    print(f"Plotting comparison of ALL jets")
    print(f"{len(total_p_pt)} di predictions, {len(total_t_pt)} di targets, {len(total_sq_p_pt)} sq predictions, {len(total_sq_t_pt)} sq targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(total_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets (Di)')
    freq_tru, bins, _    = ax0.hist(total_t_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets (Di)')
    freq_tru, bins, _    = ax0.hist(total_sq_t_pt,bins=bins,histtype='step',ls='--',color='green',lw=1.5,label='Target Jets (Sq)')
    freq_tru, bins, _    = ax0.hist(total_sq_p_pt,bins=bins,histtype='step',ls='--',color='red',lw=1.5,label='Pred Jets (Sq)')
    ax0.set_title('Transverse Momentum', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
    f.savefig(save_folder + f'/comp_jet_pt_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    print(f"Plotting comparison of IOU MATCHED jets")
    print(f"{len(match_p_pt)} matched di predictions, {len(match_t_pt)} matched di targets, {len(match_sq_t_pt)} matched sq predictions, {len(match_sq_p_pt)} matched sq targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(match_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets (Di)')
    freq_tru, bins, _    = ax0.hist(match_t_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets (Di)')
    freq_tru, bins, _    = ax0.hist(match_sq_t_pt,bins=bins,histtype='step',ls='--',color='green',lw=1.5,label='Target Jets (Sq)')
    freq_tru, bins, _    = ax0.hist(match_sq_p_pt,bins=bins,histtype='step',ls='--',color='red',lw=1.5,label='Pred Jets (Sq)')
    ax0.set_title('Transverse Momentum IoU Matched', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
    f.savefig(save_folder + f'/comp_jet_pt_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    print(f"Plotting comparison of dR MATCHED jets")
    print(f"{len(dRmatch_p_pt)} dR matched di predictions, {len(dRmatch_t_pt)} dR matched di targets, {len(dRmatch_sq_t_pt)} dR matched sq predictions, {len(dRmatch_sq_p_pt)} dR matched sq targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(dRmatch_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets (Di)')
    freq_tru, bins, _    = ax0.hist(dRmatch_t_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets (Di)')
    freq_tru, bins, _    = ax0.hist(dRmatch_sq_t_pt,bins=bins,histtype='step',ls='--',color='green',lw=1.5,label='Target Jets (Sq)')
    freq_tru, bins, _    = ax0.hist(dRmatch_sq_p_pt,bins=bins,histtype='step',ls='--',color='red',lw=1.5,label='Pred Jets (Sq)')
    ax0.set_title('Transverse Momentum deltaR Matched', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
    f.savefig(save_folder + f'/comp_jet_pt_dRmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    print(f"Plotting comparison of IOU UNMATCHED jets")
    print(f"{len(unmatch_p_pt)} unmatched di predictions, {len(unmatch_t_pt)} unmatched di targets, {len(unmatch_sq_t_pt)} unmatched sq predictions, {len(unmatch_sq_p_pt)} unmatched sq targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(unmatch_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets (Di)')
    freq_tru, bins, _    = ax0.hist(unmatch_t_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets (Di)')
    freq_tru, bins, _    = ax0.hist(unmatch_sq_t_pt,bins=bins,histtype='step',ls='--',color='green',lw=1.5,label='Target Jets (Sq)')
    freq_tru, bins, _    = ax0.hist(unmatch_sq_p_pt,bins=bins,histtype='step',ls='--',color='red',lw=1.5,label='Pred Jets (Sq)')
    ax0.set_title('Transverse Momentum IoU Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
    f.savefig(save_folder + f'/comp_jet_pt_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    print(f"Plotting comparison of dR UNMATCHED jets")
    print(f"{len(unmatch_p_pt)} dR unmatched di predictions, {len(unmatch_t_pt)} dR unmatched di targets, {len(dRunmatch_sq_t_pt)} dR unmatched sq predictions, {len(dRunmatch_sq_p_pt)} dR unmatched sq targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(dRunmatch_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets (Di)')
    freq_tru, bins, _    = ax0.hist(dRunmatch_t_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets (Di)')
    freq_tru, bins, _    = ax0.hist(dRunmatch_sq_t_pt,bins=bins,histtype='step',ls='--',color='green',lw=1.5,label='Target Jets (Sq)')
    freq_tru, bins, _    = ax0.hist(dRunmatch_sq_p_pt,bins=bins,histtype='step',ls='--',color='red',lw=1.5,label='Pred Jets (Sq)')
    ax0.set_title('Transverse Momentum deltaR Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
    f.savefig(save_folder + f'/comp_jet_pt_dRunmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


 


