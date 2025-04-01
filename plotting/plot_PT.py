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



model_name = "jetSSD_custom_convnext_central_32e"
proc = "JZcomb0_test"
date = "20250313-06"
# proc = "ttbar_test"
# date = "20250306-16"

metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/{proc}/{date}/box_metrics"
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/{proc}/{date}/jet_kin/"
# metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/ttbar/20250124-12/box_metrics"
# save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/ttbar/20250124-12/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

square_comp = False
image_format = "png"

print("=======================================================================================================")
print(f"Loading all jets from\n{metrics_folder}")
print("=======================================================================================================\n")

event_tar_pt      = load_object(metrics_folder+"/tarboxes_pt.pkl")
event_tru_pt      = load_object(metrics_folder+"/truboxes_pt.pkl")
event_p_pt        = load_object(metrics_folder+"/pboxes_pt.pkl")

total_tar_pt      = np.concatenate(event_tar_pt)
total_tru_pt      = np.concatenate(event_tru_pt)
total_p_pt        = np.concatenate(event_p_pt)
total_p_scr       = np.concatenate(load_object(metrics_folder+"/pboxes_scores.pkl"))
# IOU matched
match_tar_pt      = np.concatenate(load_object(metrics_folder+"/tarboxes_matched_pt.pkl"))
match_p_pt      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_pt.pkl"))
match_p_scr     = np.concatenate(load_object(metrics_folder+"/pboxes_matched_scr.pkl"))
unmatch_tar_pt    = np.concatenate(load_object(metrics_folder+"/tarboxes_unmatched_pt.pkl"))
unmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_pt.pkl"))
unmatch_p_scr   = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_scr.pkl"))
# dR matched
dRmatch_tar_pt    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRmatched_pt.pkl"))
dRmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_pt.pkl"))
dRmatch_p_scr   = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_scr.pkl"))
dRunmatch_tar_pt  = np.concatenate(load_object(metrics_folder+"/tarboxes_dRunmatched_pt.pkl"))
dRunmatch_p_pt  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_pt.pkl"))
dRunmatch_p_scr = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_scr.pkl"))
# dR matched (truth)
dRtruthmatch_tru_pt    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthmatched_pt.pkl"))
dRtruthmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_pt.pkl"))
dRtruthmatch_p_scr   = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_scr.pkl"))
dRtruthunmatch_tru_pt  = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthunmatched_pt.pkl"))
dRtruthunmatch_p_pt  = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthunmatched_pt.pkl"))
dRtruthunmatch_p_scr = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthunmatched_scr.pkl"))




print("Implementing a temporary, post-hoc confidence threshold:")
print(f"Number        total targets: {len(total_tar_pt)}\nNumber          total truth: {len(total_tru_pt)}\nNumber        total predictions: {len(total_p_pt)}")
print(f"Number      matched targets: {len(match_tar_pt)}\nNumber      matched predictions: {len(match_p_pt)}")
print(f"Number    unmatched targets: {len(unmatch_tar_pt)}\nNumber    unmatched predictions: {len(unmatch_p_pt)}")
print(f"Number dR   matched targets: {len(dRmatch_tar_pt)}\nNumber dR   matched predictions: {len(dRmatch_p_pt)}")
print(f"Number dR unmatched targets: {len(dRunmatch_tar_pt)}\nNumber dR unmatched predictions: {len(dRunmatch_p_pt)}")
print()
print(f"Number dR (truth) matched truth: {len(dRtruthmatch_tru_pt)}\nNumber dR (truth) matched preds: {len(dRtruthmatch_p_pt)}")
print(f"Number dR (truth) unmatched truth: {len(dRtruthunmatch_tru_pt)}\nNumber dR (truth) unmatched preds: {len(dRtruthunmatch_p_pt)}")

scr_threshold = 0.5
total_scr_mask = total_p_scr > scr_threshold
total_p_pt = total_p_pt[total_scr_mask]

match_scr_mask = match_p_scr > scr_threshold
match_tar_pt = match_tar_pt[match_scr_mask]
match_p_pt = match_p_pt[match_scr_mask]

dRmatch_scr_mask = dRmatch_p_scr > scr_threshold
dRmatch_tar_pt = dRmatch_tar_pt[dRmatch_scr_mask]
dRmatch_p_pt = dRmatch_p_pt[dRmatch_scr_mask]

unmatch_scr_mask = unmatch_p_scr > scr_threshold
unmatch_p_pt = unmatch_p_pt[unmatch_scr_mask]

dRunmatch_scr_mask = dRunmatch_p_scr > scr_threshold
dRunmatch_p_pt = dRunmatch_p_pt[dRunmatch_scr_mask]

# truth matching
dRtruthmatch_scr_mask = dRtruthmatch_p_scr > scr_threshold
dRtruthmatch_tru_pt = dRtruthmatch_tru_pt[dRtruthmatch_scr_mask]
dRtruthmatch_p_pt = dRtruthmatch_p_pt[dRtruthmatch_scr_mask]

dRtruthunmatch_scr_mask = dRtruthunmatch_p_scr > scr_threshold
dRtruthunmatch_p_pt = dRtruthunmatch_p_pt[dRtruthunmatch_scr_mask]

print("\n\nNew score threshold applied")
print(f"Number        total targets: {len(total_tar_pt)}\nNumber        total predictions: {len(total_p_pt)}")
print(f"Number      matched targets: {len(match_tar_pt)}\nNumber      matched predictions: {len(match_p_pt)}")
print(f"Number    unmatched targets: {len(unmatch_tar_pt)}\nNumber    unmatched predictions: {len(unmatch_p_pt)}")
print(f"Number dR   matched targets: {len(dRmatch_tar_pt)}\nNumber dR   matched predictions: {len(dRmatch_p_pt)}")
print(f"Number dR unmatched targets: {len(dRunmatch_tar_pt)}\nNumber dR unmatched predictions: {len(dRunmatch_p_pt)}")
print()
print(f"Number dR (truth) matched truth: {len(dRtruthmatch_tru_pt)}\nNumber dR (truth) matched preds: {len(dRtruthmatch_p_pt)}")
print(f"Number dR (truth) unmatched truth: {len(dRtruthunmatch_tru_pt)}\nNumber dR (truth) unmatched preds: {len(dRtruthunmatch_p_pt)}")


################################################################
print("=======================================================================================================")
print(f"Plotting jet pT, saving to {save_folder}")
print("=======================================================================================================\n")


print(f"Plotting total jet pT: {len(total_p_pt)} predictions, {len(total_tar_pt)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tar, bins, _    = ax0.hist(total_tar_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
freq_tru, bins, _    = ax0.hist(total_tru_pt,bins=bins,histtype='step',color='gold',lw=1.5,label='Truth Jets')
ax0.set_title('Transverse Momentum', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
f.savefig(save_folder + f'/jet_pt_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


print(f"Plotting IOU matched jet pT: {len(match_p_pt)} predictions, {len(match_tar_pt)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(match_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tar, bins, _    = ax0.hist(match_tar_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.set_title('Transverse Momentum IoU Matched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
f.savefig(save_folder + f'/jet_pt_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

print(f"Plotting dR matched jet pT: {len(dRmatch_p_pt)} predictions, {len(dRmatch_tar_pt)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(dRmatch_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tar, bins, _    = ax0.hist(dRmatch_tar_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
freq_tru, bins, _    = ax0.hist(dRtruthmatch_tru_pt,bins=bins,histtype='step',color='gold',lw=1.5,label='Truth Jets')
ax0.set_title('Transverse Momentum deltaR Matched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
f.savefig(save_folder + f'/jet_pt_dRmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


print(f"Plotting IOU unmatched jet pT: {len(unmatch_p_pt)} predictions, {len(unmatch_tar_pt)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(unmatch_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tar, bins, _    = ax0.hist(unmatch_tar_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.set_title('Transverse Momentum IoU Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
f.savefig(save_folder + f'/jet_pt_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


print(f"Plotting dR unmatched jet pT: {len(dRunmatch_p_pt)} predictions, {len(dRunmatch_tar_pt)} targets")
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(dRunmatch_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tar, bins, _    = ax0.hist(dRunmatch_tar_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
freq_tru, bins, _    = ax0.hist(dRtruthunmatch_tru_pt,bins=bins,histtype='step',color='gold',lw=1.5,label='Truth Jets')
ax0.set_title('Transverse Momentum deltaR Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
f.savefig(save_folder + f'/jet_pt_dRunmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()




print(f"Plotting total jet pT in match fraction bins!: {len(total_p_pt)} predictions, {len(total_tar_pt)} targets")
bin_edges = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 850]
bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
bin_width = np.diff(bin_edges)

f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_p_pt,bins=bin_edges,histtype='step',color='red',alpha=0.6,lw=1.5,label='Predicted Jets')
freq_tar, bins, _    = ax0.hist(total_tar_pt,bins=bin_edges,histtype='step',color='green',alpha=0.6,lw=1.5,label='Target Jets')
freq_tru, bins, _    = ax0.hist(total_tru_pt,bins=bin_edges,histtype='step',color='gold',alpha=0.6,lw=1.5,label='Truth Jets')
ax0.set_title('Transverse Momentum', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
f.savefig(save_folder + f'/jet_pt_total_binning.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


print(f"Plotting leading jet pT in each event: {len(event_tar_pt)} events, {len(event_p_pt)} events, {len(event_tru_pt)} events")
tar_lead_pt = np.array([max(x) for x in event_tar_pt])
tru_lead_pt = np.array([max(x) for x in event_tru_pt])
p_lead_pt = np.array([max(x) for x in event_p_pt])

f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(p_lead_pt,bins=100,histtype='step',color='red',lw=1.5,label='Predicted Jets')
freq_tar, bins, _    = ax0.hist(tar_lead_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
freq_tru, bins, _    = ax0.hist(tru_lead_pt,bins=bins,histtype='step',color='gold',lw=1.5,label='Truth Jets')
ax0.set_title('Transverse Momentum', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Leading Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
f.savefig(save_folder + f'/jet_pt_lead.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
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

    total_sq_tar_pt = np.concatenate(load_object(square_metrics_folder+"/tarboxes_pt.pkl"))
    total_sq_p_pt = np.concatenate(load_object(square_metrics_folder+"/pboxes_pt.pkl"))
    total_sq_p_scr = np.concatenate(load_object(square_metrics_folder+"/pboxes_scores.pkl"))
    # IOU matched
    match_sq_tar_pt = np.concatenate(load_object(square_metrics_folder+"/tarboxes_matched_pt.pkl"))
    match_sq_p_pt = np.concatenate(load_object(square_metrics_folder+"/pboxes_matched_pt.pkl"))
    unmatch_sq_tar_pt = np.concatenate(load_object(square_metrics_folder+"/tarboxes_unmatched_pt.pkl"))
    unmatch_sq_p_pt = np.concatenate(load_object(square_metrics_folder+"/pboxes_unmatched_pt.pkl"))
    # dR matched
    dRmatch_sq_tar_pt = np.concatenate(load_object(square_metrics_folder+"/tarboxes_dRmatched_pt.pkl"))
    dRmatch_sq_p_pt = np.concatenate(load_object(square_metrics_folder+"/pboxes_dRmatched_pt.pkl"))
    dRunmatch_sq_tar_pt = np.concatenate(load_object(square_metrics_folder+"/tarboxes_dRunmatched_pt.pkl"))
    dRunmatch_sq_p_pt = np.concatenate(load_object(square_metrics_folder+"/pboxes_dRunmatched_pt.pkl"))


    print(f"Plotting comparison of ALL jets")
    print(f"{len(total_p_pt)} di predictions, {len(total_tar_pt)} di targets, {len(total_sq_p_pt)} sq predictions, {len(total_sq_tar_pt)} sq targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(total_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets (Di)')
    freq_tar, bins, _    = ax0.hist(total_tar_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets (Di)')
    freq_tar, bins, _    = ax0.hist(total_sq_tar_pt,bins=bins,histtype='step',ls='--',color='green',lw=1.5,label='Target Jets (Sq)')
    freq_tar, bins, _    = ax0.hist(total_sq_p_pt,bins=bins,histtype='step',ls='--',color='red',lw=1.5,label='Pred Jets (Sq)')
    ax0.set_title('Transverse Momentum', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
    f.savefig(save_folder + f'/comp_jet_pt_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    print(f"Plotting comparison of IOU MATCHED jets")
    print(f"{len(match_p_pt)} matched di predictions, {len(match_tar_pt)} matched di targets, {len(match_sq_tar_pt)} matched sq predictions, {len(match_sq_p_pt)} matched sq targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(match_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets (Di)')
    freq_tar, bins, _    = ax0.hist(match_tar_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets (Di)')
    freq_tar, bins, _    = ax0.hist(match_sq_tar_pt,bins=bins,histtype='step',ls='--',color='green',lw=1.5,label='Target Jets (Sq)')
    freq_tar, bins, _    = ax0.hist(match_sq_p_pt,bins=bins,histtype='step',ls='--',color='red',lw=1.5,label='Pred Jets (Sq)')
    ax0.set_title('Transverse Momentum IoU Matched', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
    f.savefig(save_folder + f'/comp_jet_pt_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    print(f"Plotting comparison of dR MATCHED jets")
    print(f"{len(dRmatch_p_pt)} dR matched di predictions, {len(dRmatch_tar_pt)} dR matched di targets, {len(dRmatch_sq_tar_pt)} dR matched sq predictions, {len(dRmatch_sq_p_pt)} dR matched sq targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(dRmatch_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets (Di)')
    freq_tar, bins, _    = ax0.hist(dRmatch_tar_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets (Di)')
    freq_tar, bins, _    = ax0.hist(dRmatch_sq_tar_pt,bins=bins,histtype='step',ls='--',color='green',lw=1.5,label='Target Jets (Sq)')
    freq_tar, bins, _    = ax0.hist(dRmatch_sq_p_pt,bins=bins,histtype='step',ls='--',color='red',lw=1.5,label='Pred Jets (Sq)')
    ax0.set_title('Transverse Momentum deltaR Matched', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
    f.savefig(save_folder + f'/comp_jet_pt_dRmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    print(f"Plotting comparison of IOU UNMATCHED jets")
    print(f"{len(unmatch_p_pt)} unmatched di predictions, {len(unmatch_tar_pt)} unmatched di targets, {len(unmatch_sq_tar_pt)} unmatched sq predictions, {len(unmatch_sq_p_pt)} unmatched sq targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(unmatch_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets (Di)')
    freq_tar, bins, _    = ax0.hist(unmatch_tar_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets (Di)')
    freq_tar, bins, _    = ax0.hist(unmatch_sq_tar_pt,bins=bins,histtype='step',ls='--',color='green',lw=1.5,label='Target Jets (Sq)')
    freq_tar, bins, _    = ax0.hist(unmatch_sq_p_pt,bins=bins,histtype='step',ls='--',color='red',lw=1.5,label='Pred Jets (Sq)')
    ax0.set_title('Transverse Momentum IoU Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
    f.savefig(save_folder + f'/comp_jet_pt_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    print(f"Plotting comparison of dR UNMATCHED jets")
    print(f"{len(unmatch_p_pt)} dR unmatched di predictions, {len(unmatch_tar_pt)} dR unmatched di targets, {len(dRunmatch_sq_tar_pt)} dR unmatched sq predictions, {len(dRunmatch_sq_p_pt)} dR unmatched sq targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(dRunmatch_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets (Di)')
    freq_tar, bins, _    = ax0.hist(dRunmatch_tar_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets (Di)')
    freq_tar, bins, _    = ax0.hist(dRunmatch_sq_tar_pt,bins=bins,histtype='step',ls='--',color='green',lw=1.5,label='Target Jets (Sq)')
    freq_tar, bins, _    = ax0.hist(dRunmatch_sq_p_pt,bins=bins,histtype='step',ls='--',color='red',lw=1.5,label='Pred Jets (Sq)')
    ax0.set_title('Transverse Momentum deltaR Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
    f.savefig(save_folder + f'/comp_jet_pt_dRunmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


 


