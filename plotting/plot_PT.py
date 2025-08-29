import numpy as np 
import torch
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
# proc = "JZcomb0_test"
# date = "20250313-06"
# # date = "20250406-23"
proc = "ttbar_test"
date = "20250407-14"
# date = "20250626-18"

metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/{proc}/{date}/box_metrics/"
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/{proc}/{date}/jet_kin/"
if not os.path.exists(save_folder): os.makedirs(save_folder)
image_format = "png"


print("=======================================================================================================")
print(f"Loading all jets from\n{metrics_folder}")
print("=======================================================================================================\n")

event_tar_pt      = load_object(metrics_folder+"/tarboxes_pt.pkl")
event_tru_pt      = load_object(metrics_folder+"/truboxes_pt.pkl")
event_p_pt        = load_object(metrics_folder+"/pboxes_pt.pkl")
total_jet_weight  = np.concatenate(load_object(metrics_folder+"/jet_evt_weight.pkl"))
total_tar_jet_weight = total_jet_weight 
total_evt_weight  = load_object(metrics_folder+"/evt_weight.pkl")

total_tru_jet_weight = list()
for i in range(len(total_evt_weight)):
    total_tru_jet_weight.append([total_evt_weight[i] for j in range(len(event_tru_pt[i]))])
total_tru_jet_weight = np.concatenate(total_tru_jet_weight)

total_p_weight = list()
for i in range(len(total_evt_weight)):
    total_p_weight.append([total_evt_weight[i] for j in range(len(event_p_pt[i]))])
total_p_weight = np.concatenate(total_p_weight)
print(total_tar_jet_weight.shape, len(total_evt_weight), total_tru_jet_weight.shape, total_p_weight.shape)
print(len(event_tar_pt),len(event_tru_pt),len(event_p_pt))
print(len(np.concatenate(event_tar_pt)),len(np.concatenate(event_tru_pt)),len(np.concatenate(event_p_pt)))

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
total_p_weight = total_p_weight[total_scr_mask]

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



nominal = False
square_comp = False
total_unc = False
jet_lead_pt = False
jet_sublead_pt = False
jet_asymmetry = False
jet_thresh_cut = False
jet_thresh_cut2 = False
jet_thresh_cut3 = False
jet_thresh_cut1b = True
jet_thresh_cut3b = True

################################################################
print("=======================================================================================================")
print(f"Plotting jet pT, saving to {save_folder}")
print("=======================================================================================================\n")

if nominal:
    print(f"Plotting total jet pT: {len(total_p_pt)} predictions, {len(total_tar_pt)} targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(total_p_pt,bins=100,histtype='step',color='red',lw=1.5,label='Predicted Jets')
    freq_tar, bins, _    = ax0.hist(total_tar_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
    freq_tru, bins, _    = ax0.hist(total_tru_pt,bins=bins,histtype='step',color='gold',lw=1.5,label='Truth Jets')
    ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
    # hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
    f.savefig(save_folder + f'/jet_pt_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(total_p_pt,bins=100,weights=total_p_weight,histtype='step',color='red',lw=1.5,label='Predicted Jets')
    freq_tar, bins, _    = ax0.hist(total_tar_pt,bins=bins,weights=total_jet_weight,histtype='step',color='green',lw=1.5,label='Target Jets')
    # freq_tru, bins, _    = ax0.hist(total_tru_pt,bins=bins,weights=total_tru_jet_weight,histtype='step',color='gold',lw=1.5,label='Truth Jets')
    ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
    # hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
    f.savefig(save_folder + f'/jet_pt_total2.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
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


    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(p_lead_pt,bins=100,histtype='step',color='red',lw=1.5,label='Predicted Jets')
    freq_tar, bins, _    = ax0.hist(tar_lead_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
    ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.7),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Leading Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
    f.savefig(save_folder + f'/jet_pt_leadb.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
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







if total_unc:
    # pt w/ errors
    print("Plotting pT with stat. unc and ratio ")
    unc_save_folder = save_folder + "/unc/"
    if not os.path.exists(unc_save_folder): os.makedirs(unc_save_folder)

    bin_start = min(min(total_p_pt),min(total_tar_pt),min(total_tru_pt))
    bin_stop = max(max(total_p_pt),max(total_tar_pt),max(total_tru_pt))
    print(bin_start, bin_stop)
    bin_start = (np.floor(bin_start / 10) * 10) - 1
    # bin_start = -21
    bin_stop = np.ceil(bin_stop / 10) * 10
    bin_width = 20
    n_bins = int(np.ceil((bin_stop - bin_start) / bin_width))
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # plot 1
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tru, bins, _   = ax[0].hist(total_tru_pt,bins=bins,histtype='step',color='gold',lw=2,label='Truth Jets')
    freq_tar, bins, _   = ax[0].hist(total_tar_pt,bins=bins,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(total_p_pt,bins=bins,histtype='step',color='red',lw=2,label='CNN Jets')
    # stat. unc sqrt(counts)
    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)

    ax[0].errorbar(bin_centers, freq_tru,  yerr=tru_errors, color='gold', ls='none')
    ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')

    ratio_pred_tru = np.divide(freq_pre, freq_tru, out=np.zeros_like(freq_pre), where=freq_tru != 0)
    ratio_tar_tru  = np.divide(freq_tar, freq_tru, out=np.zeros_like(freq_tar), where=freq_tru != 0)
    ax[1].hlines(y=1, xmin=20, xmax=bin_centers[-2], color='gold', linestyle='-', lw=1.6)

    # uncertainty on the ratio
    # ratio_errors1 = ratio_pred_tru * np.sqrt((pred_errors / freq_pred) ** 2 + (tru_errors / freq_tru) ** 2)
    ratio_pred_erro = np.sqrt((pre_errors / freq_tru) ** 2 + (freq_pre * tru_errors / freq_tru**2)**2)
    ratio_tar_erro = np.sqrt((tar_errors / freq_tru) ** 2 + (freq_tar * tru_errors / freq_tru**2)**2)
    # ax[1].fill_between(bin_centers, ratio_pred_tru - ratio_pred_erro, ratio_pred_tru + ratio_pred_erro, alpha=0.5, edgecolor='crimson', facecolor='red')
    ax[1].fill_between(bin_centers[1:-1], ratio_pred_tru[1:-1] - ratio_pred_erro[1:-1], ratio_pred_tru[1:-1] + ratio_pred_erro[1:-1], alpha=0.5, edgecolor='crimson', facecolor='red')
    ax[1].fill_between(bin_centers[1:-1], ratio_tar_tru[1:-1] - ratio_tar_erro[1:-1], ratio_tar_tru[1:-1] + ratio_tar_erro[1:-1], alpha=0.5, edgecolor='forestgreen', facecolor='green')

    tar_line =  matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
    pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    tru_line =  matplotlib.lines.Line2D([0], [0], color='gold', lw=3)  
    ax[0].legend([tru_line,tar_line,pre_line], ['Truth jets','Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax[0].set(yscale='log',ylabel='Number of jets',ylim=(1,5e5))
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[1].set(xlabel='Jet $p_T$ [GeV]')
    ax[1].set(ylim=(0.0,2.5))
    ax[1].set(xlim=(-50.0,bin_stop+10))
    ax[1].set_yticks(np.arange(0.0, 3.0, 1.0))
    new_yticklabels = [f'{int(tick)}' for tick in np.arange(0.0, 3.0, 1.0)]  
    ax[1].set_yticklabels(new_yticklabels)
    tick_labels = ax[1].get_xticklabels()
    tick_positions = ax[1].get_xticks()
    for label in tick_labels:
        label.set_verticalalignment('bottom')  
        label.set_y(label.get_position()[1] - 0.22)
    f.savefig(unc_save_folder + f'/jet_pt_total_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()




    # plot 2
    print(total_tar_jet_weight.shape, len(total_evt_weight), total_tru_jet_weight.shape, total_p_weight.shape)
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tru, bins, _   = ax[0].hist(total_tru_pt,bins=bins,weights=total_tru_jet_weight,histtype='step',color='gold',lw=2,label='Truth Jets')
    freq_tar, bins, _   = ax[0].hist(total_tar_pt,bins=bins,weights=total_tar_jet_weight,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(total_p_pt,bins=bins,weights=total_p_weight,histtype='step',color='red',lw=2,label='CNN Jets')
    # stat. unc sqrt(counts)
    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)

    ax[0].errorbar(bin_centers, freq_tru,  yerr=tru_errors, color='gold', ls='none')
    ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')

    ratio_pred_tru = np.divide(freq_pre, freq_tru, out=np.zeros_like(freq_pre), where=freq_tru != 0)
    ratio_tar_tru  = np.divide(freq_tar, freq_tru, out=np.zeros_like(freq_tar), where=freq_tru != 0)
    ax[1].hlines(y=1, xmin=20, xmax=bin_centers[-2], color='gold', linestyle='-', lw=1.6)

    # uncertainty on the ratio
    # ratio_errors1 = ratio_pred_tru * np.sqrt((pred_errors / freq_pred) ** 2 + (tru_errors / freq_tru) ** 2)
    ratio_pred_erro = np.sqrt((pre_errors / freq_tru) ** 2 + (freq_pre * tru_errors / freq_tru**2)**2)
    ratio_tar_erro = np.sqrt((tar_errors / freq_tru) ** 2 + (freq_tar * tru_errors / freq_tru**2)**2)
    # ax[1].fill_between(bin_centers, ratio_pred_tru - ratio_pred_erro, ratio_pred_tru + ratio_pred_erro, alpha=0.5, edgecolor='crimson', facecolor='red')
    ax[1].fill_between(bin_centers[1:-1], ratio_pred_tru[1:-1] - ratio_pred_erro[1:-1], ratio_pred_tru[1:-1] + ratio_pred_erro[1:-1], alpha=0.5, edgecolor='crimson', facecolor='red')
    ax[1].fill_between(bin_centers[1:-1], ratio_tar_tru[1:-1] - ratio_tar_erro[1:-1], ratio_tar_tru[1:-1] + ratio_tar_erro[1:-1], alpha=0.5, edgecolor='forestgreen', facecolor='green')

    tar_line =  matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
    pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    tru_line =  matplotlib.lines.Line2D([0], [0], color='gold', lw=3)  
    ax[0].legend([tru_line,tar_line,pre_line], ['Truth jets','Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax[0].set(ylabel='Number of jets (weighted)')
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[1].set(xlabel='Jet $p_T$ [GeV]')
    ax[1].set(ylim=(0.0,2.5))
    ax[1].set(xlim=(-50.0,bin_stop+10))
    ax[1].set_yticks(np.arange(0.0, 3.0, 1.0))
    new_yticklabels = [f'{int(tick)}' for tick in np.arange(0.0, 3.0, 1.0)]  
    ax[1].set_yticklabels(new_yticklabels)
    tick_labels = ax[1].get_xticklabels()
    tick_positions = ax[1].get_xticks()
    for label in tick_labels:
        label.set_verticalalignment('bottom')  
        label.set_y(label.get_position()[1] - 0.22)
    f.savefig(unc_save_folder + f'/jet_pt_weight_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    ##########################################################################################################
    # plot 1b
    bin_start = min(min(total_p_pt),min(total_tar_pt))
    bin_stop = max(max(total_p_pt),max(total_tar_pt))
    print(bin_start, bin_stop)
    bin_start = (np.floor(bin_start / 10) * 10) - 1
    # bin_start = -21
    bin_stop = np.ceil(bin_stop / 10) * 10
    bin_width = 20
    n_bins = int(np.ceil((bin_stop - bin_start) / bin_width))
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tar, bins, _   = ax[0].hist(total_tar_pt,bins=bins,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(total_p_pt,bins=bins,histtype='step',color='red',lw=2,label='CNN Jets')
    # stat. unc sqrt(counts)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)
    ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')

    ratio_pred_tar = np.divide(freq_pre, freq_tar, out=np.zeros_like(freq_pre), where=freq_tar != 0)
    ax[1].hlines(y=1, xmin=20, xmax=bin_centers[-2], color='green', linestyle='-', lw=1.6)
    # uncertainty on the ratio
    # ratio_errors1 = ratio_pred_tru * np.sqrt((pred_errors / freq_pred) ** 2 + (tru_errors / freq_tru) ** 2)
    ratio_pred_erro = np.sqrt((pre_errors / freq_tar) ** 2 + (freq_pre * tar_errors / freq_tar**2)**2)
    # ax[1].fill_between(bin_centers, ratio_pred_tru - ratio_pred_erro, ratio_pred_tru + ratio_pred_erro, alpha=0.5, edgecolor='crimson', facecolor='red')
    ax[1].fill_between(bin_centers[:-1], ratio_pred_tar[:-1] - ratio_pred_erro[:-1], ratio_pred_tar[:-1] + ratio_pred_erro[:-1], alpha=0.5, edgecolor='crimson', facecolor='red')

    tar_line =  matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
    pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    ax[0].legend([tar_line,pre_line], ['Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax[0].set(yscale='log',ylabel='Number of jets',ylim=(1,5e5))
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[1].set(xlabel='Jet $p_T$ [GeV]')
    ax[1].set(ylim=(0.0,2.5))
    ax[1].set(xlim=(-50.0,bin_stop+10))
    ax[1].set_yticks(np.arange(0.0, 3.0, 1.0))
    new_yticklabels = [f'{int(tick)}' for tick in np.arange(0.0, 3.0, 1.0)]  
    ax[1].set_yticklabels(new_yticklabels)
    tick_labels = ax[1].get_xticklabels()
    tick_positions = ax[1].get_xticks()
    for label in tick_labels:
        label.set_verticalalignment('bottom')  
        label.set_y(label.get_position()[1] - 0.22)
    f.savefig(unc_save_folder + f'/jet_pt_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()



if jet_lead_pt:
    print(f"Plotting leading jet pT in each event: {len(event_tar_pt)} events, {len(event_p_pt)} events, {len(event_tru_pt)} events")
    tar_lead_pt = np.array([max(x) for x in event_tar_pt])
    tru_lead_pt = np.array([max(x) for x in event_tru_pt])
    p_lead_pt = np.array([max(x) for x in event_p_pt])

    bin_start = min(min(p_lead_pt),min(tar_lead_pt),min(tru_lead_pt))
    bin_stop = max(max(p_lead_pt),max(tar_lead_pt),max(tru_lead_pt))
    print(bin_start, bin_stop)
    bin_start = (np.floor(bin_start / 10) * 10) - 1
    # bin_start = -21
    bin_stop = np.ceil(bin_stop / 10) * 10
    bin_width = 20
    n_bins = int(np.ceil((bin_stop - bin_start) / bin_width))
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # plot 1
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tru, bins, _   = ax[0].hist(tru_lead_pt,bins=bins,histtype='step',color='gold',lw=2,label='Truth Jets')
    freq_tar, bins, _   = ax[0].hist(tar_lead_pt,bins=bins,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(p_lead_pt,bins=bins,histtype='step',color='red',lw=2,label='CNN Jets')
    # stat. unc sqrt(counts)
    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)
    ax[0].errorbar(bin_centers, freq_tru,  yerr=tru_errors, color='gold', ls='none')
    ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')

    ratio_pred_tru = np.divide(freq_pre, freq_tru, out=np.zeros_like(freq_pre), where=freq_tru != 0)
    ratio_tar_tru  = np.divide(freq_tar, freq_tru, out=np.zeros_like(freq_tar), where=freq_tru != 0)
    ax[1].hlines(y=1, xmin=20, xmax=bin_centers[-2], color='gold', linestyle='-', lw=1.6)

    # uncertainty on the ratio
    # ratio_errors1 = ratio_pred_tru * np.sqrt((pred_errors / freq_pred) ** 2 + (tru_errors / freq_tru) ** 2)
    ratio_pred_erro = np.sqrt((pre_errors / freq_tru) ** 2 + (freq_pre * tru_errors / freq_tru**2)**2)
    ratio_tar_erro = np.sqrt((tar_errors / freq_tru) ** 2 + (freq_tar * tru_errors / freq_tru**2)**2)
    ax[1].fill_between(bin_centers[0:-1], ratio_pred_tru[0:-1] - ratio_pred_erro[0:-1], ratio_pred_tru[0:-1] + ratio_pred_erro[0:-1], alpha=0.5, edgecolor='crimson', facecolor='red')
    ax[1].fill_between(bin_centers[0:-1], ratio_tar_tru[0:-1] - ratio_tar_erro[0:-1], ratio_tar_tru[0:-1] + ratio_tar_erro[0:-1], alpha=0.5, edgecolor='forestgreen', facecolor='green')

    tar_line =  matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
    pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    tru_line =  matplotlib.lines.Line2D([0], [0], color='gold', lw=3)  
    ax[0].legend([tru_line,tar_line,pre_line], ['Truth jets','Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax[0].set(yscale='log',ylabel='Number of events',ylim=(1,5e5))
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[1].set(xlabel='Leading Jet $p_T$ [GeV]')
    ax[1].set(ylim=(0.0,2.5))
    ax[1].set(xlim=(-50.0,bin_stop+10))
    ax[1].set_yticks(np.arange(0.0, 3.0, 1.0))
    new_yticklabels = [f'{int(tick)}' for tick in np.arange(0.0, 3.0, 1.0)]  
    ax[1].set_yticklabels(new_yticklabels)
    tick_labels = ax[1].get_xticklabels()
    tick_positions = ax[1].get_xticks()
    for label in tick_labels:
        label.set_verticalalignment('bottom')  
        label.set_y(label.get_position()[1] - 0.22)
    f.savefig(save_folder + f'/jet_lead_pt_total_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()
   
    ########################################################################################################################
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tru, bins, _   = ax[0].hist(tru_lead_pt,bins=bins,weights=total_evt_weight,histtype='step',color='gold',lw=2,label='Truth Jets')
    freq_tar, bins, _   = ax[0].hist(tar_lead_pt,bins=bins,weights=total_evt_weight,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(p_lead_pt,bins=bins,weights=total_evt_weight,histtype='step',color='red',lw=2,label='CNN Jets')
    # stat. unc sqrt(counts)
    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)
    # ax[0].errorbar(bin_centers, freq_tru,  yerr=tru_errors, color='gold', ls='none')
    # ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    # ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')

    ratio_pred_tru = np.divide(freq_pre, freq_tru, out=np.zeros_like(freq_pre), where=freq_tru != 0)
    ratio_tar_tru  = np.divide(freq_tar, freq_tru, out=np.zeros_like(freq_tar), where=freq_tru != 0)
    ax[1].hlines(y=1, xmin=20, xmax=bin_centers[-2], color='gold', linestyle='-', lw=1.6)

    # uncertainty on the ratio
    ratio_pred_erro = np.sqrt((pre_errors / freq_tru) ** 2 + (freq_pre * tru_errors / freq_tru**2)**2)
    ratio_tar_erro = np.sqrt((tar_errors / freq_tru) ** 2 + (freq_tar * tru_errors / freq_tru**2)**2)
    ax[1].fill_between(bin_centers[0:-1], ratio_pred_tru[0:-1] - ratio_pred_erro[0:-1], ratio_pred_tru[0:-1] + ratio_pred_erro[0:-1], alpha=0.5, edgecolor='crimson', facecolor='red')
    ax[1].fill_between(bin_centers[0:-1], ratio_tar_tru[0:-1] - ratio_tar_erro[0:-1], ratio_tar_tru[0:-1] + ratio_tar_erro[0:-1], alpha=0.5, edgecolor='forestgreen', facecolor='green')

    tar_line =  matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
    pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    tru_line =  matplotlib.lines.Line2D([0], [0], color='gold', lw=3)  
    ax[0].legend([tru_line,tar_line,pre_line], ['Truth jets','Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax[0].set(yscale='log',ylabel='Number of weighted events')
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[1].set(xlabel='Leading Jet $p_T$ [GeV]')
    ax[1].set(ylim=(0.0,2.5))
    ax[1].set(xlim=(-50.0,bin_stop+10))
    ax[1].set_yticks(np.arange(0.0, 3.0, 1.0))
    new_yticklabels = [f'{int(tick)}' for tick in np.arange(0.0, 3.0, 1.0)]  
    ax[1].set_yticklabels(new_yticklabels)
    tick_labels = ax[1].get_xticklabels()
    tick_positions = ax[1].get_xticks()
    for label in tick_labels:
        label.set_verticalalignment('bottom')  
        label.set_y(label.get_position()[1] - 0.22)
    f.savefig(save_folder + f'/jet_lead_pt_weight_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    #################################################################################################################
    # plot 1b
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tar, bins, _   = ax[0].hist(tar_lead_pt,bins=bins,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(p_lead_pt,bins=bins,histtype='step',color='red',lw=2,label='CNN Jets')
    # stat. unc sqrt(counts)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)
    ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')

    ratio_pred_tar = np.divide(freq_pre, freq_tar, out=np.zeros_like(freq_pre), where=freq_tar != 0)
    ax[1].hlines(y=1, xmin=20, xmax=bin_centers[-2], color='green', linestyle='-', lw=1.6)
    # uncertainty on the ratio
    ratio_pred_erro = np.sqrt((pre_errors / freq_tar) ** 2 + (freq_pre * tar_errors / freq_tar**2)**2)
    ax[1].fill_between(bin_centers[0:-1], ratio_pred_tar[0:-1] - ratio_pred_erro[0:-1], ratio_pred_tar[0:-1] + ratio_pred_erro[0:-1], alpha=0.5, edgecolor='crimson', facecolor='red')

    pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    tru_line =  matplotlib.lines.Line2D([0], [0], color='gold', lw=3)  
    ax[0].legend([tar_line,pre_line], ['Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax[0].set(yscale='log',ylabel='Number of events',ylim=(1,5e5))
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[1].set(xlabel='Leading Jet $p_T$ [GeV]')
    ax[1].set(ylim=(0.0,2.5))
    ax[1].set(xlim=(-50.0,bin_stop+10))
    ax[1].set_yticks(np.arange(0.0, 3.0, 1.0))
    new_yticklabels = [f'{int(tick)}' for tick in np.arange(0.0, 3.0, 1.0)]  
    ax[1].set_yticklabels(new_yticklabels)
    tick_labels = ax[1].get_xticklabels()
    tick_positions = ax[1].get_xticks()
    for label in tick_labels:
        label.set_verticalalignment('bottom')  
        label.set_y(label.get_position()[1] - 0.22)
    f.savefig(save_folder + f'/jet_lead_pt_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()




def nth_leading_jet_pt(list_of_jet_pts_in_event,n):
    try:
        return sorted(list_of_jet_pts_in_event,reverse=True)[n-1]
    except IndexError or ValueError:
        # Doesn't have enough (or any) jets, automatically lost in cut
        return np.nan



if jet_sublead_pt:
    print(f"Plotting subleading jet pT in each event: {len(event_tar_pt)} events, {len(event_p_pt)} events, {len(event_tru_pt)} events")
    tar_sublead_pt = np.array([nth_leading_jet_pt(x,2) for x in event_tar_pt])
    tru_sublead_pt = np.array([nth_leading_jet_pt(x,2) for x in event_tru_pt])
    p_sublead_pt = np.array([nth_leading_jet_pt(x,2) for x in event_p_pt])

    bin_start = min(min(p_sublead_pt),min(tar_sublead_pt),min(tru_sublead_pt))
    bin_stop = max(max(p_sublead_pt),max(tar_sublead_pt),max(tru_sublead_pt))
    print(bin_start, bin_stop)
    bin_start = (np.floor(bin_start / 10) * 10) - 1
    bin_stop = np.ceil(bin_stop / 10) * 10
    bin_width = 20
    n_bins = int(np.ceil((bin_stop - bin_start) / bin_width))
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # plot 1
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tru, bins, _   = ax[0].hist(tru_lead_pt,bins=bins,histtype='step',color='gold',lw=2,label='Truth Jets')
    freq_tar, bins, _   = ax[0].hist(tar_lead_pt,bins=bins,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(p_lead_pt,bins=bins,histtype='step',color='red',lw=2,label='CNN Jets')
    # stat. unc sqrt(counts)
    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)
    ax[0].errorbar(bin_centers, freq_tru,  yerr=tru_errors, color='gold', ls='none')
    ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')

    ratio_pred_tru = np.divide(freq_pre, freq_tru, out=np.zeros_like(freq_pre), where=freq_tru != 0)
    ratio_tar_tru  = np.divide(freq_tar, freq_tru, out=np.zeros_like(freq_tar), where=freq_tru != 0)
    ax[1].hlines(y=1, xmin=20, xmax=bin_centers[-2], color='gold', linestyle='-', lw=1.6)

    # uncertainty on the ratio
    # ratio_errors1 = ratio_pred_tru * np.sqrt((pred_errors / freq_pred) ** 2 + (tru_errors / freq_tru) ** 2)
    ratio_pred_erro = np.sqrt((pre_errors / freq_tru) ** 2 + (freq_pre * tru_errors / freq_tru**2)**2)
    ratio_tar_erro = np.sqrt((tar_errors / freq_tru) ** 2 + (freq_tar * tru_errors / freq_tru**2)**2)
    ax[1].fill_between(bin_centers[0:-1], ratio_pred_tru[0:-1] - ratio_pred_erro[0:-1], ratio_pred_tru[0:-1] + ratio_pred_erro[0:-1], alpha=0.5, edgecolor='crimson', facecolor='red')
    ax[1].fill_between(bin_centers[0:-1], ratio_tar_tru[0:-1] - ratio_tar_erro[0:-1], ratio_tar_tru[0:-1] + ratio_tar_erro[0:-1], alpha=0.5, edgecolor='forestgreen', facecolor='green')

    tar_line =  matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
    pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    tru_line =  matplotlib.lines.Line2D([0], [0], color='gold', lw=3)  
    ax[0].legend([tru_line,tar_line,pre_line], ['Truth jets','Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax[0].set(yscale='log',ylabel='Number of events',ylim=(1,5e5))
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[1].set(xlabel='Subleading Jet $p_T$ [GeV]')
    ax[1].set(ylim=(0.0,2.5))
    ax[1].set(xlim=(-50.0,bin_stop+10))
    ax[1].set_yticks(np.arange(0.0, 3.0, 1.0))
    new_yticklabels = [f'{int(tick)}' for tick in np.arange(0.0, 3.0, 1.0)]  
    ax[1].set_yticklabels(new_yticklabels)
    tick_labels = ax[1].get_xticklabels()
    tick_positions = ax[1].get_xticks()
    for label in tick_labels:
        label.set_verticalalignment('bottom')  
        label.set_y(label.get_position()[1] - 0.22)
    f.savefig(save_folder + f'/jet_sublead_pt_total_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()
    


    ####################################################################################################################
    # weighted plot
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tru, bins, _   = ax[0].hist(tru_lead_pt,bins=bins,weights=total_evt_weight,histtype='step',color='gold',lw=2,label='Truth Jets')
    freq_tar, bins, _   = ax[0].hist(tar_lead_pt,bins=bins,weights=total_evt_weight,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(p_lead_pt,bins=bins,weights=total_evt_weight,histtype='step',color='red',lw=2,label='CNN Jets')
    # stat. unc sqrt(counts)
    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)
    # ax[0].errorbar(bin_centers, freq_tru,  yerr=tru_errors, color='gold', ls='none')
    # ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    # ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')

    ratio_pred_tru = np.divide(freq_pre, freq_tru, out=np.zeros_like(freq_pre), where=freq_tru != 0)
    ratio_tar_tru  = np.divide(freq_tar, freq_tru, out=np.zeros_like(freq_tar), where=freq_tru != 0)
    ax[1].hlines(y=1, xmin=20, xmax=bin_centers[-2], color='gold', linestyle='-', lw=1.6)

    # uncertainty on the ratio
    ratio_pred_erro = np.sqrt((pre_errors / freq_tru) ** 2 + (freq_pre * tru_errors / freq_tru**2)**2)
    ratio_tar_erro = np.sqrt((tar_errors / freq_tru) ** 2 + (freq_tar * tru_errors / freq_tru**2)**2)
    ax[1].fill_between(bin_centers[0:-1], ratio_pred_tru[0:-1] - ratio_pred_erro[0:-1], ratio_pred_tru[0:-1] + ratio_pred_erro[0:-1], alpha=0.5, edgecolor='crimson', facecolor='red')
    ax[1].fill_between(bin_centers[0:-1], ratio_tar_tru[0:-1] - ratio_tar_erro[0:-1], ratio_tar_tru[0:-1] + ratio_tar_erro[0:-1], alpha=0.5, edgecolor='forestgreen', facecolor='green')

    tar_line =  matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
    pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    tru_line =  matplotlib.lines.Line2D([0], [0], color='gold', lw=3)  
    ax[0].legend([tru_line,tar_line,pre_line], ['Truth jets','Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax[0].set(yscale='log',ylabel='Number of weighted events')
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[1].set(xlabel='Subleading Jet $p_T$ [GeV]')
    ax[1].set(ylim=(0.0,2.5))
    ax[1].set(xlim=(-50.0,bin_stop+10))
    ax[1].set_yticks(np.arange(0.0, 3.0, 1.0))
    new_yticklabels = [f'{int(tick)}' for tick in np.arange(0.0, 3.0, 1.0)]  
    ax[1].set_yticklabels(new_yticklabels)
    tick_labels = ax[1].get_xticklabels()
    tick_positions = ax[1].get_xticks()
    for label in tick_labels:
        label.set_verticalalignment('bottom')  
        label.set_y(label.get_position()[1] - 0.22)
    f.savefig(save_folder + f'/jet_sublead_pt_weight_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    #####################################################################################################
    # plot 1b
    bin_start = min(min(p_sublead_pt),min(tar_sublead_pt))
    bin_stop = max(max(p_sublead_pt),max(tar_sublead_pt))
    print(bin_start, bin_stop)
    bin_start = (np.floor(bin_start / 10) * 10) - 1
    bin_stop = np.ceil(bin_stop / 10) * 10
    bin_width = 20
    n_bins = int(np.ceil((bin_stop - bin_start) / bin_width))
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tar, bins, _   = ax[0].hist(tar_lead_pt,bins=bins,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(p_lead_pt,bins=bins,histtype='step',color='red',lw=2,label='CNN Jets')
    # stat. unc sqrt(counts)
    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)
    ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')

    ratio_pred_tar = np.divide(freq_pre, freq_tar, out=np.zeros_like(freq_pre), where=freq_tar != 0)
    ax[1].hlines(y=1, xmin=20, xmax=bin_centers[-2], color='green', linestyle='-', lw=1.6)

    # uncertainty on the ratio
    # ratio_errors1 = ratio_pred_tru * np.sqrt((pred_errors / freq_pred) ** 2 + (tru_errors / freq_tru) ** 2)
    ratio_pred_erro = np.sqrt((pre_errors / freq_tar) ** 2 + (freq_pre * tar_errors / freq_tar**2)**2)
    ax[1].fill_between(bin_centers[0:-1], ratio_pred_tar[0:-1] - ratio_pred_erro[0:-1], ratio_pred_tar[0:-1] + ratio_pred_erro[0:-1], alpha=0.5, edgecolor='crimson', facecolor='red')

    tar_line =  matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
    pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    ax[0].legend([tar_line,pre_line], ['Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax[0].set(yscale='log',ylabel='Number of events',ylim=(1,5e5))
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[1].set(xlabel='Subleading Jet $p_T$ [GeV]')
    ax[1].set(ylim=(0.0,2.5))
    ax[1].set(xlim=(-50.0,bin_stop+10))
    ax[1].set_yticks(np.arange(0.0, 3.0, 1.0))
    new_yticklabels = [f'{int(tick)}' for tick in np.arange(0.0, 3.0, 1.0)]  
    ax[1].set_yticklabels(new_yticklabels)
    tick_labels = ax[1].get_xticklabels()
    tick_positions = ax[1].get_xticks()
    for label in tick_labels:
        label.set_verticalalignment('bottom')  
        label.set_y(label.get_position()[1] - 0.22)
    f.savefig(save_folder + f'/jet_sublead_pt_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()







if jet_asymmetry:
    print(f"Plotting jet asymmetry 2 ways: {len(event_tar_pt)} events, {len(event_p_pt)} events, {len(event_tru_pt)} events")
    tar_lead_pt = np.array([max(x) for x in event_tar_pt])
    tru_lead_pt = np.array([max(x) for x in event_tru_pt])
    p_lead_pt = np.array([max(x) for x in event_p_pt])

    tar_sublead_pt = np.array([nth_leading_jet_pt(x,2) for x in event_tar_pt])
    tru_sublead_pt = np.array([nth_leading_jet_pt(x,2) for x in event_tru_pt])
    p_sublead_pt   = np.array([nth_leading_jet_pt(x,2) for x in event_p_pt])

    # first asymmetry: diff/avg
    tar_jet_num   = tar_lead_pt - tar_sublead_pt
    tar_jet_denom = (tar_lead_pt + tar_sublead_pt) / 2
    tar_jet_asym  = tar_jet_num / tar_jet_denom

    tru_jet_num   = tru_lead_pt - tru_sublead_pt
    tru_jet_denom = (tru_lead_pt + tru_sublead_pt) / 2
    tru_jet_asym  = tru_jet_num / tru_jet_denom

    pre_jet_num   = p_lead_pt - p_sublead_pt
    pre_jet_denom = (p_lead_pt + p_sublead_pt) / 2
    pre_jet_asym  = pre_jet_num / pre_jet_denom

    ###################################################################################################################
    bin_start = min(min(pre_jet_asym),min(tru_jet_asym),min(tar_jet_asym))
    bin_stop = max(max(pre_jet_asym),max(tru_jet_asym),max(tar_jet_asym))
    print(bin_start, bin_stop)
    # bin_start = (np.floor(bin_start / 10) * 10) - 1
    # bin_stop = np.ceil(bin_stop / 10) * 10
    n_bins = 150
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # plot 1
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tru, bins, _   = ax[0].hist(tru_jet_asym,bins=bins,histtype='step',color='gold',lw=2,label='Truth Jets')
    freq_tar, bins, _   = ax[0].hist(tar_jet_asym,bins=bins,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(pre_jet_asym,bins=bins,histtype='step',color='red',lw=2,label='CNN Jets')
    # stat. unc sqrt(counts)
    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)
    ax[0].errorbar(bin_centers, freq_tru,  yerr=tru_errors, color='gold', ls='none')
    ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')
    ratio_pred_tru = np.divide(freq_pre, freq_tru, out=np.zeros_like(freq_pre), where=freq_tru != 0)
    ratio_tar_tru  = np.divide(freq_tar, freq_tru, out=np.zeros_like(freq_tar), where=freq_tru != 0)
    ax[1].hlines(y=1, xmin=bin_start, xmax=bin_stop, color='gold', linestyle='-', lw=1.6)

    # uncertainty on the ratio
    ratio_pred_erro = np.sqrt((pre_errors / freq_tru) ** 2 + (freq_pre * tru_errors / freq_tru**2)**2)
    ratio_tar_erro = np.sqrt((tar_errors / freq_tru) ** 2 + (freq_tar * tru_errors / freq_tru**2)**2)
    ax[1].fill_between(bin_centers[1:-1], ratio_pred_tru[1:-1] - ratio_pred_erro[1:-1], ratio_pred_tru[1:-1] + ratio_pred_erro[1:-1], alpha=0.5, edgecolor='crimson', facecolor='red')
    ax[1].fill_between(bin_centers[1:-1], ratio_tar_tru[1:-1] - ratio_tar_erro[1:-1], ratio_tar_tru[1:-1] + ratio_tar_erro[1:-1], alpha=0.5, edgecolor='forestgreen', facecolor='green')

    tar_line =  matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
    pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    tru_line =  matplotlib.lines.Line2D([0], [0], color='gold', lw=3)  
    ax[0].legend([tru_line,tar_line,pre_line], ['Truth jets','Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax[0].set(yscale='log',ylabel='Number of events')
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[1].set(xlabel='Jet $p_T$ Asymmetry',xlim=(-0.1,2.1),ylim=(0,2))
    f.savefig(save_folder + f'/jet_pt_asym_total_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    # plot 1b
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tru, bins, _   = ax[0].hist(tru_jet_asym,bins=bins,weights=total_evt_weight,histtype='step',color='gold',lw=2,label='Truth Jets')
    freq_tar, bins, _   = ax[0].hist(tar_jet_asym,bins=bins,weights=total_evt_weight,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(pre_jet_asym,bins=bins,weights=total_evt_weight,histtype='step',color='red',lw=2,label='CNN Jets')
    # stat. unc sqrt(counts)
    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)
    # ax[0].errorbar(bin_centers, freq_tru,  yerr=tru_errors, color='gold', ls='none')
    # ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    # ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')
    ratio_pred_tru = np.divide(freq_pre, freq_tru, out=np.zeros_like(freq_pre), where=freq_tru != 0)
    ratio_tar_tru  = np.divide(freq_tar, freq_tru, out=np.zeros_like(freq_tar), where=freq_tru != 0)
    ax[1].hlines(y=1, xmin=bin_start, xmax=bin_stop, color='gold', linestyle='-', lw=1.6)

    # uncertainty on the ratio
    ratio_pred_erro = np.sqrt((pre_errors / freq_tru) ** 2 + (freq_pre * tru_errors / freq_tru**2)**2)
    ratio_tar_erro = np.sqrt((tar_errors / freq_tru) ** 2 + (freq_tar * tru_errors / freq_tru**2)**2)
    ax[1].fill_between(bin_centers[1:-1], ratio_pred_tru[1:-1] - ratio_pred_erro[1:-1], ratio_pred_tru[1:-1] + ratio_pred_erro[1:-1], alpha=0.5, edgecolor='crimson', facecolor='red')
    ax[1].fill_between(bin_centers[1:-1], ratio_tar_tru[1:-1] - ratio_tar_erro[1:-1], ratio_tar_tru[1:-1] + ratio_tar_erro[1:-1], alpha=0.5, edgecolor='forestgreen', facecolor='green')

    tar_line =  matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
    pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    tru_line =  matplotlib.lines.Line2D([0], [0], color='gold', lw=3)  
    ax[0].legend([tru_line,tar_line,pre_line], ['Truth jets','Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax[0].set(yscale='log',ylabel='Number of events (weighted)')
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[1].set(xlabel='Jet $p_T$ Asymmetry',xlim=(-0.1,2.1),ylim=(0,10))
    f.savefig(save_folder + f'/jet_pt_asym_weight_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    ###################################################################################################################
    # second asymmetry: jet pt1/jet pt2
    tar_jet_asym2   = tar_lead_pt / tar_sublead_pt
    tru_jet_asym2   = tru_lead_pt / tru_sublead_pt
    pre_jet_asym2   = p_lead_pt / p_sublead_pt
    bin_start = min(min(pre_jet_asym2),min(tru_jet_asym2),min(tar_jet_asym2))
    bin_stop = max(max(pre_jet_asym2),max(tru_jet_asym2),max(tar_jet_asym2))
    print(bin_start, bin_stop)
    n_bins = 150
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # plot 2
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tru, bins, _   = ax[0].hist(tru_jet_asym2,bins=bins,histtype='step',color='gold',lw=2,label='Truth Jets')
    freq_tar, bins, _   = ax[0].hist(tar_jet_asym2,bins=bins,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(pre_jet_asym2,bins=bins,histtype='step',color='red',lw=2,label='CNN Jets')
    # stat. unc sqrt(counts)
    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)
    ax[0].errorbar(bin_centers, freq_tru,  yerr=tru_errors, color='gold', ls='none')
    ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')
    ratio_pred_tru = np.divide(freq_pre, freq_tru, out=np.zeros_like(freq_pre), where=freq_tru != 0)
    ratio_tar_tru  = np.divide(freq_tar, freq_tru, out=np.zeros_like(freq_tar), where=freq_tru != 0)
    ax[1].hlines(y=1, xmin=bin_start, xmax=bin_stop, color='gold', linestyle='-', lw=1.6)

    # uncertainty on the ratio
    ratio_pred_erro = np.sqrt((pre_errors / freq_tru) ** 2 + (freq_pre * tru_errors / freq_tru**2)**2)
    ratio_tar_erro = np.sqrt((tar_errors / freq_tru) ** 2 + (freq_tar * tru_errors / freq_tru**2)**2)
    ax[1].fill_between(bin_centers[1:-1], ratio_pred_tru[1:-1] - ratio_pred_erro[1:-1], ratio_pred_tru[1:-1] + ratio_pred_erro[1:-1], alpha=0.5, edgecolor='crimson', facecolor='red')
    ax[1].fill_between(bin_centers[1:-1], ratio_tar_tru[1:-1] - ratio_tar_erro[1:-1], ratio_tar_tru[1:-1] + ratio_tar_erro[1:-1], alpha=0.5, edgecolor='forestgreen', facecolor='green')

    tar_line =  matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
    pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    tru_line =  matplotlib.lines.Line2D([0], [0], color='gold', lw=3)  
    ax[0].legend([tru_line,tar_line,pre_line], ['Truth jets','Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax[0].set(yscale='log',ylabel='Number of events')
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[1].set(xlabel='Leading/Subleading Jet $p_T$ Ratio',ylim=(0,2))
    f.savefig(save_folder + f'/jet_pt_asym2_total_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    # plot 2b
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tru, bins, _   = ax[0].hist(tru_jet_asym2,bins=bins,weights=total_evt_weight,histtype='step',color='gold',lw=2,label='Truth Jets')
    freq_tar, bins, _   = ax[0].hist(tar_jet_asym2,bins=bins,weights=total_evt_weight,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(pre_jet_asym2,bins=bins,weights=total_evt_weight,histtype='step',color='red',lw=2,label='CNN Jets')
    # stat. unc sqrt(counts)
    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)
    # ax[0].errorbar(bin_centers, freq_tru,  yerr=tru_errors, color='gold', ls='none')
    # ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    # ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')
    ratio_pred_tru = np.divide(freq_pre, freq_tru, out=np.zeros_like(freq_pre), where=freq_tru != 0)
    ratio_tar_tru  = np.divide(freq_tar, freq_tru, out=np.zeros_like(freq_tar), where=freq_tru != 0)
    ax[1].hlines(y=1, xmin=bin_start, xmax=bin_stop, color='gold', linestyle='-', lw=1.6)

    # uncertainty on the ratio
    ratio_pred_erro = np.sqrt((pre_errors / freq_tru) ** 2 + (freq_pre * tru_errors / freq_tru**2)**2)
    ratio_tar_erro = np.sqrt((tar_errors / freq_tru) ** 2 + (freq_tar * tru_errors / freq_tru**2)**2)
    ax[1].fill_between(bin_centers[1:-1], ratio_pred_tru[1:-1] - ratio_pred_erro[1:-1], ratio_pred_tru[1:-1] + ratio_pred_erro[1:-1], alpha=0.5, edgecolor='crimson', facecolor='red')
    ax[1].fill_between(bin_centers[1:-1], ratio_tar_tru[1:-1] - ratio_tar_erro[1:-1], ratio_tar_tru[1:-1] + ratio_tar_erro[1:-1], alpha=0.5, edgecolor='forestgreen', facecolor='green')

    tar_line =  matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
    pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    tru_line =  matplotlib.lines.Line2D([0], [0], color='gold', lw=3)  
    ax[0].legend([tru_line,tar_line,pre_line], ['Truth jets','Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax[0].set(yscale='log',ylabel='Number of events (weighted)')
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[1].set(xlabel='Leading/Subleading Jet $p_T$ Ratio',ylim=(0,10))
    f.savefig(save_folder + f'/jet_pt_asym2_weight_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()




if jet_thresh_cut:
    print(f"Plotting leading jet pT in each event: {len(event_tar_pt)} events, {len(event_p_pt)} events, {len(event_tru_pt)} events")
    tar_lead_pt = np.array([max(x) for x in event_tar_pt])
    tru_lead_pt = np.array([max(x) for x in event_tru_pt])
    p_lead_pt = np.array([max(x) for x in event_p_pt])
    mask_99 = (tar_lead_pt > 1.0) & (tru_lead_pt > 1.0) & (p_lead_pt > 1.0)
    tar_lead_pt = tar_lead_pt[mask_99]
    tru_lead_pt = tru_lead_pt[mask_99]
    p_lead_pt = p_lead_pt[mask_99]
    total_evt_weight = np.array(total_evt_weight)
    evt_weights = total_evt_weight[mask_99]


    print("bin start options:", min(p_lead_pt),min(tar_lead_pt),min(tru_lead_pt))
    bin_stop = max(max(p_lead_pt),max(tar_lead_pt),max(tru_lead_pt))
    bin_start = 10
    # bin_start = -21
    bin_stop = np.ceil(bin_stop / 10) * 10
    bin_width = 20
    n_bins = int(np.ceil((bin_stop - bin_start) / bin_width))
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    freq_tru, bins, _   = ax.hist(tru_lead_pt,bins=bins,weights=evt_weights,histtype='step',color='gold',lw=1.5,label='Truth')
    freq_tar, bins, _   = ax.hist(tar_lead_pt,bins=bins,weights=evt_weights,histtype='step',color='green',lw=1.5,label='AKT4EMTopo')
    freq_pre, bins, _   = ax.hist(p_lead_pt,bins=bins,weights=evt_weights,histtype='step',color='red',lw=1.5,label='CaloJetSSD')

    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)

    ax.set(xlabel="Leading jet " + r"$p_T$ [GeV]",ylabel='Weighted Events',xlim=(0,bin_stop),ylim=(1e-6, max(max(freq_tru),max(freq_tar),max(freq_pre))*25),yscale='log')
    ax.legend(bbox_to_anchor=(0.7,0.6), loc="lower left",fontsize='small')
    ax.text(0.04,0.93, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold',transform=ax.transAxes)
    ax.text(0.17,0.93, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic',transform=ax.transAxes)
    ax.text(0.04,0.89, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12,transform=ax.transAxes)
    ax.text(0.04,0.855, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12,transform=ax.transAxes)

    f.savefig(save_folder + f'/jet_lead_pt_thresh1a.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    threshold_axis = np.linspace(0,bin_stop,num=1000)
    tru_pass_frac,akt_pass_frac,cnn_pass_frac = [],[],[]
    for threshold in threshold_axis:
        tru_pass_frac.append(len(tru_lead_pt[tru_lead_pt>threshold]) / len(tru_lead_pt))
        akt_pass_frac.append(len(tar_lead_pt[tar_lead_pt>threshold]) / len(tar_lead_pt))
        cnn_pass_frac.append(len(p_lead_pt[p_lead_pt>threshold]) / len(p_lead_pt))


    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(threshold_axis,tru_pass_frac,color='gold',lw=2,label='Truth jets')
    ax.plot(threshold_axis,akt_pass_frac,color='green',lw=2,label='AKT4EMTopo jets')
    ax.plot(threshold_axis,cnn_pass_frac,color='red',lw=2,label='CNN jets')

    ax.legend(loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax.set(xlim=(0,bin_stop),ylim=(0,1.1),ylabel='Fraction of weighted events',xlabel=f'Leading jet pt threshold [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh1b.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(threshold_axis,tru_pass_frac,color='gold',lw=2,label='Truth jets')
    ax.plot(threshold_axis,akt_pass_frac,color='green',lw=2,label='AKT4EMTopo jets')
    ax.plot(threshold_axis,cnn_pass_frac,color='red',lw=2,label='CNN jets')

    ax.legend(loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax.set(xlim=(0,bin_stop),yscale='log',ylabel='Fraction of weighted events',xlabel=f'Leading jet pt threshold [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh1b_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    print("Doing cut-for-cut analysis")
    print("For a given cut in AKT jets, what cut gives equivalent fraction in CNN jets")
    akt_cuts = [50,75,100,125,150,200,250,300,400,500]
    akt_fracs = np.interp(akt_cuts,threshold_axis,akt_pass_frac)
    print(akt_cuts)
    print(akt_fracs)
    cnn_cuts = np.interp(-akt_fracs,-np.array(cnn_pass_frac),-np.array(threshold_axis))
    print(-cnn_cuts)

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(akt_cuts,-cnn_cuts,color='red',marker='o',markersize=5,lw=2,ls='--')
    ax.plot(akt_cuts,akt_cuts,color='green',lw=1,ls='-')
    ax.set(xlabel='AKT4EMTopo leading jet cut [GeV]',ylabel=f'CNN leading jet cut [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh1c.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    akt_cuts = np.array(akt_cuts)
    cnn_cuts = np.array(cnn_cuts)
    ax.plot(akt_cuts,-cnn_cuts / akt_cuts,color='red',marker='o',markersize=5,lw=2,ls='--')
    ax.plot(akt_cuts,np.ones_like(akt_cuts),color='green',lw=1,ls='-')
    ax.set(xlabel='AKT4EMTopo leading jet cut [GeV]',ylabel=f'Ratio CNN / AKT4EMTopo leading jet cut [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh1e.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    tru_cuts = [50,75,100,125,150,200,250,300,400,500]
    tru_fracs = np.interp(tru_cuts,threshold_axis,tru_pass_frac)
    print(tru_cuts)
    print(tru_fracs)
    akt_cuts = np.interp(-tru_fracs,-np.array(akt_pass_frac),-np.array(threshold_axis))
    print(-akt_cuts)
    cnn_cuts = np.interp(-tru_fracs,-np.array(cnn_pass_frac),-np.array(threshold_axis))
    print(-cnn_cuts)

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(tru_cuts,-akt_cuts,color='green',marker='o',markersize=5,lw=2,ls='--',label='AKT4EMTopo')
    ax.plot(tru_cuts,-cnn_cuts,color='red',marker='o',markersize=5,lw=2,ls='--',label='CNN')
    ax.plot(tru_cuts,tru_cuts,color='gold',lw=1,ls='-')
    ax.set(xlabel='Truth leading jet cut [GeV]',ylabel=f'X leading jet cut [GeV]')
    ax.legend()
    f.savefig(save_folder + f'/jet_lead_pt_thresh1d.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    tru_cuts = np.array(tru_cuts)
    akt_cuts = np.array(akt_cuts)
    cnn_cuts = np.array(cnn_cuts)
    ax.plot(tru_cuts,-cnn_cuts / tru_cuts,color='red',marker='o',markersize=5,lw=2,ls='--',label='CNN')
    ax.plot(tru_cuts,-akt_cuts / tru_cuts,color='green',marker='o',markersize=5,lw=2,ls='--',label='AKT4EMTopo')
    ax.plot(tru_cuts,np.ones_like(tru_cuts),color='gold',lw=1,ls='-')
    ax.set(xlabel='Truth leading jet cut [GeV]',ylabel=f'Ratio X / Truth leading jet cut [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh1f.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()




if jet_thresh_cut2:
    print(f"Plotting subleading jet pT in each event: {len(event_tar_pt)} events, {len(event_p_pt)} events, {len(event_tru_pt)} events")
    tar_sublead_pt = np.array([nth_leading_jet_pt(x,2) for x in event_tar_pt])
    tru_sublead_pt = np.array([nth_leading_jet_pt(x,2) for x in event_tru_pt])
    p_sublead_pt = np.array([nth_leading_jet_pt(x,2) for x in event_p_pt])
    total_evt_weight = np.array(total_evt_weight)
    tar_evt_weight= total_evt_weight[~np.isnan(tar_sublead_pt)]
    tar_sublead_pt = tar_sublead_pt[~np.isnan(tar_sublead_pt)]
    tru_evt_weight= total_evt_weight[~np.isnan(tru_sublead_pt)]
    tru_sublead_pt = tru_sublead_pt[~np.isnan(tru_sublead_pt)]
    p_evt_weight= total_evt_weight[~np.isnan(p_sublead_pt)]
    p_sublead_pt = p_sublead_pt[~np.isnan(p_sublead_pt)]

    bin_start = min(min(p_sublead_pt),min(tar_sublead_pt),min(tru_sublead_pt))
    bin_stop = max(max(p_sublead_pt),max(tar_sublead_pt),max(tru_sublead_pt))
    print(bin_start, bin_stop)
    bin_start = (np.floor(bin_start / 10) * 10) - 1
    bin_stop = np.ceil(bin_stop / 10) * 10
    bin_width = 20
    n_bins = int(np.ceil((bin_stop - bin_start) / bin_width))
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ###########
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    freq_tru, bins, _   = ax.hist(tru_sublead_pt,bins=bins,weights=tru_evt_weight,histtype='step',color='gold',lw=2,label='Truth Jets')
    freq_tar, bins, _   = ax.hist(tar_sublead_pt,bins=bins,weights=tar_evt_weight,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax.hist(p_sublead_pt,bins=bins,weights=p_evt_weight,histtype='step',color='red',lw=2,label='CNN Jets')

    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)

    ax.legend(loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax.set(yscale='log', xlim=(0,bin_stop),ylabel='Weighted events',xlabel=f'Subleading jet pt [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh2a.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    threshold_axis = np.linspace(0,bin_stop,num=1000)
    tru_pass_frac,akt_pass_frac,cnn_pass_frac = [],[],[]
    for threshold in threshold_axis:
        tru_pass_frac.append(len(tru_sublead_pt[tru_sublead_pt>threshold]) / len(tru_sublead_pt))
        akt_pass_frac.append(len(tar_sublead_pt[tar_sublead_pt>threshold]) / len(tar_sublead_pt))
        cnn_pass_frac.append(len(p_sublead_pt[p_sublead_pt>threshold]) / len(p_sublead_pt))


    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(threshold_axis,tru_pass_frac,color='gold',lw=2,label='Truth jets')
    ax.plot(threshold_axis,akt_pass_frac,color='green',lw=2,label='AKT4EMTopo jets')
    ax.plot(threshold_axis,cnn_pass_frac,color='red',lw=2,label='CNN jets')

    ax.legend(loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax.set(xlim=(0,bin_stop),ylim=(0,1.1),ylabel='Fraction of weighted events',xlabel=f'Subleading jet pt threshold [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh2b.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(threshold_axis,tru_pass_frac,color='gold',lw=2,label='Truth jets')
    ax.plot(threshold_axis,akt_pass_frac,color='green',lw=2,label='AKT4EMTopo jets')
    ax.plot(threshold_axis,cnn_pass_frac,color='red',lw=2,label='CNN jets')

    ax.legend(loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax.set(xlim=(0,bin_stop),yscale='log',ylabel='Fraction of weighted events',xlabel=f'Subleading jet pt threshold [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh2b_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    print("Doing cut-for-cut analysis")
    print("For a given cut in AKT jets, what cut gives equivalent fraction in CNN jets")
    akt_cuts = [40,50,60,70,80,90,100,125,150,175,200,250,300,350,400]
    akt_fracs = np.interp(akt_cuts,threshold_axis,akt_pass_frac)
    print(akt_cuts)
    print(akt_fracs)
    cnn_cuts = np.interp(-akt_fracs,-np.array(cnn_pass_frac),-np.array(threshold_axis))
    print(-cnn_cuts)

    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(akt_cuts,-cnn_cuts,color='red',marker='o',markersize=5,lw=2,ls='--')
    ax.plot(akt_cuts,akt_cuts,color='green',lw=1,ls='-')
    ax.set(xlabel='AKT4EMTopo subleading jet cut [GeV]',ylabel=f'CNN subleading jet cut [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh2c.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    tru_cuts = [40,50,60,70,80,90,100,125,150,175,200,250,300,350,400]
    tru_fracs = np.interp(tru_cuts,threshold_axis,tru_pass_frac)
    print(tru_cuts)
    print(tru_fracs)
    akt_cuts = np.interp(-tru_fracs,-np.array(akt_pass_frac),-np.array(threshold_axis))
    print(-akt_cuts)
    cnn_cuts = np.interp(-tru_fracs,-np.array(cnn_pass_frac),-np.array(threshold_axis))
    print(-cnn_cuts)

    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(tru_cuts,-akt_cuts,color='green',marker='o',markersize=5,lw=2,ls='--',label='AKT4EMTopo')
    ax.plot(tru_cuts,-cnn_cuts,color='red',marker='o',markersize=5,lw=2,ls='--',label='CNN')
    ax.plot(tru_cuts,tru_cuts,color='gold',lw=1,ls='-')
    ax.set(xlabel='Truth subleading jet cut [GeV]',ylabel=f'X subleading jet cut [GeV]')
    ax.legend()
    f.savefig(save_folder + f'/jet_lead_pt_thresh2d.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()




if jet_thresh_cut3:
    print(f"Plotting 3rd leading jet pT in each event: {len(event_tar_pt)} events, {len(event_p_pt)} events, {len(event_tru_pt)} events")
    tar_3lead_pt = np.array([nth_leading_jet_pt(x,3) for x in event_tar_pt])
    tru_3lead_pt = np.array([nth_leading_jet_pt(x,3) for x in event_tru_pt])
    p_3lead_pt = np.array([nth_leading_jet_pt(x,3) for x in event_p_pt])
    total_evt_weight = np.array(total_evt_weight)
    tar_evt_weight= total_evt_weight[~np.isnan(tar_3lead_pt)]
    tar_3lead_pt = tar_3lead_pt[~np.isnan(tar_3lead_pt)]
    tru_evt_weight= total_evt_weight[~np.isnan(tru_3lead_pt)]
    tru_3lead_pt = tru_3lead_pt[~np.isnan(tru_3lead_pt)]
    p_evt_weight= total_evt_weight[~np.isnan(p_3lead_pt)]
    p_3lead_pt = p_3lead_pt[~np.isnan(p_3lead_pt)]


    # bin_start = min(min(p_3lead_pt),min(tar_3lead_pt),min(tru_3lead_pt))
    # bin_start = (np.floor(bin_start / 10) * 10) - 1
    bin_start = 5
    bin_stop = max(max(p_3lead_pt),max(tar_3lead_pt),max(tru_3lead_pt))
    print(bin_start, bin_stop)
    print("Bin start options: ", min(p_3lead_pt),min(tar_3lead_pt),min(tru_3lead_pt))
    print("Bin start options: ", max(p_3lead_pt),max(tar_3lead_pt),max(tru_3lead_pt))
    bin_stop = np.ceil(bin_stop / 10) * 10
    bin_stop = 500
    bin_width = 5
    n_bins = int(np.ceil((bin_stop - bin_start) / bin_width))
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ###########
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    freq_tru, bins, _   = ax.hist(tru_3lead_pt,bins=bins,weights=tru_evt_weight,histtype='step',color='gold',lw=2,label='Truth Jets')
    freq_tar, bins, _   = ax.hist(tar_3lead_pt,bins=bins,weights=tar_evt_weight,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax.hist(p_3lead_pt,bins=bins,weights=p_evt_weight,histtype='step',color='red',lw=2,label='CNN Jets')

    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)

    ax.legend(loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax.set(yscale='log', ylim=(1e-8,max(max(freq_tru),max(freq_tar),max(freq_pre))*12), xlim=(0,bin_stop),ylabel='Weighted events',xlabel=f'3rd leading jet pt [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh3a.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    threshold_axis = np.linspace(0,bin_stop,num=1000)
    tru_pass_frac,akt_pass_frac,cnn_pass_frac = [],[],[]
    for threshold in threshold_axis:
        tru_pass_frac.append(len(tru_3lead_pt[tru_3lead_pt>threshold]) / len(tru_3lead_pt))
        akt_pass_frac.append(len(tar_3lead_pt[tar_3lead_pt>threshold]) / len(tar_3lead_pt))
        cnn_pass_frac.append(len(p_3lead_pt[p_3lead_pt>threshold]) / len(p_3lead_pt))


    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(threshold_axis,tru_pass_frac,color='gold',lw=2,label='Truth jets')
    ax.plot(threshold_axis,akt_pass_frac,color='green',lw=2,label='AKT4EMTopo jets')
    ax.plot(threshold_axis,cnn_pass_frac,color='red',lw=2,label='CNN jets')

    ax.legend(loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax.set(xlim=(0,bin_stop),ylim=(0,1.1),ylabel='Fraction of weighted events',xlabel=f'3rd leading jet pt threshold [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh3b.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(threshold_axis,tru_pass_frac,color='gold',lw=2,label='Truth jets')
    ax.plot(threshold_axis,akt_pass_frac,color='green',lw=2,label='AKT4EMTopo jets')
    ax.plot(threshold_axis,cnn_pass_frac,color='red',lw=2,label='CNN jets')

    ax.legend(loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax.set(xlim=(0,bin_stop),yscale='log',ylabel='Fraction of weighted events',xlabel=f'3rd leading jet pt threshold [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh3b_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    print("Doing cut-for-cut analysis")
    print("For a given cut in AKT jets, what cut gives equivalent fraction in CNN jets")
    akt_cuts = [30,40,50,60,70,80,90,100,125,150,175,200,225,250]
    akt_fracs = np.interp(akt_cuts,threshold_axis,akt_pass_frac)
    print(akt_cuts)
    print(akt_fracs)
    cnn_cuts = np.interp(-akt_fracs,-np.array(cnn_pass_frac),-np.array(threshold_axis))
    print(-cnn_cuts)

    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(akt_cuts,-cnn_cuts,color='red',marker='o',markersize=5,lw=2,ls='--')
    ax.plot(akt_cuts,akt_cuts,color='green',lw=1,ls='-')
    ax.set(xlabel='AKT4EMTopo 3rd leading jet cut [GeV]',ylabel=f'CNN 3rd leading jet cut [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh3c.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    akt_cuts = np.array(akt_cuts)
    cnn_cuts = np.array(cnn_cuts)
    ax.plot(akt_cuts,-cnn_cuts / akt_cuts,color='red',marker='o',markersize=5,lw=2,ls='--')
    ax.plot(akt_cuts,np.ones_like(akt_cuts),color='green',lw=1,ls='-')
    ax.set(xlabel='AKT4EMTopo leading jet cut [GeV]',ylabel=f'Ratio CNN / AKT4EMTopo leading jet cut [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh3e.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    tru_cuts = [30,40,50,60,70,80,90,100,125,150,175,200,225,250]
    tru_fracs = np.interp(tru_cuts,threshold_axis,tru_pass_frac)
    print(tru_cuts)
    print(tru_fracs)
    akt_cuts = np.interp(-tru_fracs,-np.array(akt_pass_frac),-np.array(threshold_axis))
    print(-akt_cuts)
    cnn_cuts = np.interp(-tru_fracs,-np.array(cnn_pass_frac),-np.array(threshold_axis))
    print(-cnn_cuts)

    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(tru_cuts,-akt_cuts,color='green',marker='o',markersize=5,lw=2,ls='--',label='AKT4EMTopo')
    ax.plot(tru_cuts,-cnn_cuts,color='red',marker='o',markersize=5,lw=2,ls='--',label='CNN')
    ax.plot(tru_cuts,tru_cuts,color='gold',lw=1,ls='-')
    ax.set(xlabel='Truth 3rd leading jet cut [GeV]',ylabel=f'X 3rd leading jet cut [GeV]')
    ax.legend()
    f.savefig(save_folder + f'/jet_lead_pt_thresh3d.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    tru_cuts = np.array(tru_cuts)
    akt_cuts = np.array(akt_cuts)
    cnn_cuts = np.array(cnn_cuts)
    ax.plot(tru_cuts,-cnn_cuts / tru_cuts,color='red',marker='o',markersize=5,lw=2,ls='--',label='CNN')
    ax.plot(tru_cuts,-akt_cuts / tru_cuts,color='green',marker='o',markersize=5,lw=2,ls='--',label='AKT4EMTopo')
    ax.plot(tru_cuts,np.ones_like(tru_cuts),color='gold',lw=1,ls='-')
    ax.set(xlabel='Truth leading jet cut [GeV]',ylabel=f'Ratio X / Truth leading jet cut [GeV]')
    f.savefig(save_folder + f'/jet_lead_pt_thresh3f.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()























if jet_thresh_cut1b:

    total_tar_pt    = load_object(metrics_folder+"/tarboxes_pt.pkl")
    total_tar_eta   = load_object(metrics_folder+"/tarboxes_eta.pkl")
    total_tru_pt    = load_object(metrics_folder+"/truboxes_pt.pkl")
    total_tru_eta   = load_object(metrics_folder+"/truboxes_eta.pkl")
    total_p_pt      = load_object(metrics_folder+"/pboxes_pt.pkl")
    total_p_eta     = load_object(metrics_folder+"/pboxes_eta.pkl")

    # find leading jet & eta
    lead_tar_jet_idx  = np.array([np.argmax(event_tar_pt) for event_tar_pt in total_tar_pt])
    lead_tar_jet_pt   = np.array([total_tar_pt[i][lead_tar_jet_idx[i]] for i in range(len(lead_tar_jet_idx))])
    lead_tar_jet_eta  = np.array([total_tar_eta[i][lead_tar_jet_idx[i]] for i in range(len(lead_tar_jet_idx))])
    
    lead_tru_jet_idx  = np.array([np.argmax(event_tru_pt) for event_tru_pt in total_tru_pt])
    lead_tru_jet_pt   = np.array([total_tru_pt[i][lead_tru_jet_idx[i]] for i in range(len(lead_tru_jet_idx))])
    lead_tru_jet_eta  = np.array([total_tru_eta[i][lead_tru_jet_idx[i]] for i in range(len(lead_tru_jet_idx))])
    
    lead_pre_jet_idx  = np.array([np.argmax(event_pre_pt) for event_pre_pt in total_p_pt])
    lead_pre_jet_pt   = np.array([total_p_pt[i][lead_pre_jet_idx[i]] for i in range(len(lead_pre_jet_idx))])
    lead_pre_jet_eta  = np.array([total_p_eta[i][lead_pre_jet_idx[i]] for i in range(len(lead_pre_jet_idx))])
    
    # mask out events with truth leding jet outside central
    lead_tar_jet_pt = lead_tar_jet_pt[np.abs(lead_tru_jet_eta) < 0.8]
    lead_tru_jet_pt = lead_tru_jet_pt[np.abs(lead_tru_jet_eta) < 0.8]
    lead_pre_jet_pt = lead_pre_jet_pt[np.abs(lead_tru_jet_eta) < 0.8]
    total_evt_weight = np.array(total_evt_weight)
    evt_weights = total_evt_weight[np.abs(lead_tru_jet_eta) < 0.8]

    print(f"Plotting leading jet pT in each event: {len(event_tar_pt)} events, {len(event_p_pt)} events, {len(event_tru_pt)} events")

    mask_99 = (lead_tar_jet_pt > 1.0) & (lead_tru_jet_pt > 1.0) & (lead_pre_jet_pt > 1.0)
    tar_lead_pt = lead_tar_jet_pt[mask_99]
    tru_lead_pt = lead_tru_jet_pt[mask_99]
    p_lead_pt = lead_pre_jet_pt[mask_99]
    evt_weights = evt_weights[mask_99]


    print("bin start options:", min(p_lead_pt),min(tar_lead_pt),min(tru_lead_pt))
    bin_stop = max(max(p_lead_pt),max(tar_lead_pt),max(tru_lead_pt))
    bin_start = 10
    # bin_start = -21
    bin_stop = np.ceil(bin_stop / 10) * 10
    bin_width = 20
    n_bins = int(np.ceil((bin_stop - bin_start) / bin_width))
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    freq_tru, bins, _   = ax.hist(tru_lead_pt,bins=bins,weights=evt_weights,histtype='step',color='gold',lw=1.5,label='Truth')
    freq_tar, bins, _   = ax.hist(tar_lead_pt,bins=bins,weights=evt_weights,histtype='step',color='green',lw=1.5,label='AKT4EMTopo')
    freq_pre, bins, _   = ax.hist(p_lead_pt,bins=bins,weights=evt_weights,histtype='step',color='red',lw=1.5,label='CaloJetSSD')

    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)

    ax.set(xlabel="Leading jet " + r"$p_T$ [GeV]",ylabel='Weighted Events',xlim=(0,bin_stop),ylim=(1e-6, max(max(freq_tru),max(freq_tar),max(freq_pre))*25),yscale='log')
    ax.legend(bbox_to_anchor=(0.7,0.6), loc="lower left",fontsize='small')
    ax.text(0.04,0.93, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold',transform=ax.transAxes)
    ax.text(0.17,0.93, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic',transform=ax.transAxes)
    ax.text(0.04,0.89, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12,transform=ax.transAxes)
    ax.text(0.04,0.855, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<0.8$",fontfamily='sans-serif',fontsize=12,transform=ax.transAxes)

    f.savefig(save_folder + f'/central_jet_lead_pt_thresh1a.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    threshold_axis = np.linspace(0,bin_stop,num=1000)
    tru_pass_frac,akt_pass_frac,cnn_pass_frac = [],[],[]
    for threshold in threshold_axis:
        tru_pass_frac.append(len(tru_lead_pt[tru_lead_pt>threshold]) / len(tru_lead_pt))
        akt_pass_frac.append(len(tar_lead_pt[tar_lead_pt>threshold]) / len(tar_lead_pt))
        cnn_pass_frac.append(len(p_lead_pt[p_lead_pt>threshold]) / len(p_lead_pt))


    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(threshold_axis,tru_pass_frac,color='gold',lw=2,label='Truth jets')
    ax.plot(threshold_axis,akt_pass_frac,color='green',lw=2,label='AKT4EMTopo jets')
    ax.plot(threshold_axis,cnn_pass_frac,color='red',lw=2,label='CNN jets')

    ax.legend(loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax.set(xlim=(0,bin_stop),ylim=(0,1.1),ylabel='Fraction of weighted events',xlabel=f'Leading jet pt threshold [GeV]')
    f.savefig(save_folder + f'/central_jet_lead_pt_thresh1b.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(threshold_axis,tru_pass_frac,color='gold',lw=2,label='Truth jets')
    ax.plot(threshold_axis,akt_pass_frac,color='green',lw=2,label='AKT4EMTopo jets')
    ax.plot(threshold_axis,cnn_pass_frac,color='red',lw=2,label='CNN jets')

    ax.legend(loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax.set(xlim=(0,bin_stop),yscale='log',ylabel='Fraction of weighted events',xlabel=f'Leading jet pt threshold [GeV]')
    f.savefig(save_folder + f'/central_jet_lead_pt_thresh1b_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    print("Doing cut-for-cut analysis")
    print("For a given cut in AKT jets, what cut gives equivalent fraction in CNN jets")
    akt_cuts = [50,75,100,125,150,200,250,300,400,500]
    akt_fracs = np.interp(akt_cuts,threshold_axis,akt_pass_frac)
    print(akt_cuts)
    print(akt_fracs)
    cnn_cuts = np.interp(-akt_fracs,-np.array(cnn_pass_frac),-np.array(threshold_axis))
    print(-cnn_cuts)

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(akt_cuts,-cnn_cuts,color='red',marker='o',markersize=5,lw=2,ls='--')
    ax.plot(akt_cuts,akt_cuts,color='green',lw=1,ls='-')
    ax.set(xlabel='AKT4EMTopo leading jet cut [GeV]',ylabel=f'CNN leading jet cut [GeV]')
    f.savefig(save_folder + f'/central_jet_lead_pt_thresh1c.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    akt_cuts = np.array(akt_cuts)
    cnn_cuts = np.array(cnn_cuts)
    ax.plot(akt_cuts,-cnn_cuts / akt_cuts,color='red',marker='o',markersize=5,lw=2,ls='--')
    ax.plot(akt_cuts,np.ones_like(akt_cuts),color='green',lw=1,ls='-')
    ax.set(xlabel='AKT4EMTopo leading jet cut [GeV]',ylabel=f'Ratio CNN / AKT4EMTopo leading jet cut')
    f.savefig(save_folder + f'/central_jet_lead_pt_thresh1e.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    tru_cuts = [50,75,100,125,150,200,250,300,400,500]
    tru_fracs = np.interp(tru_cuts,threshold_axis,tru_pass_frac)
    print(tru_cuts)
    print(tru_fracs)
    akt_cuts = np.interp(-tru_fracs,-np.array(akt_pass_frac),-np.array(threshold_axis))
    print(-akt_cuts)
    cnn_cuts = np.interp(-tru_fracs,-np.array(cnn_pass_frac),-np.array(threshold_axis))
    print(-cnn_cuts)

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(tru_cuts,-akt_cuts,color='green',marker='o',markersize=5,lw=2,ls='--',label='AKT4EMTopo')
    ax.plot(tru_cuts,-cnn_cuts,color='red',marker='o',markersize=5,lw=2,ls='--',label='CNN')
    ax.plot(tru_cuts,tru_cuts,color='gold',lw=1,ls='-')
    ax.set(xlabel='Truth leading jet cut [GeV]',ylabel=f'X leading jet cut [GeV]')
    ax.legend()
    f.savefig(save_folder + f'/central_jet_lead_pt_thresh1d.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    tru_cuts = np.array(tru_cuts)
    akt_cuts = np.array(akt_cuts)
    cnn_cuts = np.array(cnn_cuts)
    ax.plot(tru_cuts,-cnn_cuts / tru_cuts,color='red',marker='o',markersize=5,lw=2,ls='--',label='CNN')
    ax.plot(tru_cuts,-akt_cuts / tru_cuts,color='green',marker='o',markersize=5,lw=2,ls='--',label='AKT4EMTopo')
    ax.plot(tru_cuts,np.ones_like(tru_cuts),color='gold',lw=1,ls='-')
    ax.legend()
    ax.set(xlabel='Truth leading jet cut [GeV]',ylabel=f'Ratio X / Truth leading jet cut')
    ax.text(60,1.02, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    ax.text(120,1.02, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    ax.text(60,1.005, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    ax.text(60,0.985, f"MC21 {proc[:2]} " +  r"$p_T > 20\,$GeV, $|\eta|<0.8$",fontfamily='sans-serif',fontsize=12)
    f.savefig(save_folder + f'/central_jet_lead_pt_thresh1f.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()








if jet_thresh_cut3b:
    print(f"Plotting 3rd leading jet pT in each event: {len(event_tar_pt)} events, {len(event_p_pt)} events, {len(event_tru_pt)} events")

    total_tar_pt    = load_object(metrics_folder+"/tarboxes_pt.pkl")
    total_tar_eta   = load_object(metrics_folder+"/tarboxes_eta.pkl")
    total_tar_phi      = load_object(metrics_folder+"/tarboxes_phi.pkl")
    total_tru_pt    = load_object(metrics_folder+"/truboxes_pt.pkl")
    total_tru_eta   = load_object(metrics_folder+"/truboxes_eta.pkl")
    total_p_pt      = load_object(metrics_folder+"/pboxes_pt.pkl")
    total_p_eta     = load_object(metrics_folder+"/pboxes_eta.pkl")

    # find subleading jet & eta
    sublead_tar_jet_pt  = np.zeros(len(total_tar_pt))
    sublead_tar_jet_eta = np.zeros(len(total_tar_pt))
    for j in range(len(total_tar_pt)):
        event_tar_pt = total_tar_pt[j]
        event_tar_eta = total_tar_eta[j]
        if len(event_tar_pt) < 3:
            continue
        else:
            _, idxs = torch.topk(event_tar_pt,3)
            sublead_tar_jet_idx = idxs[-1]
            sublead_tar_jet_pt[j]  = event_tar_pt[sublead_tar_jet_idx]
            sublead_tar_jet_eta[j] = event_tar_eta[sublead_tar_jet_idx]

    sublead_tru_jet_pt  = np.zeros(len(total_tru_pt))
    sublead_tru_jet_eta = np.zeros(len(total_tru_pt))
    for j in range(len(total_tru_pt)):
        event_tru_pt = total_tru_pt[j]
        event_tru_eta = total_tru_eta[j]
        if len(event_tru_pt) < 3:
            continue
        else:
            _, idxs = torch.topk(event_tru_pt,3)
            sublead_tru_jet_idx = idxs[-1]
            sublead_tru_jet_pt[j]  = event_tru_pt[sublead_tru_jet_idx]
            sublead_tru_jet_eta[j] = event_tru_eta[sublead_tru_jet_idx]

    sublead_pre_jet_pt  = np.zeros(len(total_p_pt))
    sublead_pre_jet_eta = np.zeros(len(total_p_pt))
    for j in range(len(total_p_pt)):
        event_pre_pt = total_p_pt[j]
        event_pre_eta = total_p_eta[j]
        if len(event_pre_pt) < 3:
            continue
        else:
            _, idxs = torch.topk(event_pre_pt,3)
            sublead_pre_jet_idx = idxs[-1]
            sublead_pre_jet_pt[j]  = event_pre_pt[sublead_pre_jet_idx]
            sublead_pre_jet_eta[j] = event_pre_eta[sublead_pre_jet_idx]
    
    # mask out events with truth leading jet outside central
    mask_99 = (sublead_tru_jet_pt>1.0) & (sublead_tar_jet_pt>1.0) & (sublead_pre_jet_pt>1.0)
    sublead_tru_jet_pt1 = sublead_tru_jet_pt[(np.abs(sublead_tru_jet_eta) < 0.8) & mask_99]
    sublead_pre_jet_pt1 = sublead_pre_jet_pt[(np.abs(sublead_tru_jet_eta) < 0.8) & mask_99]
    sublead_tar_jet_pt1 = sublead_tar_jet_pt[(np.abs(sublead_tru_jet_eta) < 0.8) & mask_99]
    total_evt_weight = np.array(total_evt_weight)
    evt_weights = total_evt_weight[(np.abs(sublead_tru_jet_eta) < 0.8) & mask_99]
    print(torch.topk(torch.tensor(sublead_tru_jet_pt1),10,largest=False))
    print(torch.topk(torch.tensor(sublead_pre_jet_pt1),10,largest=False))
    print(torch.topk(torch.tensor(sublead_tar_jet_pt1),10,largest=False))

    bin_start = 5
    bin_stop = max(max(sublead_pre_jet_pt1),max(sublead_tru_jet_pt1),max(sublead_tar_jet_pt1))
    print(bin_start, bin_stop)
    bin_stop = np.ceil(bin_stop / 10) * 10
    bin_stop = 500
    bin_width = 5
    n_bins = int(np.ceil((bin_stop - bin_start) / bin_width))
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ###########
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    freq_tru, bins, _   = ax.hist(sublead_tru_jet_pt1,bins=bins,weights=evt_weights,histtype='step',color='gold',lw=2,label='Truth')
    freq_tar, bins, _   = ax.hist(sublead_tar_jet_pt1,bins=bins,weights=evt_weights,histtype='step',color='green',lw=2,label='AKT4EMTopo')
    freq_pre, bins, _   = ax.hist(sublead_pre_jet_pt1,bins=bins,weights=evt_weights,histtype='step',color='red',lw=2,label='CNN')

    tru_errors = np.sqrt(freq_tru)
    tar_errors = np.sqrt(freq_tar)
    pre_errors = np.sqrt(freq_pre)

    ax.legend(loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax.set(yscale='log', ylim=(1e-8,max(max(freq_tru),max(freq_tar),max(freq_pre))*12), xlim=(0,bin_stop),ylabel='Weighted events',xlabel=f'3rd leading jet pt [GeV]')
    f.savefig(save_folder + f'/central_jet_lead_pt_thresh3a.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    threshold_axis = np.linspace(0,bin_stop,num=1000)
    tru_pass_frac,akt_pass_frac,cnn_pass_frac = [],[],[]
    for threshold in threshold_axis:
        tru_pass_frac.append(len(sublead_tru_jet_pt1[sublead_tru_jet_pt1>threshold]) / len(sublead_tru_jet_pt1))
        akt_pass_frac.append(len(sublead_tar_jet_pt1[sublead_tar_jet_pt1>threshold]) / len(sublead_tar_jet_pt1))
        cnn_pass_frac.append(len(sublead_pre_jet_pt1[sublead_pre_jet_pt1>threshold]) / len(sublead_pre_jet_pt1))


    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(threshold_axis,tru_pass_frac,color='gold',lw=2,label='Truth')
    ax.plot(threshold_axis,akt_pass_frac,color='green',lw=2,label='AKT4EMTopo')
    ax.plot(threshold_axis,cnn_pass_frac,color='red',lw=2,label='CNN')
    ax.legend(loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax.set(xlim=(0,bin_stop),ylim=(0,1.1),ylabel='Fraction of weighted events',xlabel=f'3rd leading jet pt threshold [GeV]')
    f.savefig(save_folder + f'/central_jet_lead_pt_thresh3b.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(threshold_axis,tru_pass_frac,color='gold',lw=2,label='Truth')
    ax.plot(threshold_axis,akt_pass_frac,color='green',lw=2,label='AKT4EMTopo')
    ax.plot(threshold_axis,cnn_pass_frac,color='red',lw=2,label='CNN')
    ax.legend(loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 
    ax.set(xlim=(0,bin_stop),yscale='log',ylabel='Fraction of weighted events',xlabel=f'3rd leading jet pt threshold [GeV]')
    f.savefig(save_folder + f'/central_jet_lead_pt_thresh3b_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    print("Doing cut-for-cut analysis")
    print("For a given cut in AKT jets, what cut gives equivalent fraction in CNN jets")
    akt_cuts = [30,40,50,60,70,80,90,100,125,150,175,200,225,250]
    akt_fracs = np.interp(akt_cuts,threshold_axis,akt_pass_frac)
    print(akt_cuts)
    print(akt_fracs)
    cnn_cuts = np.interp(-akt_fracs,-np.array(cnn_pass_frac),-np.array(threshold_axis))
    print(-cnn_cuts)

    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(akt_cuts,-cnn_cuts,color='red',marker='o',markersize=5,lw=2,ls='--')
    ax.plot(akt_cuts,akt_cuts,color='green',lw=1,ls='-')
    ax.set(xlabel='AKT4EMTopo 3rd leading jet cut [GeV]',ylabel=f'CNN 3rd leading jet cut [GeV]')
    f.savefig(save_folder + f'/central_jet_lead_pt_thresh3c.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    akt_cuts = np.array(akt_cuts)
    cnn_cuts = np.array(cnn_cuts)
    ax.plot(akt_cuts,-cnn_cuts / akt_cuts,color='red',marker='o',markersize=5,lw=2,ls='--')
    ax.plot(akt_cuts,np.ones_like(akt_cuts),color='green',lw=1,ls='-')
    ax.set(xlabel='AKT4EMTopo leading jet cut [GeV]',ylabel=f'Ratio CNN / AKT4EMTopo leading jet cut')
    f.savefig(save_folder + f'/central_jet_lead_pt_thresh3e.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    tru_cuts = [30,40,50,60,70,80,90,100,125,150,175,200,225,250]
    tru_fracs = np.interp(tru_cuts,threshold_axis,tru_pass_frac)
    print(tru_cuts)
    print(tru_fracs)
    akt_cuts = np.interp(-tru_fracs,-np.array(akt_pass_frac),-np.array(threshold_axis))
    print(-akt_cuts)
    cnn_cuts = np.interp(-tru_fracs,-np.array(cnn_pass_frac),-np.array(threshold_axis))
    print(-cnn_cuts)

    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    ax.plot(tru_cuts,-akt_cuts,color='green',marker='o',markersize=5,lw=2,ls='--',label='AKT4EMTopo')
    ax.plot(tru_cuts,-cnn_cuts,color='red',marker='o',markersize=5,lw=2,ls='--',label='CNN')
    ax.plot(tru_cuts,tru_cuts,color='gold',lw=1,ls='-')
    ax.set(xlabel='Truth 3rd leading jet cut [GeV]',ylabel=f'X 3rd leading jet cut [GeV]')
    ax.legend()
    f.savefig(save_folder + f'/central_jet_lead_pt_thresh3d.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    ####
    f, ax = plt.subplots(1, 1, figsize=(9, 6)) 
    tru_cuts = np.array(tru_cuts)
    akt_cuts = np.array(akt_cuts)
    cnn_cuts = np.array(cnn_cuts)
    ax.plot(tru_cuts,-cnn_cuts / tru_cuts,color='red',marker='o',markersize=5,lw=2,ls='--',label='CNN')
    ax.plot(tru_cuts,-akt_cuts / tru_cuts,color='green',marker='o',markersize=5,lw=2,ls='--',label='AKT4EMTopo')
    ax.plot(tru_cuts,np.ones_like(tru_cuts),color='gold',lw=1,ls='-')
    ax.set(xlabel='Truth leading jet cut [GeV]',ylabel=f'Ratio X / Truth leading jet cut')
    f.savefig(save_folder + f'/central_jet_lead_pt_thresh3f.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()




