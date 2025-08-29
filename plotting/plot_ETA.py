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
# date = "20250406-23"
# proc = "ttbar_test"
# date = "20250407-14"

metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/{proc}/{date}/box_metrics/"
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/{proc}/{date}/jet_kin/"
if not os.path.exists(save_folder): os.makedirs(save_folder)
image_format = "png"

print("=======================================================================================================")
print(f"Loading all jets from\n{metrics_folder}")
print("=======================================================================================================\n")

# total_tar_pt      = np.concatenate(load_object(metrics_folder+"/tarboxes_pt.pkl"))
# total_p_pt      = np.concatenate(load_object(metrics_folder+"/pboxes_pt.pkl"))
total_tar_eta      = np.concatenate(load_object(metrics_folder+"/tarboxes_eta.pkl"))
total_tru_eta      = np.concatenate(load_object(metrics_folder+"/truboxes_eta.pkl"))
total_p_eta      = np.concatenate(load_object(metrics_folder+"/pboxes_eta.pkl"))
total_p_scr      = np.concatenate(load_object(metrics_folder+"/pboxes_scores.pkl"))
# IOU matched
# match_tar_pt      = np.concatenate(load_object(metrics_folder+"/tarboxes_matched_pt.pkl"))
# match_p_pt      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_pt.pkl"))
match_tar_eta      = np.concatenate(load_object(metrics_folder+"/tarboxes_matched_eta.pkl"))
match_p_eta      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_eta.pkl"))
match_p_scr      = np.concatenate(load_object(metrics_folder+"/pboxes_matched_scr.pkl"))
# unmatch_tar_pt    = np.concatenate(load_object(metrics_folder+"/tarboxes_unmatched_pt.pkl"))
# unmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_pt.pkl"))
unmatch_tar_eta    = np.concatenate(load_object(metrics_folder+"/tarboxes_unmatched_eta.pkl"))
unmatch_p_eta    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_eta.pkl"))
unmatch_p_scr    = np.concatenate(load_object(metrics_folder+"/pboxes_unmatched_scr.pkl"))

## dR matched
# dRmatch_tar_pt    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRmatched_pt.pkl"))
# dRmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_pt.pkl"))
dRmatch_tar_eta    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRmatched_eta.pkl"))
dRmatch_p_eta    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_eta.pkl"))
dRmatch_p_scr    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_scr.pkl"))
# dRunmatch_tar_pt  = np.concatenate(load_object(metrics_folder+"/tarboxes_dRunmatched_pt.pkl"))
# dRunmatch_p_pt  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_pt.pkl"))
dRunmatch_tar_eta  = np.concatenate(load_object(metrics_folder+"/tarboxes_dRunmatched_eta.pkl"))
dRunmatch_p_eta  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_eta.pkl"))
dRunmatch_p_scr  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_scr.pkl"))

#dR truth matched
dRtruthmatch_tru_eta    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthmatched_eta.pkl"))
dRtruthmatch_p_eta    = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_eta.pkl"))
dRtruthmatch_p_scr    = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_scr.pkl"))
dRtruthunmatch_tru_eta  = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthunmatched_eta.pkl"))
dRtruthunmatch_p_eta  = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthunmatched_eta.pkl"))
dRtruthunmatch_p_scr  = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthunmatched_scr.pkl"))

#dR truth-tar matched
dRtruthtarmatch_tru_eta    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthtarmatched_eta.pkl"))
dRtruthtarmatch_tar_eta    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRtruthtarmatched_eta.pkl"))


print("Implementing a temporary, post-hoc confidence threshold:")
print(f"Number        total targets: {len(total_tar_eta)}\nNumber         total truths: {len(total_tru_eta)}\nNumber        total predictions: {len(total_p_eta)}")
print(f"Number      matched targets: {len(match_tar_eta)}\nNumber      matched predictions: {len(match_p_eta)}")
print(f"Number    unmatched targets: {len(unmatch_tar_eta)}\nNumber    unmatched predictions: {len(unmatch_p_eta)}")
print(f"Number dR   matched targets: {len(dRmatch_tar_eta)}\nNumber dR   matched predictions: {len(dRmatch_p_eta)}")
print(f"Number dR unmatched targets: {len(dRunmatch_tar_eta)}\nNumber dR unmatched predictions: {len(dRunmatch_p_eta)}")
print()
print(f"Number dR (truth) matched truth: {len(dRtruthmatch_tru_eta)}\nNumber dR (truth) matched preds: {len(dRtruthmatch_p_eta)}")
print(f"Number dR (truth) unmatched truth: {len(dRtruthunmatch_tru_eta)}\nNumber dR (truth) unmatched preds: {len(dRtruthunmatch_p_eta)}")

scr_threshold = 0.5
total_scr_mask = total_p_scr > scr_threshold
total_p_eta = total_p_eta[total_scr_mask]

match_scr_mask = match_p_scr > scr_threshold
match_tar_eta = match_tar_eta[match_scr_mask]
match_p_eta = match_p_eta[match_scr_mask]

dRmatch_scr_mask = dRmatch_p_scr > scr_threshold
dRmatch_tar_eta = dRmatch_tar_eta[dRmatch_scr_mask]
dRmatch_p_eta = dRmatch_p_eta[dRmatch_scr_mask]

unmatch_scr_mask = unmatch_p_scr > scr_threshold
unmatch_p_eta = unmatch_p_eta[unmatch_scr_mask]

dRunmatch_scr_mask = dRunmatch_p_scr > scr_threshold
dRunmatch_p_eta = dRunmatch_p_eta[dRunmatch_scr_mask]

# truth matching
dRtruthmatch_scr_mask = dRtruthmatch_p_scr > scr_threshold
dRtruthmatch_tru_eta = dRtruthmatch_tru_eta[dRtruthmatch_scr_mask]
dRtruthmatch_p_eta = dRtruthmatch_p_eta[dRtruthmatch_scr_mask]

# dRtruthunmatch_scr_mask = dRtruthunmatch_p_scr > scr_threshold
# dRtruthunmatch_p_eta = dRtruthunmatch_p_eta[dRtruthunmatch_scr_mask]

print("\n\nNew score threshold applied")
print(f"Number        total targets: {len(total_tar_eta)}\nNumber        total predictions: {len(total_p_eta)}")
print(f"Number      matched targets: {len(match_tar_eta)}\nNumber      matched predictions: {len(match_p_eta)}")
print(f"Number    unmatched targets: {len(unmatch_tar_eta)}\nNumber    unmatched predictions: {len(unmatch_p_eta)}")
print(f"Number dR   matched targets: {len(dRmatch_tar_eta)}\nNumber dR   matched predictions: {len(dRmatch_p_eta)}")
print(f"Number dR unmatched targets: {len(dRunmatch_tar_eta)}\nNumber dR unmatched predictions: {len(dRunmatch_p_eta)}")
print()
print(f"Number dR (truth) matched truth: {len(dRtruthmatch_tru_eta)}\nNumber dR (truth) matched preds: {len(dRtruthmatch_p_eta)}")
print(f"Number dR (truth) unmatched truth: {len(dRtruthunmatch_tru_eta)}\nNumber dR (truth) unmatched preds: {len(dRtruthunmatch_p_eta)}")


nominal = False
total_unc = False
signed_diff = False
signed_diff_var = False
ptplot2d_tru = True
ptplot2d_tar = True

################################################################
print("=======================================================================================================")
print(f"Plotting jet eta, saving to {save_folder}")
print("=======================================================================================================\n")


################################################################
if nominal:
    print(f"Plotting total jet eta: {len(total_p_eta)} predictions, {len(total_tar_eta)} targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(total_p_eta,bins=50,histtype='step',color='red',lw=1.5,label='Predicted Jets')
    freq_tar, bins, _    = ax0.hist(total_tar_eta,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
    freq_tru, bins, _    = ax0.hist(total_tru_eta,bins=bins,histtype='step',color='gold',lw=1.5,label='Truth Jets')
    ax0.set_title('Pseudorapidity Total', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $\eta$')
    f.savefig(save_folder + f'/jet_eta_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    print(f"Plotting matched jet eta: {len(match_p_eta)} predictions, {len(match_tar_eta)} targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(match_p_eta,bins=50,histtype='step',color='red',lw=1.5,label='Predicted Jets')
    freq_tar, bins, _    = ax0.hist(match_tar_eta,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
    ax0.set_title('Pseudorapidity IoU Matched', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $\eta$')
    f.savefig(save_folder + f'/jet_eta_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    print(f"Plotting dR matched jet eta: {len(dRmatch_p_eta)} predictions, {len(dRmatch_tar_eta)} targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(dRmatch_p_eta,bins=50,histtype='step',color='red',lw=1.5,label='Predicted Jets')
    freq_tar, bins, _    = ax0.hist(dRmatch_tar_eta,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
    freq_pred, bins, _   = ax0.hist(dRtruthmatch_p_eta,bins=bins,histtype='step',color='firebrick',lw=1.5,label='Predicted Jets (Truth match)')
    freq_tru, bins, _    = ax0.hist(dRtruthmatch_tru_eta,bins=bins,histtype='step',color='gold',lw=1.5,label='Truth Jets')
    ax0.set_title('Pseudorapidity deltaR Matched', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $\eta$')
    f.savefig(save_folder + f'/jet_eta_dRmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    print(f"Plotting unmatched jet eta: {len(unmatch_p_eta)} predictions, {len(unmatch_tar_eta)} targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(unmatch_p_eta,bins=50,histtype='step',color='red',lw=1.5,label='Predicted Jets')
    freq_tar, bins, _    = ax0.hist(unmatch_tar_eta,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
    ax0.set_title('Pseudorapidity IoU Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $\eta$')
    f.savefig(save_folder + f'/jet_eta_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    print(f"Plotting dR unmatched jet eta: {len(dRunmatch_p_eta)} predictions, {len(dRunmatch_tar_eta)} targets")
    f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    freq_pred, bins, _   = ax0.hist(dRunmatch_p_eta,bins=50,histtype='step',color='red',lw=1.5,label='Predicted Jets')
    freq_tar, bins, _    = ax0.hist(dRunmatch_tar_eta,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
    freq_tru, bins, _    = ax0.hist(dRtruthunmatch_tru_eta,bins=bins,histtype='step',color='gold',lw=1.5,label='Truth Jets')
    ax0.set_title('Pseudorapidity deltaR Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
    ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
    hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    ax0.set(yscale='log',xlabel='Jet $\eta$')
    f.savefig(save_folder + f'/jet_eta_dRunmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


if total_unc:
    # eta w/ errors
    print("Plotting eta with stat. unc and ratio ")
    unc_save_folder = save_folder + "/unc/"
    if not os.path.exists(unc_save_folder): os.makedirs(unc_save_folder)

    bin_start = min(min(total_p_eta),min(total_tar_eta),min(total_tru_eta))
    bin_stop = max(max(total_p_eta),max(total_tar_eta),max(total_tru_eta))
    print(bin_start, bin_stop)
    n_bins = 20
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # plot 1
    f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    freq_tru, bins, _   = ax[0].hist(total_tru_eta,bins=bins,histtype='step',color='gold',lw=2,label='Truth Jets')
    freq_tar, bins, _   = ax[0].hist(total_tar_eta,bins=bins,histtype='step',color='green',lw=2,label='AKT jets')
    freq_pre, bins, _   = ax[0].hist(total_p_eta,bins=bins,histtype='step',color='red',lw=2,label='CNN Jets')
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
    ax[0].legend([tru_line,tar_line,pre_line], ['Truth jets','Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.76, 0.8), fontsize=12) 
    ax[0].set(yscale='log',ylabel='Number of jets',ylim=(1,5e5))
    y_ticks = ax[0].yaxis.get_major_ticks()
    y_ticks[0].label1.set_visible(False)
    ax[0].text(-2.7,2e5, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    ax[0].text(-2.0,2e5, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    ax[0].text(-2.7,1.2e5, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    ax[0].text(-2.7,8e4, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    ax[1].set(xlabel='Jet $\eta$ [GeV]')
    # ax[1].set(ylim=(0.0,2.5))
    ax[1].set_yticks(np.arange(0.0, 6.0, 1.0))
    f.savefig(unc_save_folder + f'/jet_eta_total_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()



if signed_diff:

    print(f"Plotting signed delta eta.")
    # calculate signed difference, between pred-truth, target-truth
    pre_sgn_eta = np.sign(dRtruthmatch_p_eta) * (dRtruthmatch_p_eta - dRtruthmatch_tru_eta)
    tar_sgn_eta = np.sign(dRtruthtarmatch_tar_eta) * (dRtruthtarmatch_tar_eta - dRtruthtarmatch_tru_eta)

    # now bin and take an average
    bin_start = min(min(dRtruthmatch_p_eta),min(dRtruthmatch_tru_eta),min(dRtruthtarmatch_tar_eta),min(dRtruthtarmatch_tru_eta))
    bin_stop = max(max(dRtruthmatch_p_eta),max(dRtruthmatch_tru_eta),max(dRtruthtarmatch_tar_eta),max(dRtruthtarmatch_tru_eta))
    print(bin_start, bin_stop)
    n_bins = 50
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    total_pre_sgn_eta, total_tar_sgn_eta = [],[]
    for bin_idx in range(len(bin_centers)):
        bin_mask = (bins[bin_idx]<dRtruthmatch_tru_eta) & (dRtruthmatch_tru_eta<bins[bin_idx+1])
        pre_sgn_eta_bin_i = pre_sgn_eta[bin_mask]
        total_pre_sgn_eta.append(np.mean(pre_sgn_eta_bin_i))
        bin_mask = (bins[bin_idx]<dRtruthtarmatch_tru_eta) & (dRtruthtarmatch_tru_eta<bins[bin_idx+1])
        tar_sgn_eta_bin_i = tar_sgn_eta[bin_mask]
        total_tar_sgn_eta.append(np.mean(tar_sgn_eta_bin_i))

    # plot 1
    f, ax = plt.subplots(1, 1, figsize=(9, 8)) 
    ax.scatter(bin_centers,total_pre_sgn_eta,marker="D",s=120,color="red",label="CNN Jets")
    ax.scatter(bin_centers,total_tar_sgn_eta,marker="x",s=120,color="green",label="AKT Jets")
    ax.set(xlabel=r'Jet $\eta$',ylabel=r'Mean $\text{sgn}(\eta^{\text{reco}}) \times (\eta^{\text{reco}} - \eta^{\text{true}})$')
    ax.ticklabel_format(style='plain')
    ax.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
    f.savefig(save_folder + f'/jet_sgn_eta_diff.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    # now bin and take an average
    bin_start = 0
    bin_stop = max(max(dRtruthmatch_p_eta),max(dRtruthmatch_tru_eta),max(dRtruthtarmatch_tar_eta),max(dRtruthtarmatch_tru_eta))
    print(bin_start, bin_stop)
    n_bins = 50
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    total_pre_sgn_abs_eta, total_tar_sgn_abs_eta = [],[]
    for bin_idx in range(len(bin_centers)):
        bin_mask = (bins[bin_idx]<np.abs(dRtruthmatch_tru_eta)) & (np.abs(dRtruthmatch_tru_eta)<bins[bin_idx+1])
        pre_sgn_eta_bin_i = pre_sgn_eta[bin_mask]
        total_pre_sgn_abs_eta.append(np.mean(pre_sgn_eta_bin_i))
        bin_mask = (bins[bin_idx]<np.abs(dRtruthtarmatch_tru_eta)) & (np.abs(dRtruthtarmatch_tru_eta)<bins[bin_idx+1])
        tar_sgn_eta_bin_i = tar_sgn_eta[bin_mask]
        total_tar_sgn_abs_eta.append(np.mean(tar_sgn_eta_bin_i))

    # plot 2
    f, ax = plt.subplots(1, 1, figsize=(9, 8)) 
    ax.scatter(bin_centers,total_pre_sgn_abs_eta,marker="D",s=80,color="red",label="CNN Jets")
    ax.scatter(bin_centers,total_tar_sgn_abs_eta,marker="x",s=80,color="green",label="AKT Jets")
    ax.set(xlabel=r'Jet $|\eta|$',ylabel=r'Mean $\text{sgn}(\eta^{\text{reco}}) \times (\eta^{\text{reco}} - \eta^{\text{true}})$')
    ax.set(xlim=(0.0, 2.75))
    ax.ticklabel_format(style='plain')
    ax.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
    ax.text(0.13,0.025, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    ax.text(0.55,0.025, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    ax.text(0.13,0.0225, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    ax.text(0.13,0.0205, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    f.savefig(save_folder + f'/jet_sgn_abs_eta_diff.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()




if signed_diff_var:
    print(f"Plotting signed delta eta split in pT")
    dRtruthmatch_tru_pt    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthmatched_pt.pkl"))
    dRtruthmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_pt.pkl"))
    dRtruthmatch_tru_eta    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthmatched_eta.pkl"))
    dRtruthmatch_p_eta    = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_eta.pkl"))

    low_pt_mask = (dRtruthmatch_p_pt > 20) & (dRtruthmatch_p_pt < 60)
    mid_pt_mask = (dRtruthmatch_p_pt > 60) & (dRtruthmatch_p_pt < 100)
    hi_pt_mask  = (dRtruthmatch_p_pt > 100) & (dRtruthmatch_p_pt < 150)
    top_pt_mask = (dRtruthmatch_p_pt > 150)
    print(dRtruthmatch_tru_eta.shape,dRtruthmatch_p_eta.shape,dRtruthmatch_p_pt.shape,dRtruthmatch_tru_pt.shape)

    low_pt_dRtruthmatch_tru_eta = dRtruthmatch_tru_eta[low_pt_mask]
    low_pt_dRtruthmatch_pre_eta = dRtruthmatch_p_eta[low_pt_mask]

    mid_pt_dRtruthmatch_tru_eta = dRtruthmatch_tru_eta[mid_pt_mask]
    mid_pt_dRtruthmatch_pre_eta = dRtruthmatch_p_eta[mid_pt_mask]
    
    hi_pt_dRtruthmatch_tru_eta = dRtruthmatch_tru_eta[hi_pt_mask]
    hi_pt_dRtruthmatch_pre_eta = dRtruthmatch_p_eta[hi_pt_mask]

    top_pt_dRtruthmatch_tru_eta = dRtruthmatch_tru_eta[top_pt_mask]
    top_pt_dRtruthmatch_pre_eta = dRtruthmatch_p_eta[top_pt_mask]

    # calculate signed difference, between pred-truth, target-truth
    low_pre_sgn_eta = np.sign(low_pt_dRtruthmatch_pre_eta) * (low_pt_dRtruthmatch_pre_eta - low_pt_dRtruthmatch_tru_eta)
    mid_pre_sgn_eta = np.sign(mid_pt_dRtruthmatch_pre_eta) * (mid_pt_dRtruthmatch_pre_eta - mid_pt_dRtruthmatch_tru_eta)
    hi_pre_sgn_eta = np.sign(hi_pt_dRtruthmatch_pre_eta) * (hi_pt_dRtruthmatch_pre_eta - hi_pt_dRtruthmatch_tru_eta)
    top_pre_sgn_eta = np.sign(top_pt_dRtruthmatch_pre_eta) * (top_pt_dRtruthmatch_pre_eta - top_pt_dRtruthmatch_tru_eta)
    print('here:',low_pt_dRtruthmatch_tru_eta[:4],low_pt_dRtruthmatch_pre_eta[:4])
    print('here:',low_pre_sgn_eta[:4],top_pre_sgn_eta[:4])

    # now bin and take an average
    bin_start = min(min(dRtruthmatch_p_eta),min(dRtruthmatch_tru_eta),min(dRtruthtarmatch_tar_eta),min(dRtruthtarmatch_tru_eta))
    bin_stop = max(max(dRtruthmatch_p_eta),max(dRtruthmatch_tru_eta),max(dRtruthtarmatch_tar_eta),max(dRtruthtarmatch_tru_eta))
    print(bin_start, bin_stop)
    n_bins = 50
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    total_low_pre_sgn_eta = []
    total_mid_pre_sgn_eta = []
    total_hi_pre_sgn_eta = []
    total_top_pre_sgn_eta = []
    for bin_idx in range(len(bin_centers)):
        low_bin_mask = (bins[bin_idx]<low_pt_dRtruthmatch_tru_eta) & (low_pt_dRtruthmatch_tru_eta<bins[bin_idx+1])
        low_pre_sgn_eta_bin_i = low_pre_sgn_eta[low_bin_mask]
        total_low_pre_sgn_eta.append(np.mean(low_pre_sgn_eta_bin_i))
        mid_bin_mask = (bins[bin_idx]<mid_pt_dRtruthmatch_tru_eta) & (mid_pt_dRtruthmatch_tru_eta<bins[bin_idx+1])
        mid_pre_sgn_eta_bin_i = mid_pre_sgn_eta[mid_bin_mask]
        total_mid_pre_sgn_eta.append(np.mean(mid_pre_sgn_eta_bin_i))
        hi_bin_mask = (bins[bin_idx]<hi_pt_dRtruthmatch_tru_eta) & (hi_pt_dRtruthmatch_tru_eta<bins[bin_idx+1])
        hi_pre_sgn_eta_bin_i = hi_pre_sgn_eta[hi_bin_mask]
        total_hi_pre_sgn_eta.append(np.mean(hi_pre_sgn_eta_bin_i))
        top_bin_mask = (bins[bin_idx]<top_pt_dRtruthmatch_tru_eta) & (top_pt_dRtruthmatch_tru_eta<bins[bin_idx+1])
        top_pre_sgn_eta_bin_i = top_pre_sgn_eta[top_bin_mask]
        total_top_pre_sgn_eta.append(np.mean(top_pre_sgn_eta_bin_i))

    # plot 1
    f, ax = plt.subplots(1, 1, figsize=(9, 8)) 
    ax.scatter(bin_centers,total_low_pre_sgn_eta,marker="D",s=85,color="red",label="20 GeV < CNN Jets < 60 GeV")
    ax.scatter(bin_centers,total_mid_pre_sgn_eta,marker="D",s=85,color="yellow",label="60 GeV < CNN Jets < 100 GeV")
    ax.scatter(bin_centers,total_hi_pre_sgn_eta,marker="D",s=85,color="orange",label="100 GeV < CNN Jets < 150 GeV")
    ax.scatter(bin_centers,total_top_pre_sgn_eta,marker="D",s=85,color="blue",label="150 GeV < CNN Jets")
    ax.set(xlabel=r'Jet $\eta$',ylabel=r'Mean $\text{sgn}(\eta^{\text{reco}}) \times (\eta^{\text{reco}} - \eta^{\text{true}})$')
    ax.ticklabel_format(style='plain')
    ax.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="x-small")
    f.savefig(save_folder + f'/jet_var_sgn_eta_diff.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    # now bin in abs eta and take an average
    bin_start = 0
    bin_stop = max(max(dRtruthmatch_p_eta),max(dRtruthmatch_tru_eta),max(dRtruthtarmatch_tar_eta),max(dRtruthtarmatch_tru_eta))
    print(bin_start, bin_stop)
    n_bins = 50
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    total_low_pre_sgn_eta = []
    total_mid_pre_sgn_eta = []
    total_hi_pre_sgn_eta = []
    total_top_pre_sgn_eta = []
    for bin_idx in range(len(bin_centers)):
        low_bin_mask = (bins[bin_idx]<np.abs(low_pt_dRtruthmatch_tru_eta)) & (np.abs(low_pt_dRtruthmatch_tru_eta)<bins[bin_idx+1])
        low_pre_sgn_eta_bin_i = low_pre_sgn_eta[low_bin_mask]
        total_low_pre_sgn_eta.append(np.mean(low_pre_sgn_eta_bin_i))
        mid_bin_mask = (bins[bin_idx]<np.abs(mid_pt_dRtruthmatch_tru_eta)) & (np.abs(mid_pt_dRtruthmatch_tru_eta)<bins[bin_idx+1])
        mid_pre_sgn_eta_bin_i = mid_pre_sgn_eta[mid_bin_mask]
        total_mid_pre_sgn_eta.append(np.mean(mid_pre_sgn_eta_bin_i))
        hi_bin_mask = (bins[bin_idx]<np.abs(hi_pt_dRtruthmatch_tru_eta)) & (np.abs(hi_pt_dRtruthmatch_tru_eta)<bins[bin_idx+1])
        hi_pre_sgn_eta_bin_i = hi_pre_sgn_eta[hi_bin_mask]
        total_hi_pre_sgn_eta.append(np.mean(hi_pre_sgn_eta_bin_i))
        top_bin_mask = (bins[bin_idx]<np.abs(top_pt_dRtruthmatch_tru_eta)) & (np.abs(top_pt_dRtruthmatch_tru_eta)<bins[bin_idx+1])
        top_pre_sgn_eta_bin_i = top_pre_sgn_eta[top_bin_mask]
        total_top_pre_sgn_eta.append(np.mean(top_pre_sgn_eta_bin_i))


    # plot 2
    f, ax = plt.subplots(1, 1, figsize=(9, 8)) 
    ax.scatter(bin_centers,total_low_pre_sgn_eta,marker="D",s=40,color="red",label="20 GeV < CNN Jets < 60 GeV")
    ax.scatter(bin_centers,total_mid_pre_sgn_eta,marker="D",s=40,color="yellow",label="60 GeV < CNN Jets < 100 GeV")
    ax.scatter(bin_centers,total_hi_pre_sgn_eta,marker="D",s=40,color="orange",label="100 GeV < CNN Jets < 150 GeV")
    ax.scatter(bin_centers,total_top_pre_sgn_eta,marker="D",s=40,color="blue",label="150 GeV < CNN Jets")
    ax.set(xlabel=r'Jet $|\eta|$',ylabel=r'Mean $\text{sgn}(\eta^{\text{reco}}) \cdot (\eta^{\text{reco}} - \eta^{\text{true}})$')
    ax.set(xlim=(-0.1, 2.75))
    ax.ticklabel_format(style='plain')
    ax.legend(loc='lower left',bbox_to_anchor=(0.52, 0.78),fontsize="small")
    ax.text(0.15,0.044, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    ax.text(0.56,0.044, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    ax.text(0.15,0.0415, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    ax.text(0.15,0.0395, r"MC21 Dijet JZ1-4",fontfamily='sans-serif',fontsize=12)
    f.savefig(save_folder + f'/jet_var_sgn_abs_eta_diff.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


if ptplot2d_tar:
    print("Plotting 2d histograms of pT against eta in pred, target (dRmatched for now)")
    dRmatch_tar_pt    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRmatched_pt.pkl"))
    dRmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_pt.pkl"))
    dRmatch_tar_eta    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRmatched_eta.pkl"))
    dRmatch_p_eta    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_eta.pkl"))
    dRmatch_p_scr    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_scr.pkl"))


    # print(f"Plotting dR matched jet eta: {len(dRmatch_p_eta)} predictions, {len(dRmatch_tar_eta)} targets")
    # f,ax0 = plt.subplots(1,1,figsize=(9, 6))
    # freq_pred, bins, _   = ax0.hist(dRmatch_p_eta,bins=50,histtype='step',color='red',lw=1.5,label='Predicted Jets')
    # freq_tar, bins, _    = ax0.hist(dRmatch_tar_eta,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
    # freq_pred, bins, _   = ax0.hist(dRtruthmatch_p_eta,bins=bins,histtype='step',color='firebrick',lw=1.5,label='Predicted Jets (Truth match)')
    # freq_tru, bins, _    = ax0.hist(dRtruthmatch_tru_eta,bins=bins,histtype='step',color='gold',lw=1.5,label='Truth Jets')
    # ax0.set_title('Pseudorapidity deltaR Matched', fontsize=16, fontfamily="TeX Gyre Heros")
    # ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
    # hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
    # ax0.set(yscale='log',xlabel='Jet $\eta$')
    # f.savefig(save_folder + f'/jet_eta_dRmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    # plt.close()

    eta_bins = np.arange(-2.1,2.2,step=0.1)
    pt_bins = np.arange(0,200,step=5)
    
    f,ax = plt.subplots()
    cmap_p = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","blue","violet","red"])
    h,_,_,img = ax.hist2d(dRmatch_p_pt,dRmatch_p_eta,bins=(pt_bins,eta_bins),cmap=cmap_p)
    cbar = f.colorbar(img,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('Number of predicted jets', rotation=90)
    ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
    ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
    ax.set(xlabel=f"pT",ylabel=f"eta")
    f.savefig(save_folder + f'/2d_pretar_pt_eta.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()

    f,ax = plt.subplots()
    cmap_t = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","green","lime"])
    h,_,_,img = ax.hist2d(dRmatch_tar_pt,dRmatch_tar_eta,bins=(pt_bins,eta_bins),cmap=cmap_t)
    cbar = f.colorbar(img,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('Number of target jets', rotation=90)
    ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
    ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
    ax.set(xlabel=f"pT",ylabel=f"eta")
    f.savefig(save_folder + f'/2d_tarpre_pt_eta.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()





if ptplot2d_tru:
    print("Plotting 2d histograms of pT against eta in pred, target, truth (dRtruthmatched for now)")
    dRtruthmatch_tru_pt    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthmatched_pt.pkl"))
    dRtruthmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_pt.pkl"))
    dRtruthmatch_tru_eta    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthmatched_eta.pkl"))
    dRtruthmatch_p_eta    = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_eta.pkl"))

    dRtruthtarmatch_tru_pt    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthtarmatched_pt.pkl"))
    dRtruthtarmatch_tar_pt    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRtruthtarmatched_pt.pkl"))
    dRtruthtarmatch_tru_eta    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthtarmatched_eta.pkl"))
    dRtruthtarmatch_tar_eta    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRtruthtarmatched_eta.pkl"))

    eta_bins = np.arange(-2.1,2.2,step=0.1)
    pt_bins = np.arange(0,250,step=5)
    # H_tot_tru,_,_ = np.histogram2d(dRtruthmatch_tru_pt,dRtruthmatch_tru_eta,bins=(pt_bins,eta_bins))
    # H_tot_tru = H_tot_tru.T

    f,ax = plt.subplots()
    cmap_t = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","yellow","gold"])
    # ii = ax.imshow(H_tot_tru,cmap=cmap_t)
    h,_,_,img = ax.hist2d(dRtruthmatch_tru_pt,dRtruthmatch_tru_eta,bins=(pt_bins,eta_bins),cmap=cmap_t)
    cbar = f.colorbar(img,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('Number of truth jets', rotation=90)
    # cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
    ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
    ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
    ax.set(xlabel=f"pT",ylabel=f"eta")
    f.savefig(save_folder + f'/2d_tru_pt_eta.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    H_tot_pre,_,_ = np.histogram2d(dRtruthmatch_p_pt,dRtruthmatch_p_eta,bins=(pt_bins,eta_bins))
    H_tot_pre = H_tot_pre.T

    f,ax = plt.subplots()
    cmap_p = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","blue","violet","red"])
    # ii = ax.imshow(H_tot_pre,cmap=cmap_p)
    h,_,_,img = ax.hist2d(dRtruthmatch_p_pt,dRtruthmatch_p_eta,bins=(pt_bins,eta_bins),cmap=cmap_p)
    cbar = f.colorbar(img,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('Number of predicted jets', rotation=90)
    # cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
    ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
    ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
    ax.set(xlabel=f"pT",ylabel=f"eta")
    f.savefig(save_folder + f'/2d_pre_pt_eta.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    f,ax = plt.subplots()
    cmap_t = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","yellow","gold"])
    h,_,_,img = ax.hist2d(dRtruthtarmatch_tru_pt,dRtruthtarmatch_tru_eta,bins=(pt_bins,eta_bins),cmap=cmap_t)
    # ii = ax.imshow(H_tot_tar,cmap=cmap_t)
    cbar = f.colorbar(img,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('Number of truth jets', rotation=90)
    # cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
    ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
    ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
    ax.set(xlabel=f"pT",ylabel=f"eta")
    f.savefig(save_folder + f'/2d_trutar_pt_eta.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    f,ax = plt.subplots()
    cmap_t = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","green","lime"])
    h,_,_,img = ax.hist2d(dRtruthtarmatch_tar_pt,dRtruthtarmatch_tar_eta,bins=(pt_bins,eta_bins),cmap=cmap_t)
    # ii = ax.imshow(H_tot_tru,cmap=cmap_t)
    cbar = f.colorbar(img,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('Number of target jets', rotation=90)
    # cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
    ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
    ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
    ax.set(xlabel=f"pT",ylabel=f"eta")
    f.savefig(save_folder + f'/2d_tartru_pt_eta.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()



