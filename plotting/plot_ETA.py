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
signed_diff = True


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
    n_bins = 20
    bins = np.linspace(bin_start, bin_stop, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    total_pre_sgn_eta, total_tar_sgn_eta = [],[]
    for bin_idx in range(len(bin_centers)):
        bin_mask = (bins[bin_idx]<dRtruthmatch_p_eta) & (dRtruthmatch_p_eta<bins[bin_idx+1])
        pre_sgn_eta_bin_i = pre_sgn_eta[bin_mask]
        total_pre_sgn_eta.append(np.mean(pre_sgn_eta_bin_i))
        bin_mask = (bins[bin_idx]<dRtruthtarmatch_tar_eta) & (dRtruthtarmatch_tar_eta<bins[bin_idx+1])
        tar_sgn_eta_bin_i = tar_sgn_eta[bin_mask]
        total_tar_sgn_eta.append(np.mean(tar_sgn_eta_bin_i))


    # plot 1
    f, ax = plt.subplots(1, 1, figsize=(9, 8)) 
    ax.plot(bin_centers,total_pre_sgn_eta,marker="D",size=10,color="red",label="CNN Jets")
    ax.plot(bin_centers,total_pre_sgn_eta,marker="x",size=10,color="green",label="AKT Jets")
    ax.set(xlabel=r'Jet $\eta$',ylabel=r'Mean $\text{sgn}(\eta^{\text{reco}}) \times (\eta^{\text{reco}} - \eta^{\text{true}})$')
    ax.set(yscale='log')
    ax.legend(loc='lower left',bbox_to_anchor=(0.65, 0.82),fontsize="medium")
    f.savefig(save_folder + f'/jet_sgn_eta_diff.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()




    # # plot 1
    # f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
    # freq_tru, bins, _   = ax[0].hist(total_tru_eta,bins=bins,histtype='step',color='gold',lw=2,label='Truth Jets')
    # freq_tar, bins, _   = ax[0].hist(total_tar_eta,bins=bins,histtype='step',color='green',lw=2,label='AKT jets')
    # freq_pre, bins, _   = ax[0].hist(total_p_eta,bins=bins,histtype='step',color='red',lw=2,label='CNN Jets')
    # # stat. unc sqrt(counts)
    # tru_errors = np.sqrt(freq_tru)
    # tar_errors = np.sqrt(freq_tar)
    # pre_errors = np.sqrt(freq_pre)
    # ax[0].errorbar(bin_centers, freq_tru,  yerr=tru_errors, color='gold', ls='none')
    # ax[0].errorbar(bin_centers, freq_tar, yerr=tar_errors, color='green', ls='none')
    # ax[0].errorbar(bin_centers, freq_pre, yerr=pre_errors, color='red',  ls='none')

    # ratio_pred_tru = np.divide(freq_pre, freq_tru, out=np.zeros_like(freq_pre), where=freq_tru != 0)
    # ratio_tar_tru  = np.divide(freq_tar, freq_tru, out=np.zeros_like(freq_tar), where=freq_tru != 0)
    # ax[1].hlines(y=1, xmin=bin_start, xmax=bin_stop, color='gold', linestyle='-', lw=1.6)

    # # uncertainty on the ratio
    # ratio_pred_erro = np.sqrt((pre_errors / freq_tru) ** 2 + (freq_pre * tru_errors / freq_tru**2)**2)
    # ratio_tar_erro = np.sqrt((tar_errors / freq_tru) ** 2 + (freq_tar * tru_errors / freq_tru**2)**2)
    # ax[1].fill_between(bin_centers[1:-1], ratio_pred_tru[1:-1] - ratio_pred_erro[1:-1], ratio_pred_tru[1:-1] + ratio_pred_erro[1:-1], alpha=0.5, edgecolor='crimson', facecolor='red')
    # ax[1].fill_between(bin_centers[1:-1], ratio_tar_tru[1:-1] - ratio_tar_erro[1:-1], ratio_tar_tru[1:-1] + ratio_tar_erro[1:-1], alpha=0.5, edgecolor='forestgreen', facecolor='green')

    # tar_line =  matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
    # pre_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
    # tru_line =  matplotlib.lines.Line2D([0], [0], color='gold', lw=3)  
    # ax[0].legend([tru_line,tar_line,pre_line], ['Truth jets','Target jets', 'CNN Jets'], loc='lower left', bbox_to_anchor=(0.76, 0.8), fontsize=12) 
    # ax[0].set(yscale='log',ylabel='Number of jets',ylim=(1,5e5))
    # y_ticks = ax[0].yaxis.get_major_ticks()
    # y_ticks[0].label1.set_visible(False)
    # ax[0].text(-2.7,2e5, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    # ax[0].text(-2.0,2e5, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    # ax[0].text(-2.7,1.2e5, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    # ax[0].text(-2.7,8e4, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    # ax[1].set(xlabel='Jet $\eta$ [GeV]')
    # # ax[1].set(ylim=(0.0,2.5))
    # ax[1].set_yticks(np.arange(0.0, 6.0, 1.0))
    # f.savefig(unc_save_folder + f'/jet_eta_total_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    # plt.close()


