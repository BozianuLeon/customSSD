import numpy as np 
import scipy
import math
import os

import itertools
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.pyplot as plt
import matplotlib

import mplhep as hep
hep.style.use(hep.style.ATLAS)

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496


def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)


def leading_jet_pt(list_of_jet_pts_in_event):
    try:
        return max(list_of_jet_pts_in_event)
    except ValueError:
        #Doesn't have enough (or any) jets, automatically lost in cut
        return np.nan

model_name = "jetSSD_mu200_pt_uconvnext_15e"
metrics_folder =f"/home/users/b/bozianu/work/SSD/simul_pt/cached_metrics/{model_name}/20241007-15/"
save_folder = f"/home/users/b/bozianu/work/SSD/simul_pt/cached_plots/{model_name}/20241007-15/phys/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)



#by event
total_n_truth = load_object(metrics_folder+"/box_metrics/n_truth.pkl")
total_n_preds = load_object(metrics_folder+"/box_metrics/n_preds.pkl")
total_delta_n = load_object(metrics_folder+"/box_metrics/delta_n.pkl")

total_n_tru_match = load_object(metrics_folder+"/box_metrics/n_matched_truth.pkl")
total_n_tru_unmatch = load_object(metrics_folder+"/box_metrics/n_unmatched_truth.pkl")
total_n_pre_match = load_object(metrics_folder+"/box_metrics/n_matched_preds.pkl")
total_n_pre_unmatch = load_object(metrics_folder+"/box_metrics/n_unmatched_preds.pkl")

total_n_tbox = load_object(metrics_folder+"/phys_metrics/n_tboxes.pkl")
total_num_tbox = load_object(metrics_folder+"/phys_metrics/num_tboxes.pkl")
total_n_pbox = load_object(metrics_folder+"/phys_metrics/n_pboxes.pkl")
total_num_pbox = load_object(metrics_folder+"/phys_metrics/num_pboxes.pkl")



#by jet
total_tru_pt = np.concatenate(load_object(metrics_folder+"/box_metrics/tboxes_pt.pkl"))
total_tru_eta = np.concatenate(load_object(metrics_folder+"/box_metrics/tboxes_eta.pkl"))
total_tru_phi = np.concatenate(load_object(metrics_folder+"/box_metrics/tboxes_phi.pkl"))
total_tru_matched = np.concatenate(load_object(metrics_folder+"/box_metrics/tboxes_matched.pkl"))

total_tbox_matched = np.concatenate(load_object(metrics_folder+"/phys_metrics/tboxes_matched.pkl"))
total_tbox_pt2sig = np.concatenate(load_object(metrics_folder+"/phys_metrics/tboxes_pT_2sig.pkl")) * 1/1000
total_tbox_eta2sig = np.concatenate(load_object(metrics_folder+"/phys_metrics/tboxes_eta_2sig.pkl"))
total_tbox_phi2sig = np.concatenate(load_object(metrics_folder+"/phys_metrics/tboxes_phi_2sig.pkl"))
total_tbox_ceta = np.concatenate(load_object(metrics_folder+"/phys_metrics/tboxcentre_eta.pkl"))
total_tbox_cphi = np.concatenate(load_object(metrics_folder+"/phys_metrics/tboxcentre_phi.pkl"))


total_pred_matched = np.concatenate(load_object(metrics_folder+"/box_metrics/pboxes_matched.pkl"))
total_pred_scores = np.concatenate(load_object(metrics_folder+"/box_metrics/pboxes_scores.pkl"))
total_pred_pt = np.concatenate(load_object(metrics_folder+"/box_metrics/pboxes_pt.pkl"))
total_pred_eta = np.concatenate(load_object(metrics_folder+"/box_metrics/pboxes_eta.pkl"))
total_pred_phi = np.concatenate(load_object(metrics_folder+"/box_metrics/pboxes_phi.pkl"))

total_pbox_matched = np.concatenate(load_object(metrics_folder+"/phys_metrics/pboxes_matched.pkl"))
total_pbox_scores = np.concatenate(load_object(metrics_folder+"/phys_metrics/pboxes_scores.pkl"))
total_pbox_pt2sig = np.concatenate(load_object(metrics_folder+"/phys_metrics/pboxes_pT_2sig.pkl")) * 1/1000
total_pbox_ptadj2sig = np.concatenate(load_object(metrics_folder+"/phys_metrics/pboxes_pT_adj_2sig.pkl")) * 1/1000
total_pbox_eta2sig = np.concatenate(load_object(metrics_folder+"/phys_metrics/pboxes_eta_2sig.pkl"))
total_pbox_phi2sig = np.concatenate(load_object(metrics_folder+"/phys_metrics/pboxes_phi_2sig.pkl"))
total_pbox_ceta = np.concatenate(load_object(metrics_folder+"/phys_metrics/pboxcentre_eta.pkl"))
total_pbox_cphi = np.concatenate(load_object(metrics_folder+"/phys_metrics/pboxcentre_phi.pkl"))

print(f"In box metrics there are {len(total_tru_pt)} target jets and {len(total_pred_pt)} predicted jets")
print(f"Of which {sum(total_tru_matched)} targets are matched and {sum(total_pred_matched)} matched predicted jets.")
print()
print(f"In phys metrics there are {len(total_tbox_pt2sig)} target jets and {len(total_pbox_ptadj2sig)} predicted jets")
print(f"Of which {sum(total_tbox_matched)} targets are matched and {sum(total_pbox_matched)} matched predicted jets.")
print()


pt_thresholds = [0,10,15,20,25,30,35,40,60,100,200,250,450,625]
tru_matched_percentage = []
for threshold in pt_thresholds:
    tru_mask = total_tru_pt > threshold
    total_tru_matched_mask = total_tru_matched[tru_mask]
    pred_mask = total_pred_pt > threshold
    total_pred_matched_mask = total_pred_matched[pred_mask]
    
    print(f'For target jets above {threshold}: {sum(total_tru_matched_mask)/len(total_tru_matched_mask)} of truth jets are matched')
    print(f'For predicted jets above {threshold}: {sum(total_pred_matched_mask)/len(total_pred_matched_mask)} of truth jets are matched')
    
    tru_mask = total_tru_pt > threshold
    total_tru_matched_mask = total_tbox_matched[tru_mask]
    pred_mask = total_pbox_ptadj2sig > threshold
    total_pred_matched_mask = total_pbox_matched[pred_mask]
    print(f'For PHYS target jets above {threshold}: {sum(total_tru_matched_mask)/len(total_tru_matched_mask)} of truth jets are matched')
    print(f'For PHYS predicted jets above {threshold}: {sum(total_pred_matched_mask)/len(total_pred_matched_mask)} of truth jets are matched')
    print()




#################################################################################
# plotting
log = True
image_format = "png"


# 1.
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_tru, bins, _ = ax0.hist(total_n_truth,bins=max(total_n_truth)-1,histtype='step',color='green',lw=2,label='Target Jets {:.2f}$\pm${:.1f}'.format(np.mean(total_n_truth),np.std(total_n_truth)))
freq_pred, _, _   = ax0.hist(total_n_preds,bins=bins,histtype='step',color='red',lw=2,label='Pred Jets {:.2f}$\pm${:.1f}'.format(np.mean(total_n_preds),np.std(total_n_preds)))
freq_pred, _, _   = ax0.hist(total_n_pbox,bins=bins,histtype='step',color='indigo',ls='--',lw=2,label='Pred Jets {:.2f}$\pm${:.1f}'.format(np.mean(total_n_pbox),np.std(total_n_pbox)))
ax0.grid()
ax0.set_title('Number of jets per event', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.6, 0.75),fontsize="small")

hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Num. jets')
f.savefig(save_folder + f'/n_jets_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

plt.close()





print()
match_pred_pt = total_pred_pt[np.where(total_pred_matched==1)]
match_pred_eta = total_pred_eta[np.where(total_pred_matched==1)]
match_pred_phi = total_pred_phi[np.where(total_pred_matched==1)]
match_pred_scr = total_pred_scores[np.where(total_pred_matched==1)]

match_pbox_pt = total_pbox_ptadj2sig[np.where(total_pbox_matched==1)]
match_pbox_eta = total_pbox_eta2sig[np.where(total_pbox_matched==1)]
match_pbox_phi = total_pbox_phi2sig[np.where(total_pbox_matched==1)]
match_pbox_scr = total_pbox_scores[np.where(total_pbox_matched==1)]

match_tru_pt = total_tru_pt[np.where(total_tru_matched==1)]
match_tru_eta = total_tru_eta[np.where(total_tru_matched==1)]
match_tru_phi = total_tru_phi[np.where(total_tru_matched==1)]



unmatch_pred_pt = total_pred_pt[np.where(total_pred_matched==0)]
unmatch_pred_eta = total_pred_eta[np.where(total_pred_matched==0)]
unmatch_pred_phi = total_pred_phi[np.where(total_pred_matched==0)]
unmatch_pred_scr = total_pred_scores[np.where(total_pred_matched==0)]

unmatch_pbox_pt = total_pbox_ptadj2sig[np.where(total_pbox_matched==0)]
unmatch_pbox_eta = total_pbox_eta2sig[np.where(total_pbox_matched==0)]
unmatch_pbox_phi = total_pbox_phi2sig[np.where(total_pbox_matched==0)]
unmatch_pbox_scr = total_pbox_scores[np.where(total_pbox_matched==0)]

unmatch_tru_pt = total_tru_pt[np.where(total_tru_matched==0)]
unmatch_tru_eta = total_tru_eta[np.where(total_tru_matched==0)]
unmatch_tru_phi = total_tru_phi[np.where(total_tru_matched==0)]


################################################################
# pt total
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
bin_start = min(min(total_pbox_ptadj2sig),min(total_pred_pt))
bin_stop = max(max(total_pbox_ptadj2sig),max(total_pred_pt))
bins = np.linspace(bin_start, bin_stop, 50 + 1)
freq_pbox, bins, _   = ax0.hist(total_pbox_ptadj2sig,bins=bins,histtype='step',color='indigo',lw=2,label='Weighted circle')
freq_pred, bins, _   = ax0.hist(total_pred_pt,bins=bins,histtype='step',color='red',lw=2,label='Sumpool')
freq_tru, bins, _ = ax0.hist(total_tru_pt,bins=bins,histtype='step',color='green',lw=2,label='Target Jets')

# ax0.set_title('Transverse Momentum', fontsize=16, fontfamily="TeX Gyre Heros")
lines = [r"$p_T^{\text{uncalib}} > 20 $GeV", r"$|\eta| < 2.1$"]
multi_line_text = '\n'.join(lines)
ax0.text(0.7, 0.7, multi_line_text, transform=ax0.transAxes, fontsize="small", fontname='TeX Gyre Heros')
ax0.text(0.19, 0.87, r"Dijet MC JZ4, <$\mu$>=200", transform=ax0.transAxes, fontsize="small", fontname='TeX Gyre Heros')
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.74),fontsize="medium")

hep.atlas.label(ax=ax0,label='Internal',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet $p_T$ [GeV]',ylabel='Number of Jets')
f.savefig(save_folder + f'/jet_pt_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()



f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 

bin_start = min(min(total_pbox_ptadj2sig),min(total_pred_pt))
bin_stop = max(max(total_pbox_ptadj2sig),max(total_pred_pt))
bins = np.linspace(bin_start, bin_stop, 50 + 1)
print(bin_start,np.floor(bin_start / 10) * 10)
freq_tru, bins, _ = ax[0].hist(total_tru_pt,bins=bins,histtype='step',color='green',lw=2,label='Target Jets')
freq_pbox, bins, _   = ax[0].hist(total_pbox_ptadj2sig,bins=bins,histtype='step',color='indigo',lw=2,ls='dashed',label='Weighted circle')
freq_pred, bins, _   = ax[0].hist(total_pred_pt,bins=bins,histtype='step',color='red',lw=2,ls='dotted',label='Sumpool')

bin_centers = (bins[:-1] + bins[1:]) / 2
ratio_pred_tru = np.divide(freq_pred, freq_tru, out=np.zeros_like(freq_pred), where=freq_tru != 0)
ratio_pbox_tru = np.divide(freq_pbox, freq_tru, out=np.zeros_like(freq_pbox), where=freq_tru != 0)
ax[1].plot(bin_centers, ratio_pred_tru, color='red', marker='o',ms=2,ls='dotted', label='Sumpool / Target Jets')
ax[1].plot(bin_centers, ratio_pbox_tru, color='indigo', marker='o',ms=2,ls='dashed', label='Weighted circle / Target Jets')
ax[1].axhline(y=1, color='green', linestyle='--', lw=2)


ax[0].legend(loc='lower left',bbox_to_anchor=(0.65, 0.74),fontsize="medium")
ax[0].set(yscale='log',ylabel='Number of Jets')
y_ticks = ax[0].yaxis.get_major_ticks()
y_ticks[0].label1.set_visible(False)
ax[1].set(xlabel='Jet $p_T$ [GeV]',ylabel='Ratio')
ax[1].set(ylim=(0.0,2.5))
f.savefig(save_folder + f'/jet_pt_total_new.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()



# pt w/ errors
f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 

bin_start = min(min(total_pbox_ptadj2sig),min(total_pred_pt))
bin_stop = max(max(total_pbox_ptadj2sig),max(total_pred_pt))
print(bin_start, bin_stop)
bin_start = (np.floor(bin_start / 10) * 10) - 1
bin_start = -21
bin_stop = np.ceil(bin_stop / 10) * 10
bin_width = 20
n_bins = int(np.ceil((bin_stop - bin_start) / bin_width))

bins = np.linspace(bin_start, bin_stop, n_bins)
bin_centers = (bins[:-1] + bins[1:]) / 2

freq_tru, bins, _ = ax[0].hist(total_tru_pt,bins=bins,histtype='step',color='green',lw=2,label='Target Jets')
freq_pred, bins, _   = ax[0].hist(total_pred_pt,bins=bins,histtype='step',color='red',lw=2,label='Sumpool')#,ls='dotted')
freq_pbox, bins, _   = ax[0].hist(total_pbox_ptadj2sig,bins=bins,histtype='step',color='indigo',lw=2,label='Weighted circle')#,ls='dashed')

# stat. unc sqrt(counts)
tru_errors = np.sqrt(freq_tru)
pbox_errors = np.sqrt(freq_pbox)
pred_errors = np.sqrt(freq_pred)

ax[0].errorbar(bin_centers, freq_tru, yerr=tru_errors, color='green', ls='none')
ax[0].errorbar(bin_centers, freq_pbox, yerr=pbox_errors, color='indigo', ls='none')
ax[0].errorbar(bin_centers, freq_pred, yerr=pred_errors, color='red', ls='none')

ratio_pred_tru = np.divide(freq_pred, freq_tru, out=np.zeros_like(freq_pred), where=freq_tru != 0)
ratio_pbox_tru = np.divide(freq_pbox, freq_tru, out=np.zeros_like(freq_pbox), where=freq_tru != 0)
ax[1].hlines(y=1, xmin=20, xmax=bin_centers[-2], color='green', linestyle='-', lw=1.6)

# uncertainty on the ratio
# ratio_errors1 = ratio_pred_tru * np.sqrt((pred_errors / freq_pred) ** 2 + (tru_errors / freq_tru) ** 2)
ratio_pred_erro = np.sqrt((pred_errors / freq_tru) ** 2 + (freq_pred * tru_errors / freq_tru**2)**2)
ratio_pbox_erro = np.sqrt((pbox_errors / freq_tru) ** 2 + (freq_pbox * tru_errors / freq_tru**2)**2)
# ax[1].fill_between(bin_centers, ratio_pred_tru - ratio_pred_erro, ratio_pred_tru + ratio_pred_erro, alpha=0.5, edgecolor='crimson', facecolor='red')
ax[1].fill_between(bin_centers[1:-1], ratio_pred_tru[1:-1] - ratio_pred_erro[1:-1], ratio_pred_tru[1:-1] + ratio_pred_erro[1:-1], alpha=0.5, edgecolor='crimson', facecolor='red')
ax[1].fill_between(bin_centers[1:-1], ratio_pbox_tru[1:-1] - ratio_pbox_erro[1:-1], ratio_pbox_tru[1:-1] + ratio_pbox_erro[1:-1], alpha=0.5, edgecolor='purple', facecolor='indigo')

# new line stuff
target_line = matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
pred_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
pbox_line =  matplotlib.lines.Line2D([0], [0], color='indigo', lw=3)  
ax[0].legend([target_line,pred_line,pbox_line], ['Target jets', 'Sumpool', 'Weighted Circle'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 


# ax[0].legend(loc='lower left',bbox_to_anchor=(0.65, 0.74),fontsize="medium")
ax[0].set(yscale='log',ylabel='Number of jets',ylim=(1,5e5))
y_ticks = ax[0].yaxis.get_major_ticks()
y_ticks[0].label1.set_visible(False)
ax[1].set(xlabel='Jet $p_T$ [GeV]')
ax[1].set(ylim=(0.0,2.5))
ax[1].set_yticks(np.arange(0.0, 3.0, 1.0))
new_yticklabels = [f'{int(tick)}' for tick in np.arange(0.0, 3.0, 1.0)]  
ax[1].set_yticklabels(new_yticklabels)


tick_labels = ax[1].get_xticklabels()
tick_positions = ax[1].get_xticks()
for label in tick_labels:
    label.set_verticalalignment('bottom')  
    label.set_y(label.get_position()[1] - 0.22)


f.savefig(save_folder + f'/jet_pt_total_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()









# pt match
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
bin_start = min(min(match_pbox_pt),min(match_pred_pt))
bin_stop = max(max(match_pbox_pt),max(match_pred_pt))
bins = np.linspace(bin_start, bin_stop, 50 + 1)
freq_pbox, bins, _   = ax0.hist(match_pbox_pt,bins=bins,histtype='step',color='indigo',lw=2,label='Weighted circle')
freq_pred, bins, _   = ax0.hist(match_pred_pt,bins=bins,histtype='step',color='red',lw=2,label='Sumpool')
freq_tru, bins, _ = ax0.hist(match_tru_pt,bins=bins,histtype='step',color='green',lw=2,label='Target Jets')

ax0.set_title('Matched Jets', fontsize=16, fontfamily="TeX Gyre Heros")
lines = [r"$p_T^{\text{uncalib}} > 20 $GeV", r"$|\eta| < 2.1$"]
multi_line_text = '\n'.join(lines)
ax0.text(0.7, 0.7, multi_line_text, transform=ax0.transAxes, fontsize="small", fontname='TeX Gyre Heros')
ax0.text(0.05, 0.87, r"Dijet MC JZ4, <$\mu$>=200", transform=ax0.transAxes, fontsize="small", fontname='TeX Gyre Heros')
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.74),fontsize="medium")

hep.atlas.label(ax=ax0,label='Internal',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet pT (GeV)')
f.savefig(save_folder + f'/jet_pt_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

# pt unmatch
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
bin_start = min(min(unmatch_pred_pt),min(unmatch_pbox_pt))
bin_stop = max(max(unmatch_pred_pt),max(unmatch_pbox_pt))
bins = np.linspace(bin_start, bin_stop, 50 + 1)
freq_pred, bins, _   = ax0.hist(unmatch_pred_pt,bins=bins,histtype='step',color='red',lw=2,label='Sumpool')
freq_pbox, bins, _   = ax0.hist(unmatch_pbox_pt,bins=bins,histtype='step',color='indigo',lw=2,label='Weighted circle')
freq_tru, bins, _ = ax0.hist(unmatch_tru_pt,bins=bins,histtype='step',color='green',lw=2,label='Target Jets')

ax0.set_title('Unmatched Jets/Boxes', fontsize=16, fontfamily="TeX Gyre Heros")
lines = [r"$p_T^{\text{uncalib}} > 20 $GeV", r"$|\eta| < 2.1$"]
multi_line_text = '\n'.join(lines)
ax0.text(0.7, 0.7, multi_line_text, transform=ax0.transAxes, fontsize="small", fontname='TeX Gyre Heros')
ax0.text(0.05, 0.87, r"Dijet MC JZ4, <$\mu$>=200", transform=ax0.transAxes, fontsize="small", fontname='TeX Gyre Heros')
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.74),fontsize="medium")

hep.atlas.label(ax=ax0,label='Internal',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet pT (GeV)')
f.savefig(save_folder + f'/jet_pt_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()





event_tru_pt = load_object(metrics_folder+"/box_metrics/tboxes_pt.pkl")
event_pred_pt = load_object(metrics_folder+"/box_metrics/pboxes_pt.pkl")
event_pbox_ptadj2sig = load_object(metrics_folder+"/phys_metrics/pboxes_pT_adj_2sig.pkl")
tru_lead_pt = np.array([leading_jet_pt(x) for x in event_tru_pt])
pred_lead_pt = np.array([leading_jet_pt(x) for x in event_pred_pt])
pbox_lead_pt = np.array([leading_jet_pt(x)*1/1000 for x in event_pbox_ptadj2sig])

f,ax0 = plt.subplots(1,1,figsize=(9, 6))
bin_start = min(min(tru_lead_pt),min(pred_lead_pt),min(pbox_lead_pt))
bin_stop = max(max(tru_lead_pt),max(pred_lead_pt),max(pbox_lead_pt))
bins = np.linspace(bin_start, bin_stop, 50 + 1)
freq_pbox, bins, _   = ax0.hist(pbox_lead_pt,bins=bins,histtype='step',color='indigo',lw=2,label='Weighted circle')
freq_pred, bins, _   = ax0.hist(pred_lead_pt,bins=bins,histtype='step',color='red',lw=2,label='Sumpool')
freq_tru, bins, _ = ax0.hist(tru_lead_pt,bins=bins,histtype='step',color='green',lw=2,label='Target Jets')


lines = [r"$p_T^{\text{uncalib}} > 20 $GeV", r"$|\eta| < 2.1$"]
multi_line_text = '\n'.join(lines)
ax0.text(0.05, 0.78, multi_line_text, transform=ax0.transAxes, fontsize="small", fontname='TeX Gyre Heros')
ax0.text(0.05, 0.87, r"Dijet MC JZ4, <$\mu$>=200", transform=ax0.transAxes, fontsize="small", fontname='TeX Gyre Heros')
ax0.set_title('Leading Jet', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.01, 0.57),fontsize="small")

hep.atlas.label(ax=ax0,label='Internal',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Lead jet pT (GeV)')
f.savefig(save_folder + f'/jet_pt_lead.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()
print()



################################################################
# eta total
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_pred_eta,bins=25,histtype='step',color='red',lw=2.0,label='Sumpool')
freq_pbox, bins, _   = ax0.hist(total_pbox_eta2sig,bins=bins,histtype='step',color='indigo',lw=2.0,label='W. Circle')
freq_tru, bins, _ = ax0.hist(total_tru_eta,bins=bins,histtype='step',color='green',lw=2.0,label='Target Jets')

# ax0.set_title('Pseudorapidity Total', fontsize=16, fontfamily="TeX Gyre Heros")

lines = [r"$p_T^{\text{uncalib}} > 20 $GeV", r"$|\eta| < 2.1$"]
multi_line_text = '\n'.join(lines)
ax0.text(0.82, 0.72, multi_line_text, transform=ax0.transAxes, fontsize="small", fontname='TeX Gyre Heros')
ax0.text(0.02, 0.87, r"Dijet MC JZ4, <$\mu$>=200", transform=ax0.transAxes, fontsize="small", fontname='TeX Gyre Heros')
ax0.legend(loc='lower left',bbox_to_anchor=(0.74, 0.74),fontsize="medium")
hep.atlas.label(ax=ax0,label='Internal',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='$\eta$',ylabel='Number of Jets')
f.savefig(save_folder + f'/jet_eta_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()



f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 

freq_tru, bins, _ = ax[0].hist(total_tru_eta,bins=bins,histtype='step',color='green',lw=2.0,label='Target Jets')
freq_pred, bins, _   = ax[0].hist(total_pred_eta,bins=25,histtype='step',color='red',ls='dotted',lw=2.0,label='Sumpool')
freq_pbox, bins, _   = ax[0].hist(total_pbox_eta2sig,bins=bins,histtype='step',color='indigo',ls='dashed',lw=2.0,label='Weighted Circle')

bin_centers = (bins[:-1] + bins[1:]) / 2
ratio_pred_tru = np.divide(freq_pred, freq_tru, out=np.zeros_like(freq_pred), where=freq_tru != 0)
ratio_pbox_tru = np.divide(freq_pbox, freq_tru, out=np.zeros_like(freq_pbox), where=freq_tru != 0)
ax[1].plot(bin_centers, ratio_pred_tru, color='red', marker='o',ms=2,ls='dotted', label='Sumpool / Target Jets')
ax[1].plot(bin_centers, ratio_pbox_tru, color='indigo', marker='o',ms=2,ls='dashed', label='Weighted circle / Target Jets')
ax[1].axhline(y=1, color='green', linestyle='--', lw=2)

ax[0].legend(loc='lower left',bbox_to_anchor=(0.34, 0.01),fontsize="medium")
ax[0].set(yscale='log',ylabel='Number of Jets')
y_ticks = ax[0].yaxis.get_major_ticks()
y_ticks[0].label1.set_visible(False)
ax[1].set(xlabel='$\eta$',ylabel='Ratio')
ax[1].set(ylim=(0.25,1.5))
f.savefig(save_folder + f'/jet_eta_total_new.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()




# eta w/ errors
f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
bins = np.linspace(-2.1, 2.1, 25)
bin_centers = (bins[:-1] + bins[1:]) / 2

freq_tru, bins, _ = ax[0].hist(total_tru_eta,bins=bins,histtype='step',color='green',lw=2,label='Target Jets')
freq_pred, bins, _   = ax[0].hist(total_pred_eta,bins=bins,histtype='step',color='red',lw=2,label='Sumpool')#,ls='dotted')
freq_pbox, bins, _   = ax[0].hist(total_pbox_eta2sig,bins=bins,histtype='step',color='indigo',lw=2,label='Weighted circle')#,ls='dashed')

# stat. unc sqrt(counts)
tru_errors = np.sqrt(freq_tru)
pbox_errors = np.sqrt(freq_pbox)
pred_errors = np.sqrt(freq_pred)

ax[0].errorbar(bin_centers, freq_tru, yerr=tru_errors, color='green', ls='none')
ax[0].errorbar(bin_centers, freq_pbox, yerr=pbox_errors, color='indigo', ls='none')
ax[0].errorbar(bin_centers, freq_pred, yerr=pred_errors, color='red', ls='none')

ratio_pred_tru = np.divide(freq_pred, freq_tru, out=np.zeros_like(freq_pred), where=freq_tru != 0)
ratio_pbox_tru = np.divide(freq_pbox, freq_tru, out=np.zeros_like(freq_pbox), where=freq_tru != 0)
ax[1].hlines(y=1, xmin=-2.1, xmax=2.1, color='green', linestyle='-', lw=1.6)

# uncertainty on the ratio
# ratio_errors1 = ratio_pred_tru * np.sqrt((pred_errors / freq_pred) ** 2 + (tru_errors / freq_tru) ** 2)
ratio_pred_erro = np.sqrt((pred_errors / freq_tru) ** 2 + (freq_pred * tru_errors / freq_tru**2)**2)
ratio_pbox_erro = np.sqrt((pbox_errors / freq_tru) ** 2 + (freq_pbox * tru_errors / freq_tru**2)**2)
# ax[1].fill_between(bin_centers, ratio_pred_tru - ratio_pred_erro, ratio_pred_tru + ratio_pred_erro, alpha=0.5, edgecolor='crimson', facecolor='red')
ax[1].fill_between(bin_centers, ratio_pred_tru - ratio_pred_erro, ratio_pred_tru + ratio_pred_erro, alpha=0.5, edgecolor='crimson', facecolor='red')
ax[1].fill_between(bin_centers, ratio_pbox_tru - ratio_pbox_erro, ratio_pbox_tru + ratio_pbox_erro, alpha=0.5, edgecolor='purple', facecolor='indigo')

# new line stuff
target_line = matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
pred_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
pbox_line =  matplotlib.lines.Line2D([0], [0], color='indigo', lw=3)  
ax[0].legend([target_line,pred_line,pbox_line], ['Target jets', 'Sumpool', 'Weighted Circle'], loc='lower left', bbox_to_anchor=(0.34, 0.01), fontsize=12) 

# ax[0].legend(loc='lower left',bbox_to_anchor=(0.34, 0.01),fontsize="medium")
ax[0].set(yscale='log',ylabel='Number of jets',ylim=(1e2,1e4),xlim=(-3,3))
y_ticks = ax[0].yaxis.get_major_ticks()
y_ticks[0].label1.set_visible(False)
ax[1].set(xlabel='$\eta$')
ax[1].set(ylim=(0.0,1.5))
ax[1].set_yticks(np.arange(0.0,1.5,0.5))
# new_yticklabels = [f'{int(tick)}' for tick in np.arange(0.5,2.0,0.5)]  
new_yticklabels = ["0.0","0.5","1"]  
ax[1].set_yticklabels(new_yticklabels)
f.savefig(save_folder + f'/jet_eta_total_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()




# eta match
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(match_pred_eta,bins=25,histtype='step',color='red',lw=2,label='Sumpool')
freq_pbox, bins, _   = ax0.hist(match_pbox_eta,bins=bins,histtype='step',color='indigo',lw=2.0,label='Weighted Circle')
freq_tru, bins, _ = ax0.hist(match_tru_eta,bins=bins,histtype='step',color='green',lw=2,label='Target Jets')

ax0.set_title('Pseudorapidity Matched Jets', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.74),fontsize="medium")
hep.atlas.label(ax=ax0,label='Internal',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet eta')
f.savefig(save_folder + f'/jet_eta_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

# eta unmatch
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(unmatch_pred_eta,bins=25,histtype='step',color='red',lw=2.0,label='Sumpool')
freq_pbox, bins, _   = ax0.hist(unmatch_pbox_eta,bins=bins,histtype='step',color='indigo',lw=2.0,label='Weighted Circle')
freq_tru, bins, _ = ax0.hist(unmatch_tru_eta,bins=bins,histtype='step',color='green',lw=2.0,label='Target Jets')

ax0.set_title('Pseudorapidity Unmatched Jets', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Internal',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet eta')
f.savefig(save_folder + f'/jet_eta_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()
print()

################################################################
# phi total
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_pred_phi,bins=25,histtype='step',color='red',lw=2.0,label='Sumpool')
freq_pbox, bins, _   = ax0.hist(total_pbox_phi2sig,bins=bins,histtype='step',color='indigo',lw=2.0,label='Weighted Circle')
freq_tru, bins, _ = ax0.hist(total_tru_phi,bins=bins,histtype='step',color='green',lw=2.0,label='Target Jets')


lines = [r"$p_T^{\text{uncalib}} > 20 $GeV", r"$|\eta| < 2.1$"]
multi_line_text = '\n'.join(lines)
ax0.text(0.05, 0.77, multi_line_text, transform=ax0.transAxes, fontsize="small", fontname='TeX Gyre Heros')
ax0.text(0.05, 0.87, r"Dijet MC JZ4, <$\mu$>=200", transform=ax0.transAxes, fontsize="small", fontname='TeX Gyre Heros')
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.74),fontsize="medium")

# ax0.set_title('Azimuth Total', fontsize=16, fontfamily="TeX Gyre Heros")
hep.atlas.label(ax=ax0,label='Internal',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='$\phi$', ylabel='Number of Jets')
f.savefig(save_folder + f'/jet_phi_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()



f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 

freq_tru, bins, _ = ax[0].hist(total_tru_phi,bins=bins,histtype='step',color='green',lw=2.0,label='Target Jets')
freq_pred, bins, _   = ax[0].hist(total_pred_phi,bins=25,histtype='step',color='red',ls='dotted',lw=2.0,label='Sumpool')
freq_pbox, bins, _   = ax[0].hist(total_pbox_phi2sig,bins=bins,histtype='step',color='indigo',ls='dashed',lw=2.0,label='Weighted Circle')

bin_centers = (bins[:-1] + bins[1:]) / 2
ratio_pred_tru = np.divide(freq_pred, freq_tru, out=np.zeros_like(freq_pred), where=freq_tru != 0)
ratio_pbox_tru = np.divide(freq_pbox, freq_tru, out=np.zeros_like(freq_pbox), where=freq_tru != 0)
ax[1].plot(bin_centers, ratio_pred_tru, color='red', marker='o',ms=2,ls='dotted', label='Sumpool / Target Jets')
ax[1].plot(bin_centers, ratio_pbox_tru, color='indigo', marker='o',ms=2,ls='dashed', label='Weighted circle / Target Jets')
ax[1].axhline(y=1, color='green', linestyle='--', lw=2)

ax[0].legend(loc='lower left',bbox_to_anchor=(0.65, 0.74),fontsize="medium")
ax[0].set(yscale='log',ylabel='Number of Jets')
y_ticks = ax[0].yaxis.get_major_ticks()
y_ticks[0].label1.set_visible(False)
ax[1].set(xlabel='$\phi$',ylabel='Ratio')
ax[1].set(ylim=(0.0,2.0))
f.savefig(save_folder + f'/jet_phi_total_new.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()




# phi w/ errors
f, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0}) 
bin_centers = (bins[:-1] + bins[1:]) / 2

freq_tru, bins, _ = ax[0].hist(total_tru_phi,bins=bins,histtype='step',color='green',lw=2,label='Target Jets')
freq_pred, bins, _   = ax[0].hist(total_pred_phi,bins=bins,histtype='step',color='red',lw=2,label='Sumpool')#,ls='dotted')
freq_pbox, bins, _   = ax[0].hist(total_pbox_phi2sig,bins=bins,histtype='step',color='indigo',lw=2,label='Weighted circle')#,ls='dashed')

# stat. unc sqrt(counts)
tru_errors = np.sqrt(freq_tru)
pbox_errors = np.sqrt(freq_pbox)
pred_errors = np.sqrt(freq_pred)

ax[0].errorbar(bin_centers, freq_tru, yerr=tru_errors, color='green', ls='none')
ax[0].errorbar(bin_centers, freq_pbox, yerr=pbox_errors, color='indigo', ls='none')
ax[0].errorbar(bin_centers, freq_pred, yerr=pred_errors, color='red', ls='none')

ratio_pred_tru = np.divide(freq_pred, freq_tru, out=np.zeros_like(freq_pred), where=freq_tru != 0)
ratio_pbox_tru = np.divide(freq_pbox, freq_tru, out=np.zeros_like(freq_pbox), where=freq_tru != 0)
ax[1].hlines(y=1, xmin=-np.pi, xmax=np.pi, color='green', linestyle='-', lw=1.6)

# uncertainty on the ratio
ratio_pred_erro = np.sqrt((pred_errors / freq_tru) ** 2 + (freq_pred * tru_errors / freq_tru**2)**2)
ratio_pbox_erro = np.sqrt((pbox_errors / freq_tru) ** 2 + (freq_pbox * tru_errors / freq_tru**2)**2)
# ax[1].fill_between(bin_centers, ratio_pred_tru - ratio_pred_erro, ratio_pred_tru + ratio_pred_erro, alpha=0.5, edgecolor='crimson', facecolor='red')
ax[1].fill_between(bin_centers[1:-1], ratio_pred_tru[1:-1] - ratio_pred_erro[1:-1], ratio_pred_tru[1:-1] + ratio_pred_erro[1:-1], alpha=0.5, edgecolor='crimson', facecolor='red')
ax[1].fill_between(bin_centers[1:-1], ratio_pbox_tru[1:-1] - ratio_pbox_erro[1:-1], ratio_pbox_tru[1:-1] + ratio_pbox_erro[1:-1], alpha=0.5, edgecolor='purple', facecolor='indigo')

# new line stuff
target_line = matplotlib.lines.Line2D([0], [0], color='green', lw=3) 
pred_line =  matplotlib.lines.Line2D([0], [0], color='red', lw=3)  
pbox_line =  matplotlib.lines.Line2D([0], [0], color='indigo', lw=3)  
ax[0].legend([target_line,pred_line,pbox_line], ['Target jets', 'Sumpool', 'Weighted Circle'], loc='lower left', bbox_to_anchor=(0.65, 0.78), fontsize=12) 

# ax[0].legend(loc='lower left',bbox_to_anchor=(0.65, 0.74),fontsize="medium")
ax[0].set(yscale='log',ylabel='Number of jets')
y_ticks = ax[0].yaxis.get_major_ticks()
y_ticks[0].label1.set_visible(False)
ax[1].set(xlabel='$\phi$')
ax[1].set(ylim=(0.0,1.5))
ax[1].set_yticks(np.arange(0.0,1.5,0.5))
# new_yticklabels = [f'{int(tick)}' for tick in np.arange(0.5,2.0,0.5)]  
new_yticklabels = ["","0.5","1"]  
print(new_yticklabels,np.arange(0.5,2.0,0.5))
ax[1].set_yticklabels(new_yticklabels)
f.savefig(save_folder + f'/jet_phi_total_unc.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()






# phi match
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(match_pred_phi,bins=25,histtype='step',color='red',lw=2.0,label='Sumpool')
freq_pbox, bins, _   = ax0.hist(match_pbox_phi,bins=bins,histtype='step',color='indigo',lw=2.0,label='Weighted Circle')
freq_tru, bins, _ = ax0.hist(match_tru_phi,bins=bins,histtype='step',color='green',lw=2.0,label='Target Jets')

ax0.set_title('Azimuth Match', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.65, 0.74),fontsize="medium")
hep.atlas.label(ax=ax0,label='Internal',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet phi')
f.savefig(save_folder + f'/jet_phi_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

# phi unmatch
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(unmatch_pred_phi,bins=25,histtype='step',color='red',lw=2.0,label='Sumpool')
freq_pbox, bins, _   = ax0.hist(unmatch_pbox_phi,bins=bins,histtype='step',color='indigo',lw=2.0,label='Weighted Circle')
freq_tru, bins, _ = ax0.hist(unmatch_tru_phi,bins=bins,histtype='step',color='green',lw=2.0,label='Target Jets')

ax0.set_title('Azimuth Unmatch', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Internal',data=False,lumi=None,loc=1)
ax0.set(yscale='log',xlabel='Jet phi')
f.savefig(save_folder + f'/jet_phi_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()
print()





# 2d sumpool vs weighted circle
print(total_pred_pt[:7])
print(total_pbox_ptadj2sig[:7])
print(np.sum(np.isnan(total_pred_pt)),np.sum(np.isnan(total_pbox_ptadj2sig)))
nan_mask = np.isnan(total_pbox_ptadj2sig)


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
hist = ax.hist2d(total_pred_pt[~nan_mask], total_pbox_ptadj2sig[~nan_mask], bins=[100, 100], cmap='cool',norm=matplotlib.colors.LogNorm())
fig.colorbar(hist[3], ax=ax)
ax.set_title('All Predicted Jets', fontsize=18, fontfamily="TeX Gyre Heros")
ax.set_xlabel('Sumpool Jet pT', fontsize=14, fontfamily="TeX Gyre Heros")
ax.set_ylabel('Weighted Circle Jet pT', fontsize=14, fontfamily="TeX Gyre Heros")
plt.savefig(save_folder + f'/2d_pt_scatter.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()




print()
match_pred_pt = total_pred_pt[np.where(total_pred_matched==1)]
match_pred_scr = total_pred_scores[np.where(total_pred_matched==1)]

match_pbox_pt = total_pbox_ptadj2sig[np.where(total_pbox_matched==1)]
match_pbox_scr = total_pbox_scores[np.where(total_pbox_matched==1)]

match_tru_pt = total_tru_pt[np.where(total_tru_matched==1)]



# comparing matched sumpool and truth 
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
hist = ax.hist2d(total_pred_pt[~nan_mask], total_pbox_ptadj2sig[~nan_mask], bins=[100, 100], cmap='cool',norm=matplotlib.colors.LogNorm())
fig.colorbar(hist[3], ax=ax)
ax.set_title('All Predicted Jets', fontsize=18, fontfamily="TeX Gyre Heros")
ax.set_xlabel('Sumpool Jet pT', fontsize=14, fontfamily="TeX Gyre Heros")
ax.set_ylabel('Weighted Circle Jet pT', fontsize=14, fontfamily="TeX Gyre Heros")
plt.savefig(save_folder + f'/2d_pt_scatter.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()









########################################################################
# Trigger efficiencies    ########################################################################
########################################################################
print("Trigger efficiencies")

def get_ratio(numer,denom):
    result = np.zeros_like(numer, dtype=float)
    non_zero_indices = denom != 0
    # Perform element-wise division, handling zeros in the denominator
    result[non_zero_indices] = numer[non_zero_indices] / denom[non_zero_indices]
    return result


def clopper_pearson(x, n, alpha=0.05):
    """
    Estimate the confidence interval for a sampled Bernoulli random
    variable.
    `x` is the number of successes and `n` is the number trials (x <=
    n). `alpha` is the confidence level (i.e., the true probability is
    inside the confidence interval with probability 1-alpha). The
    function returns a `(low, high)` pair of numbers indicating the
    interval on the probability.
    https://root.cern.ch/doc/master/classTEfficiency.html#ae80c3189bac22b7ad15f57a1476ef75b
    """

    lo = scipy.stats.beta.ppf(alpha / 2, x, n - x + 1)
    hi = scipy.stats.beta.ppf(1 - alpha / 2, x + 1, n - x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi

def get_errorbars(success_array, total_array, alpha=0.05):
    """
    Function to calculate and return errorbars in matplotlib preferred format.
    Current usage of Clopper-Pearon may generalise later. Function currently
    returns interval.
    'success_array' is the count of each histogram bins after(!) cut applied
    'total_array' is the count of each histogram before trigger applied
    'alpha' is the confidence level
    Returns errors array to be used in ax.errorbars kwarg yerr
    """
    confidence_intervals = []
    
    lo, hi = np.vectorize(clopper_pearson)(success_array, total_array, alpha)
    
    confidence_intervals = np.array([lo, hi]).T
    
    zeros_mask = total_array == 0
    lower_error_bars = np.where(zeros_mask, lo, success_array/total_array - lo)
    upper_error_bars = np.where(zeros_mask, hi, hi - success_array/total_array)
    
    errors = np.array([lower_error_bars, upper_error_bars])
    
    return errors



def leading_jet_pt(list_of_jet_pts_in_event):
    try:
        return max(list_of_jet_pts_in_event)
    except ValueError:
        #Doesn't have enough (or any) jets, automatically lost in cut
        return np.nan

def nth_leading_jet_pt(list_of_jet_pts_in_event,n):
    try:
        return sorted(list_of_jet_pts_in_event,reverse=True)[n-1]
    except IndexError or ValueError:
        # Doesn't have enough (or any) jets, automatically lost in cut
        return np.nan





event_tru_pt = load_object(metrics_folder+"/box_metrics/tboxes_pt.pkl")
event_pred_pt = load_object(metrics_folder+"/box_metrics/pboxes_pt.pkl")
event_pbox_ptadj2sig = load_object(metrics_folder+"/phys_metrics/pboxes_pT_adj_2sig.pkl")
tru_lead_pt = np.array([leading_jet_pt(x) for x in event_tru_pt])
pred_lead_pt = np.array([leading_jet_pt(x) for x in event_pred_pt])
pbox_lead_pt = np.array([leading_jet_pt(x)*1/1000 for x in event_pbox_ptadj2sig])

lead_jet_pt_cut = 450 # GeV
#make a trigger decision based on antikt jet
trig_decision_akt = np.argwhere(tru_lead_pt>lead_jet_pt_cut).T[0]
trig_decision_pred = np.argwhere(pred_lead_pt>lead_jet_pt_cut).T[0]
trig_decision_pbox = np.argwhere(pbox_lead_pt>lead_jet_pt_cut).T[0]




trig_save_loc = save_folder + f"/trig/"
if not os.path.exists(trig_save_loc):
    os.makedirs(trig_save_loc)
    
start,end = 350,750
step = 10
bins = np.arange(start, end, step)

f,ax = plt.subplots(3,1,figsize=(10,16))
n_akt,bins,_ = ax[0].hist(tru_lead_pt,bins=bins,histtype='step',label='Anti-kt jetConstitScale')
ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
ax[0].set(xlabel="Leading EMTopo (Offline) jet pT (GeV)",ylabel='Freq.')
ax[0].legend()

n2_akt,bins,_ = ax[1].hist(tru_lead_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="gold")
n2_p,_,_ = ax[1].hist(tru_lead_pt[trig_decision_pred],bins=bins,histtype='step',label='Sumpool',color="red")
n2_pb,_,_ = ax[1].hist(tru_lead_pt[trig_decision_pbox],bins=bins,histtype='step',label='Weighted circle',color="indigo")
ax[1].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
ax[1].set_title(f'After {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
ax[1].set(xlabel="Leading jet pT (GeV)",ylabel='Freq.')
ax[1].legend()

ax[2].axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
with np.errstate(divide='ignore', invalid='ignore'):
    step_eff = get_ratio(n2_akt,n_akt)
    step_err = get_errorbars(n2_akt,n_akt)

    pred_eff = get_ratio(n2_p,n_akt)
    pred_err = get_errorbars(n2_p,n_akt)

    pbox_eff = get_ratio(n2_pb,n_akt)
    pbox_err = get_errorbars(n2_pb,n_akt)

bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]

ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='gold')
ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Sumpool',color='red')
ax[2].errorbar(bin_centers,pbox_eff,xerr=bin_width/2,yerr=pbox_err,elinewidth=0.4,marker='.',ls='none',label='WC',color='indigo')
ax[2].grid()
ax[2].set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency')
ax[2].legend(loc='lower right')
# hep.atlas.label(ax=ax[2],label='Internal',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
f.savefig(trig_save_loc + f'/eff_plot_leading{lead_jet_pt_cut:.0f}GeV_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



f,a = plt.subplots(1,1,figsize=(8,8))
a.axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3)
a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Sumpool',color='red')
a.errorbar(bin_centers,pbox_eff,xerr=bin_width/2,yerr=pbox_err,elinewidth=0.4,marker='.',ls='none',label='WC',color='indigo')
a.grid()
a.set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency')
a.legend(loc='lower left',bbox_to_anchor=(0.7, 0.07),fontsize="medium")
hep.atlas.label(ax=a,label='Internal',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_leading{lead_jet_pt_cut:.0f}GeV.{image_format}',dpi=400,format=image_format,bbox_inches="tight")







################
# Nth Leading Jet
nth_jet = 4
nth_lead_jet_pt_cut = 55 

tru_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in event_tru_pt])
pred_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in event_pred_pt])
pbox_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet)*1/1000 for x in event_pbox_ptadj2sig])

trig_decision_akt = np.argwhere(tru_nlead_pt>nth_lead_jet_pt_cut).T[0]
trig_decision_pred = np.argwhere(pred_nlead_pt>nth_lead_jet_pt_cut).T[0]
trig_decision_pbox = np.argwhere(pbox_nlead_pt>nth_lead_jet_pt_cut).T[0]

start,end = 20,140
step = 10
bins = np.arange(start, end, step)

f,ax = plt.subplots(3,1,figsize=(10,15))
n_akt,bins,_ = ax[0].hist(tru_nlead_pt,bins=bins,histtype='step',label='Anti-kt jetConstitScale')
ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Freq.')
ax[0].grid()
ax[0].legend()

n2_akt,bins,_ = ax[1].hist(tru_nlead_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="gold")
n2_p,_,_ = ax[1].hist(tru_nlead_pt[trig_decision_pred],bins=bins,histtype='step',label='Sumpool',color="red")
n2_pb,_,_ = ax[1].hist(tru_nlead_pt[trig_decision_pbox],bins=bins,histtype='step',label='WC',color="indigo")
ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Freq.')
ax[1].legend()

ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
with np.errstate(divide='ignore', invalid='ignore'):
    step_eff = get_ratio(n2_akt,n_akt)
    step_err = get_errorbars(n2_akt,n_akt)
    
    pred_eff = get_ratio(n2_p,n_akt)
    pred_err = get_errorbars(n2_p,n_akt)
    
    pbox_eff = get_ratio(n2_pb,n_akt)
    pbox_err = get_errorbars(n2_pb,n_akt)

bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]

ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='gold')
ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Sumpool',color='red')
ax[2].errorbar(bin_centers,pbox_eff,xerr=bin_width/2,yerr=pbox_err,elinewidth=0.4,marker='.',ls='none',label='WC',color='indigo')
ax[2].grid()
ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
ax[2].legend(loc='lower right')
hep.atlas.label(ax=ax[2],label='Internal',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
f.savefig(trig_save_loc + f'/eff_plot_{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



f,a = plt.subplots(1,1,figsize=(8,8))
a.axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3)
a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Sumpool',color='red')
a.errorbar(bin_centers,pbox_eff,xerr=bin_width/2,yerr=pbox_err,elinewidth=0.4,marker='.',ls='none',label='WC',color='indigo')
a.grid()
a.set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
a.legend(loc='lower left',bbox_to_anchor=(0.7, 0.07),fontsize="medium")
hep.atlas.label(ax=a,label='Internal',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_{nth_jet}_leading{nth_lead_jet_pt_cut:.0f}GeV.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


################
# Nth Leading Jet
nth_jet = 3
nth_lead_jet_pt_cut = 90 

tru_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in event_tru_pt])
pred_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in event_pred_pt])
pbox_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet)*1/1000 for x in event_pbox_ptadj2sig])

trig_decision_akt = np.argwhere(tru_nlead_pt>nth_lead_jet_pt_cut).T[0]
trig_decision_pred = np.argwhere(pred_nlead_pt>nth_lead_jet_pt_cut).T[0]
trig_decision_pbox = np.argwhere(pbox_nlead_pt>nth_lead_jet_pt_cut).T[0]

start,end = 40,250
step = 10
bins = np.arange(start, end, step)

f,ax = plt.subplots(3,1,figsize=(10,15))
n_akt,bins,_ = ax[0].hist(tru_nlead_pt,bins=bins,histtype='step',label='Anti-kt jetConstitScale')
ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Freq.')
ax[0].grid()
ax[0].legend()

n2_akt,bins,_ = ax[1].hist(tru_nlead_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="gold")
n2_p,_,_ = ax[1].hist(tru_nlead_pt[trig_decision_pred],bins=bins,histtype='step',label='Sumpool',color="red")
n2_pb,_,_ = ax[1].hist(tru_nlead_pt[trig_decision_pbox],bins=bins,histtype='step',label='WC',color="indigo")
ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Freq.')
ax[1].legend()

ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
with np.errstate(divide='ignore', invalid='ignore'):
    step_eff = get_ratio(n2_akt,n_akt)
    step_err = get_errorbars(n2_akt,n_akt)
    
    pred_eff = get_ratio(n2_p,n_akt)
    pred_err = get_errorbars(n2_p,n_akt)
    
    pbox_eff = get_ratio(n2_pb,n_akt)
    pbox_err = get_errorbars(n2_pb,n_akt)

bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]

ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='gold')
ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Sumpool',color='red')
ax[2].errorbar(bin_centers,pbox_eff,xerr=bin_width/2,yerr=pbox_err,elinewidth=0.4,marker='.',ls='none',label='WC',color='indigo')
ax[2].grid()
ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
ax[2].legend(loc='lower right')
hep.atlas.label(ax=ax[2],label='Internal',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
f.savefig(trig_save_loc + f'/eff_plot_{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



f,a = plt.subplots(1,1,figsize=(8,8))
a.axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3)
a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Sumpool',color='red')
a.errorbar(bin_centers,pbox_eff,xerr=bin_width/2,yerr=pbox_err,elinewidth=0.4,marker='.',ls='none',label='WC',color='indigo')
a.grid()
a.set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
a.legend(loc='lower left',bbox_to_anchor=(0.7, 0.07),fontsize="medium")
hep.atlas.label(ax=a,label='Internal',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_{nth_jet}_leading{nth_lead_jet_pt_cut:.0f}GeV.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



################
# Nth Leading Jet
nth_jet = 2
nth_lead_jet_pt_cut = 300 # 400GeV

tru_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in event_tru_pt])
pred_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in event_pred_pt])
pbox_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet)*1/1000 for x in event_pbox_ptadj2sig])

trig_decision_akt = np.argwhere(tru_nlead_pt>nth_lead_jet_pt_cut).T[0]
trig_decision_pred = np.argwhere(pred_nlead_pt>nth_lead_jet_pt_cut).T[0]
trig_decision_pbox = np.argwhere(pbox_nlead_pt>nth_lead_jet_pt_cut).T[0]

start,end = 100,500
step = 10
bins = np.arange(start, end, step)

f,ax = plt.subplots(3,1,figsize=(10,15))
n_akt,bins,_ = ax[0].hist(tru_nlead_pt,bins=bins,histtype='step',label='Anti-kt jetConstitScale')
ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Freq.')
ax[0].grid()
ax[0].legend()

n2_akt,bins,_ = ax[1].hist(tru_nlead_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="gold")
n2_p,_,_ = ax[1].hist(tru_nlead_pt[trig_decision_pred],bins=bins,histtype='step',label='Sumpool',color="red")
n2_pb,_,_ = ax[1].hist(tru_nlead_pt[trig_decision_pbox],bins=bins,histtype='step',label='WC',color="indigo")
ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Freq.')
ax[1].legend()

ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
with np.errstate(divide='ignore', invalid='ignore'):
    step_eff = get_ratio(n2_akt,n_akt)
    step_err = get_errorbars(n2_akt,n_akt)
    
    pred_eff = get_ratio(n2_p,n_akt)
    pred_err = get_errorbars(n2_p,n_akt)
    
    pbox_eff = get_ratio(n2_pb,n_akt)
    pbox_err = get_errorbars(n2_pb,n_akt)

bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]

ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='gold')
ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Sumpool',color='red')
ax[2].errorbar(bin_centers,pbox_eff,xerr=bin_width/2,yerr=pbox_err,elinewidth=0.4,marker='.',ls='none',label='WC',color='indigo')
ax[2].grid()
ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
ax[2].legend(loc='lower right')
hep.atlas.label(ax=ax[2],label='Internal',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
f.savefig(trig_save_loc + f'/eff_plot_{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



f,a = plt.subplots(1,1,figsize=(8,8))
a.axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3)
a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Sumpool',color='red')
a.errorbar(bin_centers,pbox_eff,xerr=bin_width/2,yerr=pbox_err,elinewidth=0.4,marker='.',ls='none',label='WC',color='indigo')
a.grid()
a.set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
a.legend(loc='lower left',bbox_to_anchor=(0.7, 0.07),fontsize="medium")
hep.atlas.label(ax=a,label='Internal',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_{nth_jet}_leading{nth_lead_jet_pt_cut:.0f}GeV.{image_format}',dpi=400,format=image_format,bbox_inches="tight")














def get_eff(leading_jet_pt_cut):
    event_tru_pt = load_object(metrics_folder+"box_metrics/tboxes_pt.pkl")
    event_pred_pt = load_object(metrics_folder+"box_metrics/pboxes_pt.pkl")
    antikt_lead_pt = np.array([leading_jet_pt(x) for x in event_tru_pt])
    pred_lead_pt = np.array([leading_jet_pt(x) for x in event_pred_pt])

    #make a trigger decision based on antikt jet
    trig_decision_akt = np.argwhere(antikt_lead_pt>leading_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(pred_lead_pt>leading_jet_pt_cut).T[0]

    start,end = 300,700
    step = 10
    bins = np.arange(start, end, step)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    n_akt,bins = np.histogram(antikt_lead_pt,bins=bins)
    n2_p, bins = np.histogram(antikt_lead_pt[trig_decision_pred], bins=bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        pred_eff = get_ratio(n2_p,n_akt)
        pred_err = get_errorbars(n2_p,n_akt)

    return bin_centers, pred_eff, pred_err




xs,ys,yerr = get_eff(450)
xs2,ys2,yerr2 = get_eff(550)
xs3,ys3,yerr3 = get_eff(650)

# f,a = plt.subplots(1,1,figsize=(8,8))
# a.errorbar(xs,ys,xerr=(xs[1]-xs[0])/2,yerr=yerr,elinewidth=0.4,marker='.',ms=12,lw=2,ls='none',label='450 GeV Threshold',color='red')
# a.errorbar(xs2,ys2,xerr=(xs[1]-xs[0])/2,yerr=yerr2,elinewidth=0.4,marker='.',ms=12,lw=2,ls='none',label='550 GeV Threshold',color='coral')
# # a.errorbar(xs3,ys3,xerr=(xs[1]-xs[0])/2,yerr=yerr3,elinewidth=0.4,marker='.',ls='none',label='650 GeV Cut',color='peru')

# a.set(xlabel="Uncalibrated leading jet $p_T$ [GeV]",ylabel='Trigger Efficiency')
# a.set_ylim((0.0,1.2))
# a.legend(loc='upper left', bbox_to_anchor=(0.63, 0.99),fontsize=12)
# # hep.atlas.label(ax=a,label='Internal',data=False,lumi=None,loc=1)
# f.subplots_adjust(hspace=0.4)
# f.savefig(trig_save_loc + f'/eff_plot_leading_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


f,a = plt.subplots(1,1,figsize=(8,8))
a.errorbar(xs,ys,xerr=(xs[1]-xs[0])/2,yerr=yerr,elinewidth=0.9,marker='s',ms=8,lw=2,ls='none',label='450 GeV threshold',color='deepskyblue')
a.errorbar(xs2,ys2,xerr=(xs[1]-xs[0])/2,yerr=yerr2,elinewidth=0.9,marker='^',ms=8,lw=2,ls='none',label='550 GeV threshold',color='magenta')
# a.errorbar(xs3,ys3,xerr=(xs[1]-xs[0])/2,yerr=yerr3,elinewidth=0.4,marker='.',ls='none',label='650 GeV Cut',color='peru')

a.set(xlabel="Uncalibrated leading jet $p_T$ [GeV]",ylabel='Trigger Efficiency')
a.set_ylim((0.0,1.2))
a.legend(loc='upper left', bbox_to_anchor=(0.54, 0.99),fontsize=15)
# hep.atlas.label(ax=a,label='Internal',data=False,lumi=None,loc=1)
tick_labels = a.get_xticklabels()
tick_positions = a.get_xticks()
for label in tick_labels:
    label.set_verticalalignment('bottom')  
    label.set_y(label.get_position()[1] - 0.05)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_leading_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")






def get_2eff(subleading_jet_pt_cut):
    event_tru_pt = load_object(metrics_folder+"box_metrics/tboxes_pt.pkl")
    event_pred_pt = load_object(metrics_folder+"box_metrics/pboxes_pt.pkl")
    antikt_2lead_pt = np.array([nth_leading_jet_pt(x,2) for x in event_tru_pt])
    pred_2lead_pt = np.array([nth_leading_jet_pt(x,2) for x in event_pred_pt])

    #make a trigger decision based on antikt jet
    trig_decision_akt = np.argwhere(antikt_2lead_pt>subleading_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(pred_2lead_pt>subleading_jet_pt_cut).T[0]

    start,end = 200,600
    step = 10
    bins = np.arange(start, end, step)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    n_akt,bins = np.histogram(antikt_2lead_pt,bins=bins)
    n2_p, bins = np.histogram(antikt_2lead_pt[trig_decision_pred], bins=bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        pred_eff = get_ratio(n2_p,n_akt)
        pred_err = get_errorbars(n2_p,n_akt)

    return bin_centers, pred_eff, pred_err


xs,ys,yerr = get_2eff(350)
xs2,ys2,yerr2 = get_2eff(450)
xs3,ys3,yerr3 = get_2eff(550)

# f,a = plt.subplots(1,1,figsize=(8,8))
# a.errorbar(xs,ys,xerr=(xs[1]-xs[0])/2,yerr=yerr,elinewidth=0.4,marker='.',ms=12,ls='none',label='350 GeV Threshold',color='red')
# a.errorbar(xs2,ys2,xerr=(xs[1]-xs[0])/2,yerr=yerr2,elinewidth=0.4,marker='.',ms=12,ls='none',label='450 GeV Threshold',color='coral')
# # a.errorbar(xs3,ys3,xerr=(xs[1]-xs[0])/2,yerr=yerr3,elinewidth=0.4,marker='.',ls='none',label='550 GeV Cut',color='peru')

# a.set(xlabel="Uncalibrated subleading jet $p_T$ [GeV]",ylabel='Trigger Efficiency')
# a.set_ylim((0.0,1.2))
# a.legend(loc='upper left', bbox_to_anchor=(0.63, 0.99),fontsize=12)
# # hep.atlas.label(ax=a,label='Internal',data=False,lumi=None,loc=1)
# f.subplots_adjust(hspace=0.4)
# f.savefig(trig_save_loc + f'/eff_plot_2leading_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

f,a = plt.subplots(1,1,figsize=(8,8))
a.errorbar(xs,ys,xerr=(xs[1]-xs[0])/2,yerr=yerr,elinewidth=0.9,marker='v',ms=8,ls='none',label='350 GeV threshold',color='orange')
a.errorbar(xs2,ys2,xerr=(xs[1]-xs[0])/2,yerr=yerr2,elinewidth=0.9,marker='o',ms=8,ls='none',label='450 GeV threshold',color='blueviolet')
# a.errorbar(xs3,ys3,xerr=(xs[1]-xs[0])/2,yerr=yerr3,elinewidth=0.4,marker='.',ls='none',label='550 GeV Cut',color='peru')

a.set(xlabel="Uncalibrated subleading jet $p_T$ [GeV]",ylabel='Trigger Efficiency')
a.set_ylim((0.0,1.2))
a.legend(loc='upper left', bbox_to_anchor=(0.54, 0.99),fontsize=15)
# hep.atlas.label(ax=a,label='Internal',data=False,lumi=None,loc=1)
tick_labels = a.get_xticklabels()
tick_positions = a.get_xticks()
for label in tick_labels:
    label.set_verticalalignment('bottom')  
    label.set_y(label.get_position()[1] - 0.05)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_2leading_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")







def get_3eff(subleading_jet_pt_cut):
    event_tru_pt = load_object(metrics_folder+"box_metrics/tboxes_pt.pkl")
    event_pred_pt = load_object(metrics_folder+"box_metrics/pboxes_pt.pkl")
    antikt_2lead_pt = np.array([nth_leading_jet_pt(x,3) for x in event_tru_pt])
    pred_2lead_pt = np.array([nth_leading_jet_pt(x,3) for x in event_pred_pt])

    #make a trigger decision based on antikt jet
    trig_decision_akt = np.argwhere(antikt_2lead_pt>subleading_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(pred_2lead_pt>subleading_jet_pt_cut).T[0]

    start,end = 100,600
    step = 10
    bins = np.arange(start, end, step)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    n_akt,bins = np.histogram(antikt_2lead_pt,bins=bins)
    n2_p, bins = np.histogram(antikt_2lead_pt[trig_decision_pred], bins=bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        pred_eff = get_ratio(n2_p,n_akt)
        pred_err = get_errorbars(n2_p,n_akt)

    return bin_centers, pred_eff, pred_err


xs,ys,yerr = get_3eff(150)
xs2,ys2,yerr2 = get_3eff(225)
xs3,ys3,yerr3 = get_3eff(300)

f,a = plt.subplots(1,1,figsize=(8,8))
a.errorbar(xs,ys,xerr=(xs[1]-xs[0])/2,yerr=yerr,elinewidth=0.4,marker='.',ls='none',label='150 GeV Cut',color='red')
a.errorbar(xs2,ys2,xerr=(xs[1]-xs[0])/2,yerr=yerr2,elinewidth=0.4,marker='.',ls='none',label='225 GeV Cut',color='coral')
a.errorbar(xs3,ys3,xerr=(xs[1]-xs[0])/2,yerr=yerr3,elinewidth=0.4,marker='.',ls='none',label='300 GeV Cut',color='peru')
a.grid()
a.set(xlabel="3rd leading jet pT (GeV)",ylabel='Efficiency')
a.legend(loc='upper left', bbox_to_anchor=(0.75, 0.99),fontsize=7)
hep.atlas.label(ax=a,label='Internal',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_3leading_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")




def get_Neff(nleading_jet_pt_cut,nthjet):
    event_tru_pt = load_object(metrics_folder+"box_metrics/tboxes_pt.pkl")
    event_pred_pt = load_object(metrics_folder+"box_metrics/pboxes_pt.pkl")
    antikt_nlead_pt = np.array([nth_leading_jet_pt(x,nthjet) for x in event_tru_pt])
    pred_nlead_pt = np.array([nth_leading_jet_pt(x,nthjet) for x in event_pred_pt])

    #make a trigger decision based on antikt jet
    trig_decision_akt = np.argwhere(antikt_nlead_pt>nleading_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(pred_nlead_pt>nleading_jet_pt_cut).T[0]

    start,end = 20,250
    step = 10
    bins = np.arange(start, end, step)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    n_akt,bins = np.histogram(antikt_nlead_pt,bins=bins)
    n2_p, bins = np.histogram(antikt_nlead_pt[trig_decision_pred], bins=bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        pred_eff = get_ratio(n2_p,n_akt)
        pred_err = get_errorbars(n2_p,n_akt)

    return bin_centers, pred_eff, pred_err


xs,ys,yerr = get_Neff(75,4)
xs2,ys2,yerr2 = get_Neff(100,4)
xs3,ys3,yerr3 = get_Neff(150,4)

f,a = plt.subplots(1,1,figsize=(8,8))
a.errorbar(xs,ys,xerr=(xs[1]-xs[0])/2,yerr=yerr,elinewidth=0.4,marker='.',ls='none',label='75 GeV Cut',color='red')
a.errorbar(xs2,ys2,xerr=(xs[1]-xs[0])/2,yerr=yerr2,elinewidth=0.4,marker='.',ls='none',label='100 GeV Cut',color='coral')
a.errorbar(xs3,ys3,xerr=(xs[1]-xs[0])/2,yerr=yerr3,elinewidth=0.4,marker='.',ls='none',label='150 GeV Cut',color='peru')
a.grid()
a.set(xlabel="3rd leading jet pT (GeV)",ylabel='Efficiency',ylim=(-0.2,1.2))
a.legend(loc='upper left', bbox_to_anchor=(0.75, 0.99),fontsize=7)
hep.atlas.label(ax=a,label='Internal',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_nleading_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")


































print("Fake rate vs accuracy")
eff_save_loc = save_folder + f"/trig/"
if not os.path.exists(eff_save_loc):
    os.makedirs(eff_save_loc)


bin_edges = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, max(total_pred_pt)]
bin_edges1 = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, max(total_pred_pt)]
bin_centers1 = bin_edges1[:-1] + 0.5 * np.diff(bin_edges1)
bin_width1 = np.diff(bin_edges)
n_matched_preds = []
n_truth = []
percentage_matched_in_pt = []
for bin_idx in range(len(bin_edges1)-1):
    bin_mask_tru = (bin_edges1[bin_idx]<total_tru_pt) & (total_tru_pt<bin_edges1[bin_idx+1])
    num_truth = len(total_tru_matched[bin_mask_tru])
    num_matched_truth = sum(total_tru_matched[bin_mask_tru])
    
    print('num_truth',num_truth,'num_matched_truth',num_matched_truth, num_matched_truth/num_truth)
    n_matched_preds.append(num_matched_truth)
    n_truth.append(num_truth)
    percentage_matched_in_pt.append(num_matched_truth/num_truth)


# bin_edges = [-100, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, max(total_pred_pt)]
# bin_edges = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, max(total_pred_pt)]
# bin_edges = [20, 30, 40, 50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200, 225, 250]
bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
bin_width = np.diff(bin_edges)
percentage_unmatched_in_pt = []
n_unmatched_preds = []
n_unmatched_pboxes = []
perc_unmatched_boxes = []
n_preds,n_pboxes = [],[]
for bin_idx in range(len(bin_edges)-1):
    # print(bin_edges[bin_idx],bin_edges[bin_idx+1])
    bin_mask = (bin_edges[bin_idx]<total_pred_pt) & (total_pred_pt<bin_edges[bin_idx+1])
    num_predictions = len(total_pred_matched[bin_mask])
    num_matched_predictions = sum(total_pred_matched[bin_mask])
    print('num_predictions',num_predictions,'num_matched_predictions',num_matched_predictions, num_matched_predictions/num_predictions)
    num_unmatched_predictions = num_predictions - num_matched_predictions #np.count_nonzero(total_pred_matched[(bin_edges[bin_idx]<total_pred_eT) & (total_pred_eT<bin_edges[bin_idx+1])]==0)
    # percentage_matched_in_pt.append(num_matched_predictions/num_predictions)
    percentage_unmatched_in_pt.append(num_unmatched_predictions/num_predictions)
    
    bin_mask_pb = (bin_edges[bin_idx]<total_pbox_ptadj2sig) & (total_pbox_ptadj2sig<bin_edges[bin_idx+1])
    num_pboxes = len(total_pred_matched[bin_mask_pb])
    num_matched_pboxes = sum(total_pred_matched[bin_mask_pb])
    print('num_pboxes',num_pboxes,'num_matched_pboxes',num_matched_pboxes, num_matched_pboxes/num_pboxes)
    num_unm_pboxes = num_pboxes - num_matched_pboxes 
    perc_unmatched_boxes.append(num_unm_pboxes/num_pboxes)
    
    n_preds.append(num_predictions)
    n_unmatched_preds.append(num_unmatched_predictions)
    n_pboxes.append(num_pboxes)
    n_unmatched_pboxes.append(num_unm_pboxes)
    print()

print("Number of >20GeV predictions SUMPOOL: ", len(total_pred_pt[total_pred_pt>20]), 'fraction matched: ', sum(total_pred_matched[total_pred_pt>20])/len(total_pred_matched[total_pred_pt>20]))
print("Number of <20GeV predictions SUMPOOL: ", len(total_pred_pt[total_pred_pt<20]), 'fraction matched: ', sum(total_pred_matched[total_pred_pt<20])/len(total_pred_matched[total_pred_pt<20]))
print("Number of >20GeV predictions WC: ", len(total_pbox_ptadj2sig[total_pbox_ptadj2sig>20]), 'fraction matched: ', sum(total_pred_matched[total_pbox_ptadj2sig>20])/len(total_pred_matched[total_pbox_ptadj2sig>20]))
print("Number of <20GeV predictions WC: ", len(total_pbox_ptadj2sig[total_pbox_ptadj2sig<20]), 'fraction matched: ', sum(total_pred_matched[total_pbox_ptadj2sig<20])/len(total_pred_matched[total_pbox_ptadj2sig<20]))

match_pred_errro = get_errorbars(np.array(n_matched_preds),np.array(n_truth))
unmatch_pred_errro = get_errorbars(np.array(n_unmatched_preds),np.array(n_preds))
unmatch_pbox_errro = get_errorbars(np.array(n_unmatched_pboxes),np.array(n_pboxes))


f,ax = plt.subplots(1,1,figsize=(8, 6))
# ax.plot(bin_centers1,percentage_matched_in_pt,marker='x',color='cyan',label=f'% Matched Target Boxes (Accuracy)')
# ax.plot(bin_centers,percentage_unmatched_in_pt,marker='+',color='coral',label=f'% Unmatched (Fake rate) Sumpool')
# ax.plot(bin_centers,perc_unmatched_boxes,marker='d',color='coral',label=f'% Unmatched (Fake rate) Weighted Circle')
ax.errorbar(bin_centers1,percentage_matched_in_pt,xerr=bin_width1/2,yerr=match_pred_errro,color='cyan',ls='-',marker='o',ms=5.5,elinewidth=1.2,label=f'% Matched Target Boxes (Accuracy)')
ax.errorbar(bin_centers,percentage_unmatched_in_pt,xerr=bin_width/2,yerr=unmatch_pred_errro,color='coral',ls='-',marker='x',ms=5.5,elinewidth=1.2,label=f'% Unmatched (Fake rate) Sumpool')
ax.errorbar(bin_centers,perc_unmatched_boxes,xerr=bin_width/2,yerr=unmatch_pbox_errro,color='coral',ls='-',marker='d',ms=5.5,elinewidth=1.2,label=f'% Unmatched (Fake rate) Weighted Circle')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum (GeV)',ylabel=f'Fraction of boxes')
ax.set_ylim((-0.2,1.2))
ax.legend(loc='lower left',bbox_to_anchor=(0.005, -0.03),fontsize="x-small")
print('Bin centres',bin_centers)
print(percentage_matched_in_pt)
print(percentage_unmatched_in_pt)
print(perc_unmatched_boxes)

hep.atlas.label(ax=ax,label='Internal',data=False,lumi=None,loc=1)
f.savefig(eff_save_loc + f'/match_frac_boxes_pT.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()











print("Fake rate vs accuracy")
eff_save_loc = save_folder + f"/trig/"
if not os.path.exists(eff_save_loc):
    os.makedirs(eff_save_loc)


bin_edges = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, max(total_pred_pt)]
bin_edges1 = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, max(total_pred_pt)]
bin_centers1 = bin_edges1[:-1] + 0.5 * np.diff(bin_edges1)
bin_width1 = np.diff(bin_edges)
n_matched_preds = []
n_truth = []
percentage_matched_in_pt = []
for bin_idx in range(len(bin_edges1)-1):
    bin_mask_tru = (bin_edges1[bin_idx]<total_tru_pt) & (total_tru_pt<bin_edges1[bin_idx+1])
    num_truth = len(total_tru_matched[bin_mask_tru])
    num_matched_truth = sum(total_tru_matched[bin_mask_tru])
    
    print('num_truth',num_truth,'num_matched_truth',num_matched_truth, num_matched_truth/num_truth)
    n_matched_preds.append(num_matched_truth)
    n_truth.append(num_truth)
    percentage_matched_in_pt.append(num_matched_truth/num_truth)


# bin_edges = [-100, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, max(total_pred_pt)]
# bin_edges = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, max(total_pred_pt)]
# bin_edges = [20, 30, 40, 50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200, 225, 250]
bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
bin_width = np.diff(bin_edges)
percentage_unmatched_in_pt = []
n_unmatched_preds = []
n_unmatched_pboxes = []
perc_unmatched_boxes = []
n_preds,n_pboxes = [],[]
for bin_idx in range(len(bin_edges)-1):
    # print(bin_edges[bin_idx],bin_edges[bin_idx+1])
    bin_mask = (bin_edges[bin_idx]<total_pred_pt) & (total_pred_pt<bin_edges[bin_idx+1])
    num_predictions = len(total_pred_matched[bin_mask])
    num_matched_predictions = sum(total_pred_matched[bin_mask])
    print('num_predictions',num_predictions,'num_matched_predictions',num_matched_predictions, num_matched_predictions/num_predictions)
    num_unmatched_predictions = num_predictions - num_matched_predictions #np.count_nonzero(total_pred_matched[(bin_edges[bin_idx]<total_pred_eT) & (total_pred_eT<bin_edges[bin_idx+1])]==0)
    # percentage_matched_in_pt.append(num_matched_predictions/num_predictions)
    percentage_unmatched_in_pt.append(num_unmatched_predictions/num_predictions)
    
    bin_mask_pb = (bin_edges[bin_idx]<total_pbox_ptadj2sig) & (total_pbox_ptadj2sig<bin_edges[bin_idx+1])
    num_pboxes = len(total_pred_matched[bin_mask_pb])
    num_matched_pboxes = sum(total_pred_matched[bin_mask_pb])
    print('num_pboxes',num_pboxes,'num_matched_pboxes',num_matched_pboxes, num_matched_pboxes/num_pboxes)
    num_unm_pboxes = num_pboxes - num_matched_pboxes 
    perc_unmatched_boxes.append(num_unm_pboxes/num_pboxes)
    
    n_preds.append(num_predictions)
    n_unmatched_preds.append(num_unmatched_predictions)
    n_pboxes.append(num_pboxes)
    n_unmatched_pboxes.append(num_unm_pboxes)
    print()

print("Number of >20GeV predictions SUMPOOL: ", len(total_pred_pt[total_pred_pt>20]), 'fraction matched: ', sum(total_pred_matched[total_pred_pt>20])/len(total_pred_matched[total_pred_pt>20]))
print("Number of <20GeV predictions SUMPOOL: ", len(total_pred_pt[total_pred_pt<20]), 'fraction matched: ', sum(total_pred_matched[total_pred_pt<20])/len(total_pred_matched[total_pred_pt<20]))
print("Number of >20GeV predictions WC: ", len(total_pbox_ptadj2sig[total_pbox_ptadj2sig>20]), 'fraction matched: ', sum(total_pred_matched[total_pbox_ptadj2sig>20])/len(total_pred_matched[total_pbox_ptadj2sig>20]))
print("Number of <20GeV predictions WC: ", len(total_pbox_ptadj2sig[total_pbox_ptadj2sig<20]), 'fraction matched: ', sum(total_pred_matched[total_pbox_ptadj2sig<20])/len(total_pred_matched[total_pbox_ptadj2sig<20]))

match_pred_errro = get_errorbars(np.array(n_matched_preds),np.array(n_truth))
unmatch_pred_errro = get_errorbars(np.array(n_unmatched_preds),np.array(n_preds))
unmatch_pbox_errro = get_errorbars(np.array(n_unmatched_pboxes),np.array(n_pboxes))


f,ax = plt.subplots(1,1,figsize=(8, 6))
# ax.plot(bin_centers1,percentage_matched_in_pt,marker='x',color='cyan',label=f'% Matched Target Boxes (Accuracy)')
# ax.plot(bin_centers,percentage_unmatched_in_pt,marker='+',color='coral',label=f'% Unmatched (Fake rate) Sumpool')
# ax.plot(bin_centers,perc_unmatched_boxes,marker='d',color='coral',label=f'% Unmatched (Fake rate) Weighted Circle')
ax.errorbar(bin_centers1,percentage_matched_in_pt,xerr=bin_width1/2,yerr=match_pred_errro,color='cyan',ls='-',marker='o',ms=5.5,elinewidth=1.2,label=f'Matched (accuracy)')
ax.errorbar(bin_centers,percentage_unmatched_in_pt,xerr=bin_width/2,yerr=unmatch_pred_errro,color='coral',ls='-',marker='x',ms=5.5,elinewidth=1.2,label=f'Unmatched (fake rate) Sumpool')
ax.errorbar(bin_centers,perc_unmatched_boxes,xerr=bin_width/2,yerr=unmatch_pbox_errro,color='fuchsia',ls='-',marker='d',ms=5.5,elinewidth=1.2,label=f'Unmatched (fake rate) Weighted Circle')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='$p_T$ [GeV]',ylabel=f'Fraction of boxes')
ax.set_ylim((0.0,1.2))
ax.legend(loc='lower left',bbox_to_anchor=(0.37, 0.4),fontsize="small")
# hep.atlas.label(ax=ax,label='Internal',data=False,lumi=None,loc=1)
f.savefig(eff_save_loc + f'/match_frac_boxes_pT.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()
print('Error bars for Tim')
print(match_pred_errro)
print(unmatch_pred_errro)
print(unmatch_pbox_errro)

print("\n\n\nIm interrested in the numbers below")
print('Bin centres',bin_centers)
print(bin_edges)
print([f"{num:.4f}" for num in percentage_matched_in_pt])
print([f"{num:.4f}" for num in percentage_unmatched_in_pt])
print([f"{num:.4f}" for num in perc_unmatched_boxes])
# print(percentage_matched_in_pt)
# print(percentage_unmatched_in_pt)
# print(perc_unmatched_boxes)

value = 50
print(f"Number of >{value}GeV targ: ", len(total_tru_pt[total_tru_pt>value]), 'fraction matched: ', sum(total_tru_matched[total_tru_pt>value])/len(total_tru_matched[total_tru_pt>value]))
print(f"Number of >{value}GeV predictions SUMPOOL: ", len(total_pred_pt[total_pred_pt>value]), 'fraction matched: ', sum(total_pred_matched[total_pred_pt>value])/len(total_pred_matched[total_pred_pt>value]))
print(f"Number of >{value}GeV predictions WC: ", len(total_pbox_ptadj2sig[total_pbox_ptadj2sig>value]), 'fraction matched: ', sum(total_pred_matched[total_pbox_ptadj2sig>value])/len(total_pred_matched[total_pbox_ptadj2sig>value]))

