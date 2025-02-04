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







model_name = "jetSSD_smallconvnext_central_32e"
metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/JZ4/20250124-13/box_metrics"
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/JZ4/20250124-13/"
# metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/ttbar/20250124-12/box_metrics"
# save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/ttbar/20250124-12/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)





#by event
total_n_truth = load_object(metrics_folder+"/n_truth.pkl")
total_n_preds = load_object(metrics_folder+"/n_preds.pkl")
total_delta_n = load_object(metrics_folder+"/delta_n.pkl")

total_n_tru_match = load_object(metrics_folder+"/n_matched_truth.pkl")
total_n_tru_unmatch = load_object(metrics_folder+"/n_unmatched_truth.pkl")
total_n_pre_match = load_object(metrics_folder+"/n_matched_preds.pkl")
total_n_pre_unmatch = load_object(metrics_folder+"/n_unmatched_preds.pkl")

#by jet
total_tru_pt = np.concatenate(load_object(metrics_folder+"/tboxes_pt.pkl"))
total_tru_eta = np.concatenate(load_object(metrics_folder+"/tboxes_eta.pkl"))
total_tru_phi = np.concatenate(load_object(metrics_folder+"/tboxes_phi.pkl"))
total_tru_matched = np.concatenate(load_object(metrics_folder+"/tboxes_matched.pkl"))

total_pred_matched = np.concatenate(load_object(metrics_folder+"/pboxes_matched.pkl"))
total_pred_scores = np.concatenate(load_object(metrics_folder+"/pboxes_scores.pkl"))
total_pred_pt = np.concatenate(load_object(metrics_folder+"/pboxes_pt.pkl"))
total_pred_eta = np.concatenate(load_object(metrics_folder+"/pboxes_eta.pkl"))
total_pred_phi = np.concatenate(load_object(metrics_folder+"/pboxes_phi.pkl"))

total_matched_tru_pt = np.concatenate(load_object(metrics_folder+"/tboxes_matched_pt.pkl"))
total_matched_pred_pt = np.concatenate(load_object(metrics_folder+"/pboxes_matched_pt.pkl"))

print(f"There are {len(total_tru_pt)} target jets ({sum(total_n_truth)}), of which {sum(total_tru_matched)} are matched")
print(f"There are {len(total_pred_eta)} predicted boxes ({sum(total_n_preds)}), of which {sum(total_pred_matched)} are matched, {len(total_matched_tru_pt)} or {len(total_matched_pred_pt)}")


pt_thresholds = [0,10,15,20,25,30,35,40,60,100,200,250,450,625]
tru_matched_percentage = []
for threshold in pt_thresholds:
    mask = total_tru_pt > threshold
    total_tru_matched_mask = total_tru_matched[mask]
    tru_matched_percentage.append(np.round(sum(total_tru_matched_mask) / len(total_tru_matched_mask),4))
    print(f'For jets above {threshold}: {sum(total_tru_matched_mask)/len(total_tru_matched_mask)} of truth jets are matched')
    print(f'There are {len(total_tru_pt[mask])} jets above this threshold\n')

print()
pred_matched_percentage = []
for threshold in pt_thresholds:
    mask = total_pred_pt > threshold
    total_pred_matched_mask = total_pred_matched[mask]
    pred_matched_percentage.append(np.round(sum(total_pred_matched_mask) / len(total_pred_matched_mask),4))
    print(f'For jets above {threshold}: {sum(total_pred_matched_mask) / len(total_pred_matched_mask)} of predicted jets are matched')
    print(f'There are {len(total_pred_pt[mask])} jets above this threshold\n')

print("lists")
print(tru_matched_percentage)
print(pred_matched_percentage)







#################################################################################
# plotting
log = True
image_format = "png"


# 1.
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_tru, bins, _ = ax0.hist(total_n_truth,bins=max(total_n_truth)-1,histtype='step',color='green',lw=2,label='Target Jets {:.2f}$\pm${:.1f}'.format(np.mean(total_n_truth),np.std(total_n_truth)))
freq_pred, _, _   = ax0.hist(total_n_preds,bins=bins,histtype='step',color='red',lw=2,label='Pred Jets {:.2f}$\pm${:.1f}'.format(np.mean(total_n_preds),np.std(total_n_preds)))
ax0.grid()
ax0.set_title('Number of jets per event', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.6, 0.75),fontsize="small")

hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Num. jets')
f.savefig(save_folder + f'/n_jets_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

plt.close()



# 2.
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_tru, bins, _ = ax0.hist(total_n_tru_match,bins=int(max(total_n_tru_match))-1,histtype='step',color='green',lw=2,label='Target Jets {:.2f}$\pm${:.1f}'.format(np.mean(total_n_tru_match),np.std(total_n_tru_match)))
freq_pred, _, _   = ax0.hist(total_n_pre_match,bins=bins,histtype='step',color='red',lw=2,label='Pred Jets {:.2f}$\pm${:.1f}'.format(np.mean(total_n_pre_match),np.std(total_n_pre_match)),linestyle='dashed')

ax0.grid()
ax0.set_title('Number of Matched boxes per event', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.6, 0.75),fontsize="small")

hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Num. matched jets')
f.savefig(save_folder + f'/n_match_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

plt.close()


# 3.
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_n_pre_unmatch,bins=int(max(total_n_pre_unmatch))+1,histtype='step',color='red',lw=2.0,label='Pred Jets {:.2f}$\pm${:.1f}'.format(np.mean(total_n_pre_unmatch),np.std(total_n_pre_unmatch)))
freq_tru, bins, _ = ax0.hist(total_n_tru_unmatch,bins=bins,histtype='step',color='green',lw=2.0,label='Target Jets {:.2f}$\pm${:.1f}'.format(np.mean(total_n_tru_unmatch),np.std(total_n_tru_unmatch)))

ax0.grid()
ax0.set_title('Number of Unmatched boxes per event', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")

hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Num. unmatched jets')
f.savefig(save_folder + f'/n_unmatch_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")

plt.close()





# 2d plots of all/matched/unmatched
bins_x = np.linspace(MIN_CELLS_ETA, MAX_CELLS_ETA, (int((MAX_CELLS_ETA - MIN_CELLS_ETA) / (0.1)) + 1))
bins_y = np.linspace(MIN_CELLS_PHI, MAX_CELLS_PHI, (int((MAX_CELLS_PHI - MIN_CELLS_PHI) / ((2*np.pi)/64)) + 1))
extent = (MIN_CELLS_ETA,MAX_CELLS_ETA,MIN_CELLS_PHI,MAX_CELLS_PHI)


# 6.
H_tot_tru,_,_ = np.histogram2d(total_tru_eta,total_tru_phi,bins=(bins_x,bins_y),weights=np.ones_like(total_tru_eta)/10_000)
H_tot_tru = H_tot_tru.T
f,ax = plt.subplots()
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","green","lime"])
ii = ax.imshow(H_tot_tru,extent=extent,cmap=cmap1)
cbar = f.colorbar(ii,ax=ax)
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Num. target jets per event', rotation=90)
cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
ax.set(xlabel=f"eta",ylabel=f"phi")
f.savefig(save_folder + f'/2d_tru_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


# 7.
H_match_tot_tru,_,_ = np.histogram2d(total_tru_eta[np.where(total_tru_matched==1)],total_tru_phi[np.where(total_tru_matched==1)],bins=(bins_x,bins_y),weights=np.ones_like(total_tru_eta[np.where(total_tru_matched==1)])/10_000)
H_match_tot_tru = H_match_tot_tru.T

f,ax = plt.subplots()
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","green","lime"])
ii = ax.imshow(H_match_tot_tru,extent=extent,cmap=cmap1)
cbar = f.colorbar(ii,ax=ax)
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Num. matched target jets per event', rotation=90)
cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
ax.set(xlabel=f"eta",ylabel=f"phi")
f.savefig(save_folder + f'/2d_match_tru_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


# 8.
H_unmatch_tot_tru,_,_ = np.histogram2d(total_tru_eta[np.where(total_tru_matched==0)],total_tru_phi[np.where(total_tru_matched==0)],bins=(bins_x,bins_y),weights=np.ones_like(total_tru_eta[np.where(total_tru_matched==0)])/10_000)
H_unmatch_tot_tru = H_unmatch_tot_tru.T

f,ax = plt.subplots()
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","green","lime"])
ii = ax.imshow(H_unmatch_tot_tru,extent=extent,cmap=cmap1)
cbar = f.colorbar(ii,ax=ax)
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Num. unmatched target jets per event', rotation=90)
cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
ax.set(xlabel=f"eta",ylabel=f"phi")
f.savefig(save_folder + f'/2d_unmatch_tru_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


# 9.
H_tot_pre,_,_ = np.histogram2d(total_pred_eta,total_pred_phi,bins=(bins_x,bins_y),weights=np.ones_like(total_pred_eta)/10_000)
H_tot_pre = H_tot_pre.T

f,ax = plt.subplots()
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","blue","violet","red"])
ii = ax.imshow(H_tot_pre,extent=extent,cmap=cmap)
cbar = f.colorbar(ii,ax=ax)
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Num. jet predicted per event', rotation=90)
cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
ax.set(xlabel=f"eta",ylabel=f"phi")
f.savefig(save_folder + f'/2d_pre_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


# 10.
H_match_tot_pre,_,_ = np.histogram2d(total_pred_eta[np.where(total_pred_matched==1)],total_pred_phi[np.where(total_pred_matched==1)],bins=(bins_x,bins_y),weights=np.ones_like(total_pred_eta[np.where(total_pred_matched==1)])/10_000)
H_match_tot_pre = H_match_tot_pre.T

f,ax = plt.subplots()
ii = ax.imshow(H_match_tot_pre,extent=extent,cmap=cmap)
cbar = f.colorbar(ii,ax=ax)
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Num. matched jet predicted per event', rotation=90)
cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
ax.set(xlabel=f"eta",ylabel=f"phi")
f.savefig(save_folder + f'/2d_match_pre_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


# 11.
H_unmatch_tot_pre,_,_ = np.histogram2d(total_pred_eta[np.where(total_pred_matched==0)],total_pred_phi[np.where(total_pred_matched==0)],bins=(bins_x,bins_y),weights=np.ones_like(total_pred_eta[np.where(total_pred_matched==0)])/10_000)
H_unmatch_tot_pre = H_unmatch_tot_pre.T

f,ax = plt.subplots()
ii = ax.imshow(H_unmatch_tot_pre,extent=extent,cmap=cmap)
cbar = f.colorbar(ii,ax=ax)
cbar.ax.get_yaxis().labelpad = 10
cbar.set_label('Num. unmatched jet predicted per event', rotation=90)
cbar.ax.set_yticklabels(['{:.4f}'.format(x) for x in cbar.ax.get_yticks()])
ax.set_xticks([-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0])
ax.set_yticks([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0])
ax.tick_params(axis='both',which='major',direction='out',length=5.5,labelsize=14)
ax.tick_params(axis='both',which='minor',direction='out',length=2.5)
ax.set(xlabel=f"eta",ylabel=f"phi")
f.savefig(save_folder + f'/2d_unmatch_pre_jets.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()























print()
match_pred_pt = total_pred_pt[np.where(total_pred_matched==1)]
match_pred_eta = total_pred_eta[np.where(total_pred_matched==1)]
match_pred_phi = total_pred_phi[np.where(total_pred_matched==1)]
match_pred_scr = total_pred_scores[np.where(total_pred_matched==1)]

match_tru_pt = total_tru_pt[np.where(total_tru_matched==1)]
match_tru_eta = total_tru_eta[np.where(total_tru_matched==1)]
match_tru_phi = total_tru_phi[np.where(total_tru_matched==1)]

unmatch_pred_pt = total_pred_pt[np.where(total_pred_matched==0)]
unmatch_pred_eta = total_pred_eta[np.where(total_pred_matched==0)]
unmatch_pred_phi = total_pred_phi[np.where(total_pred_matched==0)]
unmatch_pred_scr = total_pred_scores[np.where(total_pred_matched==0)]

unmatch_tru_pt = total_tru_pt[np.where(total_tru_matched==0)]
unmatch_tru_eta = total_tru_eta[np.where(total_tru_matched==0)]
unmatch_tru_phi = total_tru_phi[np.where(total_tru_matched==0)]


################################################################
# pt total
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_pred_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets')
freq_tru, bins, _ = ax0.hist(total_tru_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.grid()
ax0.set_title('Transverse Momentum', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Jet/Box pT (GeV)')
f.savefig(save_folder + f'/jet_pt_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

# pt match
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(match_pred_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets')
freq_tru, bins, _ = ax0.hist(match_tru_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.grid()
ax0.set_title('Transverse Momentum Matched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Jet/Box pT (GeV)')
f.savefig(save_folder + f'/jet_pt_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

# pt match 2
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_matched_pred_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets')
freq_tru, bins, _ = ax0.hist(total_matched_tru_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.grid()
ax0.set_title('Transverse Momentum Matched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Jet/Box pT (GeV)')
f.savefig(save_folder + f'/jet_pt_match2.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

# pt unmatch
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(unmatch_pred_pt,bins=100,histtype='step',color='red',lw=1.5,label='Pred Jets')
freq_tru, bins, _ = ax0.hist(unmatch_tru_pt,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.grid()
ax0.set_title('Transverse Momentum Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Jet/Box pT (GeV)')
f.savefig(save_folder + f'/jet_pt_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


################################################################
# eta total
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_pred_eta,bins=25,histtype='step',color='red',lw=1.5,label='Pred Jets')
freq_tru, bins, _ = ax0.hist(total_tru_eta,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.grid()
ax0.set_title('Pseudorapidity Total', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Jet/Box eta')
f.savefig(save_folder + f'/jet_eta_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

# eta match
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(match_pred_eta,bins=25,histtype='step',color='red',lw=1.5,label='Pred Jets')
freq_tru, bins, _ = ax0.hist(match_tru_eta,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.grid()
ax0.set_title('Pseudorapidity Matched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Jet/Box eta')
f.savefig(save_folder + f'/jet_eta_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

# eta match
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(unmatch_pred_eta,bins=25,histtype='step',color='red',lw=1.5,label='Pred Jets')
freq_tru, bins, _ = ax0.hist(unmatch_tru_eta,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.grid()
ax0.set_title('Pseudorapidity Unmatched', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Jet/Box eta')
f.savefig(save_folder + f'/jet_eta_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()


################################################################
# phi total
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(total_pred_phi,bins=25,histtype='step',color='red',lw=1.5,label='Pred Jets')
freq_tru, bins, _ = ax0.hist(total_tru_phi,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.grid()
ax0.set_title('Azimuth Total', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Jet/Box phi')
f.savefig(save_folder + f'/jet_phi_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

# phi match
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(match_pred_phi,bins=25,histtype='step',color='red',lw=1.5,label='Pred Jets')
freq_tru, bins, _ = ax0.hist(match_tru_phi,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.grid()
ax0.set_title('Azimuth Match', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Jet/Box phi')
f.savefig(save_folder + f'/jet_phi_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()

# phi unmatch
f,ax0 = plt.subplots(1,1,figsize=(9, 6))
freq_pred, bins, _   = ax0.hist(unmatch_pred_phi,bins=25,histtype='step',color='red',lw=1.5,label='Pred Jets')
freq_tru, bins, _ = ax0.hist(unmatch_tru_phi,bins=bins,histtype='step',color='green',lw=1.5,label='Target Jets')
ax0.grid()
ax0.set_title('Azimuth Unmatch', fontsize=16, fontfamily="TeX Gyre Heros")
ax0.legend(loc='lower left',bbox_to_anchor=(0.45, 0.7),fontsize="medium")
hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
ax0.set(yscale='log',ylabel='Freq.',xlabel='Jet/Box phi')
f.savefig(save_folder + f'/jet_phi_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()










################################################################
# 2d pt eta
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
hist = ax.hist2d(total_pred_pt, total_pred_eta, bins=[100, 50], cmap='Blues',norm=matplotlib.colors.LogNorm())
fig.colorbar(hist[3], ax=ax, label='Counts')
ax.set(xlabel="pT",ylabel="eta",title="total pred boxes",ylim=(-2.5,2.5))
plt.savefig(save_folder + f'/2d_pt_eta_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()
#
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
hist = ax.hist2d(match_tru_pt, match_tru_eta, bins=[100, 50], cmap='Blues',norm=matplotlib.colors.LogNorm())
fig.colorbar(hist[3], ax=ax, label='Counts')
ax.set(xlabel="pT",ylabel="eta",title="Match pred boxes",ylim=(-2.5,2.5))
plt.savefig(save_folder + f'/2d_pt_eta_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()
#
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
hist = ax.hist2d(unmatch_tru_pt, unmatch_tru_eta, bins=[100, 50], cmap='Blues',norm=matplotlib.colors.LogNorm())
fig.colorbar(hist[3], ax=ax, label='Counts')
ax.set(xlabel="pT",ylabel="eta",title="Unmatch pred boxes",ylim=(-2.5,2.5))
plt.savefig(save_folder + f'/2d_pt_eta_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()
#
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
hist = ax.hist2d(total_tru_pt, total_tru_eta, bins=[100, 50], cmap='Blues',norm=matplotlib.colors.LogNorm())
fig.colorbar(hist[3], ax=ax, label='Counts')
ax.set(xlabel="pT",ylabel="eta",title="total true boxes",ylim=(-2.5,2.5))
plt.savefig(save_folder + f'/2d_pt_eta_tru_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()





# 2d pt score
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
hist = ax.hist2d(total_pred_pt, total_pred_scores, bins=[100, 50], cmap='Blues',norm=matplotlib.colors.LogNorm())
fig.colorbar(hist[3], ax=ax, label='Counts')
ax.set(xlabel="pT",ylabel="score",title="total pred boxes")
plt.savefig(save_folder + f'/2d_pt_scr_total.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()
#
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
hist = ax.hist2d(match_pred_pt, match_pred_scr, bins=[100, 50], cmap='Blues',norm=matplotlib.colors.LogNorm())
fig.colorbar(hist[3], ax=ax, label='Counts')
ax.set(xlabel="pT",ylabel="score",title="Match pred boxes")
plt.savefig(save_folder + f'/2d_pt_scr_match.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()
#
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
hist = ax.hist2d(unmatch_pred_pt, unmatch_pred_scr, bins=[100, 50], cmap='Blues',norm=matplotlib.colors.LogNorm())
fig.colorbar(hist[3], ax=ax, label='Counts')
ax.set(xlabel="pT",ylabel="score",title="Unmatch pred boxes")
plt.savefig(save_folder + f'/2d_pt_scr_unmatch.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()









    


eff_save_loc = save_folder + f"/jet_res/"
if not os.path.exists(eff_save_loc):
    os.makedirs(eff_save_loc)

def _1gaussian(x, amp1,cen1,sigma1):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))

def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))

def gaussian(x, a, mean, variance):
    return a * np.exp(-((x - mean)**2 / (2 * variance)))





bin_edges = [30, 40, 55, 80, 110, 150, 200, 300, 400, 550, 800] #[20,35,50,75,100,125,175,225,300,400,500,600]
bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
bin_width = np.diff(bin_edges)

average_response, std_response = list(), list()
fit_mu, fit_mu_unc = list(), list()
fit_sigma, fit_sigma_unc = list(), list()
for bin_idx in range(len(bin_edges)-1):
    bin_mask = (bin_edges[bin_idx]<total_matched_tru_pt) & (total_matched_tru_pt<bin_edges[bin_idx+1])
    target_jet_pt_in_this_bin = total_matched_tru_pt[bin_mask]
    pred_jet_pt_in_this_bin = total_matched_pred_pt[bin_mask]
    jet_pt_response_bin_i = pred_jet_pt_in_this_bin / target_jet_pt_in_this_bin
    print("Number of jets in bin ", len(jet_pt_response_bin_i))
    print("Average in bin ", bin_idx, bin_edges[bin_idx],bin_edges[bin_idx+1], np.mean(jet_pt_response_bin_i))
    
    jet_resp_hist, bins = np.histogram(jet_pt_response_bin_i, bins=50)
    bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    # popt_g, pcov_g = scipy.optimize.curve_fit(gausmyfunc, xdata=bin_centres, ydata=jet_resp_hist, bounds=[(0.5,0.0, -np.inf),(np.inf, np.inf, np.inf)])
    popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[1.0,1.0,0.1])
    print("fit parameters ", popt_g)
    print("fit error ",np.sqrt(np.diag(pcov_g)))

    plt.figure()
    plt.stairs(jet_resp_hist, bins, fill=True, color='orange',alpha=0.5)
    plt.hist(jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='blue')
    x = np.linspace(0,jet_pt_response_bin_i.max(),100)
    plt.plot(x, gaussian(x, *popt_g), linewidth=2.5, label='Custom gausmyfunc')
    plt.xlabel('reco/target jet pt')
    plt.ylabel(f'jets in bin {bin_idx}')
    # plt.yscale('log')
    plt.title(f'Jet pT in [{bin_edges[bin_idx]},{bin_edges[bin_idx+1]}]')
    plt.savefig(f'pt_response_bin_{bin_idx}.png')
    plt.close()

    average_response.append(np.mean(jet_pt_response_bin_i))
    std_response.append(np.std(jet_pt_response_bin_i))
    fit_mu.append(popt_g[1])
    fit_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
    # we are dealing with variance now!
    # fit_sigma.append(popt_g[2])
    V = popt_g[2]
    sigma_V = np.sqrt(np.diag(pcov_g))[2]
    fit_sigma.append(np.sqrt(V))
    fit_sigma_unc.append((sigma_V*np.sqrt(V)) / (2*V))
    # fit_sigma_unc.append(np.sqrt(np.diag(pcov_g))[2])
    print()
    # yhist, xhist = np.histogram(jet_pt_response_bin_i, bins=100)
    # xh = np.where(yhist > 0)[0]
    # yh = yhist[xh]
    # popt, pcov = scipy.optimize.curve_fit(gaussian, xh, yh, [len(jet_pt_response_bin_i), 1, 1])
    # perr = np.sqrt(np.diag(pcov))

    # print(f"amplitude = {popt[0]:0.2f} (+/-) {perr[0]:0.2f}")
    # print(f"center = {popt[1]:0.2f} (+/-) {perr[2]:0.2f}")
    # print(f"sigma = {popt[2]:0.2f} (+/-) {perr[2]:0.2f}")

    # plt.figure()
    # plt.plot(yhist)
    # plt.xlabel('reco/target jet pt')
    # # xs = np.linspace(0,3,100)
    # xs = np.linspace(min(jet_pt_response_bin_i),max(jet_pt_response_bin_i),100)
    # plt.plot(xs, gaussian(xs, *popt))
    # plt.title(f'Jet pT in [{bin_edges[bin_idx]},{bin_edges[bin_idx+1]}]')
    # plt.savefig(f'pt_response_bin_{bin_idx}_gauss.png')
    # plt.close()



plt.figure()
plt.errorbar(bin_centers, average_response, xerr=bin_width/2, fmt='o', capsize=5, color='blue',label='Simple mean')
plt.errorbar(bin_centers, fit_mu, xerr=bin_width/2, yerr=fit_mu_unc, fmt='o', capsize=5, color='orange',label='Fit param')
plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
plt.xlabel('AntiKt4EMTopo Jet pT (jet constituent scale)')
plt.ylabel('Jet Energy Response')
plt.legend()
plt.savefig('jet_response_simple.png')

plt.figure()
plt.errorbar(bin_centers, std_response, xerr=bin_width/2, fmt='o', capsize=5, color='blue',label='Simple np.std')
plt.errorbar(bin_centers, fit_sigma, xerr=bin_width/2, yerr=fit_sigma_unc, fmt='o', capsize=5, color='orange',label='Fit param')
plt.errorbar(bin_centers, abs(np.array(fit_sigma)), xerr=bin_width/2, yerr=fit_sigma_unc,alpha=0.5, capsize=5, color='pink')
plt.xlabel('AntiKt4EMTopo Jet pT (jet constituent scale)')
plt.ylabel('Jet Energy Resolution')
plt.legend()
plt.savefig('jet_resolution_simple.png')

plt.figure()
plt.errorbar(bin_centers, np.array(fit_sigma)/np.array(bin_centers), xerr=bin_width/2, yerr=fit_sigma_unc, fmt='o', capsize=5, color='orange',label='Fit param')
ax = plt.gca()
ax.ticklabel_format(style='plain')
plt.xlabel('AntiKt4EMTopo Jet pT (jet constituent scale)')
plt.ylabel('Relative Jet Energy Resolution, sigma(pT) / pT')
plt.legend()
plt.savefig('jet_relative_resolution_simple.png')


#binning in eta

# # eta_bins = [0.0,0.2,0.7,1.0,1.3,1.8,2.5,2.8,3.2,3.5,4.5]
# eta_bins = [-2.5,-1.8,-1.3,-1.0,-0.7,-0.2,0.0,0.2,0.7,1.0,1.3,1.8,2.5]
# eta_bin_centers = eta_bins[:-1] + 0.5 * np.diff(eta_bins)
# eta_bin_width = np.diff(eta_bins)
# average_response = list()
# fit_mu, fit_mu_unc = list(), list()
# fit_sigma, fit_sigma_unc = list(), list()
# for bin_idx in range(len(eta_bins)-1):
#     bin_mask = (bin_edges[bin_idx]<total_matched_tru_eta) & (total_matched_tru_eta<bin_edges[bin_idx+1])
#     target_jet_pt_in_this_bin = total_matched_tru_pt[bin_mask]
#     pred_jet_pt_in_this_bin = total_matched_pred_pt[bin_mask]
#     jet_pt_response_bin_i = pred_jet_pt_in_this_bin / target_jet_pt_in_this_bin
#     print("Number of jets in bin ", len(jet_pt_response_bin_i))
#     print("Average in bin ", bin_idx, eta_bins[bin_idx],eta_bins[bin_idx+1], np.mean(jet_pt_response_bin_i))
    
#     jet_resp_hist, bins = np.histogram(jet_pt_response_bin_i, bins=50)
#     bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
#     # popt_g, pcov_g = scipy.optimize.curve_fit(gausmyfunc, xdata=bin_centres, ydata=jet_resp_hist, bounds=[(0.5,0.0, -np.inf),(np.inf, np.inf, np.inf)])
#     popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[1,1,1])
#     print("fit parameters ", popt_g)
#     print("fit error ",np.sqrt(np.diag(pcov_g)))

#     plt.figure()
#     plt.stairs(jet_resp_hist, bins, fill=True, color='orange',alpha=0.5)
#     plt.hist(jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='blue')
#     x = np.linspace(0,jet_pt_response_bin_i.max(),100)
#     plt.plot(x, gaussian(x, *popt_g), linewidth=2.5, label='Custom gausmyfunc')
#     plt.xlabel('reco/target jet pt')
#     plt.ylabel(f'jets in bin {bin_idx}')
#     # plt.yscale('log')
#     plt.title(f'Jet eta in [{eta_bins[bin_idx]},{eta_bins[bin_idx+1]}]')
#     plt.savefig(f'pt_response_eta_bin_{bin_idx}.png')
#     plt.close()

#     average_response.append(np.mean(jet_pt_response_bin_i))
#     fit_mu.append(popt_g[1])
#     fit_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
#     fit_sigma.append(popt_g[2])
#     fit_sigma_unc.append(np.sqrt(np.diag(pcov_g))[2])


# plt.figure()
# plt.errorbar(eta_bin_centers, average_response, xerr=bin_width/2, fmt='o', capsize=5, color='blue',label='Simple mean')
# plt.errorbar(eta_bin_centers, fit_mu, xerr=bin_width/2, yerr=fit_mu_unc, fmt='o', capsize=5, color='orange',label='Fit param')
# plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
# plt.xlabel('AntiKt4EMTopo Jet pT (jet constituent scale)')
# plt.ylabel('Jet Energy Response')
# plt.legend()
# plt.savefig('jet_response_eta_simple.png')

# plt.figure()
# plt.errorbar(eta_bin_centers, fit_sigma, xerr=bin_width/2, yerr=fit_sigma_unc, fmt='o', capsize=5, color='orange')
# plt.xlabel('AntiKt4EMTopo Jet pT (jet constituent scale)')
# plt.ylabel('Jet Energy Resolution')
# plt.savefig('jet_resolution_eta_simple.png')








quit()














########################################################################
# Trigger efficiencies    ########################################################################
########################################################################

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




















eff_save_loc = save_folder + f"/trig/"
if not os.path.exists(eff_save_loc):
    os.makedirs(eff_save_loc)

# bin_edges = [0, 20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, max(total_pred_pt)]
bin_edges = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, max(total_pred_pt)]
# bin_edges = [20, 30, 40, 50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200, 225, 250]
bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
bin_width = np.diff(bin_edges)
percentage_matched_in_pt,percentage_unmatched_in_pt = [],[]
n_matched_preds,n_unmatched_preds = [], []
n_preds,n_truth = [],[]
for bin_idx in range(len(bin_edges)-1):
    # print(bin_edges[bin_idx],bin_edges[bin_idx+1])
    bin_mask = (bin_edges[bin_idx]<total_pred_pt) & (total_pred_pt<bin_edges[bin_idx+1])
    num_predictions = len(total_pred_matched[bin_mask])
    num_matched_predictions = sum(total_pred_matched[bin_mask])
    
    bin_mask_tru = (bin_edges[bin_idx]<total_tru_pt) & (total_tru_pt<bin_edges[bin_idx+1])
    print('num_predictions',num_predictions,'num_matched_predictions',num_matched_predictions, num_matched_predictions/num_predictions)
    num_truth = len(total_tru_matched[bin_mask_tru])
    num_matched_truth = sum(total_tru_matched[bin_mask_tru])
    num_unmatched_predictions = num_predictions - num_matched_predictions #np.count_nonzero(total_pred_matched[(bin_edges[bin_idx]<total_pred_eT) & (total_pred_eT<bin_edges[bin_idx+1])]==0)
    try:
        print('num_truth',num_truth,'num_matched_truth',num_matched_truth, num_matched_truth/num_truth)
    except ZeroDivisionError:
        pass
    percentage_matched_in_pt.append(num_matched_truth/num_truth)
    # percentage_matched_in_pt.append(num_matched_predictions/num_predictions)
    percentage_unmatched_in_pt.append(num_unmatched_predictions/num_predictions)
    n_matched_preds.append(num_matched_predictions)
    n_unmatched_preds.append(num_unmatched_predictions)
    n_preds.append(num_predictions)
    n_truth.append(num_truth)
print("--->")
print(n_truth)
print(n_preds)
match_pred_errro = get_errorbars(np.array(n_matched_preds),np.array(n_truth))
unmatch_pred_errro = get_errorbars(np.array(n_unmatched_preds),np.array(n_preds))

f,ax = plt.subplots(1,1,figsize=(8, 6))
ax.plot(bin_centers,percentage_matched_in_pt,marker='x',color='cyan',label=f'% Matched Truth Boxes (Accuracy)')
ax.plot(bin_centers,percentage_unmatched_in_pt,marker='+',color='coral',label=f'% Unmatched Prediction Boxes (Fake rate)')
# ax.errorbar(bin_centers,percentage_matched_in_pt,xerr=bin_width/2,yerr=match_pred_errro,color='black',marker='.',ms=5.5,elinewidth=1.2,ls='none',label=f'% Matched Truth Boxes (Accuracy)')
# ax.errorbar(bin_centers,percentage_unmatched_in_pt,xerr=bin_width/2,yerr=unmatch_pred_errro,color='black',marker='x',ms=5.5,elinewidth=1.2,ls='none',label=f'% Unmatched Prediction Boxes (Fake rate)')
ax.axhline(y=1.0,color='silver',ls='--',alpha=0.7)
ax.axhline(y=0.0,color='silver',ls='--',alpha=0.7)
ax.set(xlabel='Transverse Momentum (GeV)',ylabel=f'Fraction of boxes',title='Box Matching Test Set')
ax.set_ylim((-0.2,1.2))
ax.legend(loc='lower left',bbox_to_anchor=(0.005, 0.005),fontsize="x-small")
print('Bin centres',bin_centers)
print(percentage_matched_in_pt)
print(percentage_unmatched_in_pt)

hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=1)
f.savefig(eff_save_loc + f'/match_frac_boxes_pT.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
plt.close()





























event_tru_pt = load_object(metrics_folder+"/tboxes_pt.pkl")
event_pred_pt = load_object(metrics_folder+"/pboxes_pt.pkl")
antikt_lead_pt = np.array([leading_jet_pt(x) for x in event_tru_pt])
pred_lead_pt = np.array([leading_jet_pt(x) for x in event_pred_pt])


lead_jet_pt_cut = 450 # GeV
#make a trigger decision based on antikt jet
trig_decision_akt = np.argwhere(antikt_lead_pt>lead_jet_pt_cut).T[0]
trig_decision_pred = np.argwhere(pred_lead_pt>lead_jet_pt_cut).T[0]



trig_save_loc = save_folder + f"/trig/"
if not os.path.exists(trig_save_loc):
    os.makedirs(trig_save_loc)
    
start,end = 350,750
step = 10
bins = np.arange(start, end, step)

f,ax = plt.subplots(3,1,figsize=(8,14))
n_akt,bins,_ = ax[0].hist(antikt_lead_pt,bins=bins,histtype='step',label='Anti-kt jetConstitScale')
ax[0].set_title(f'Before {lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
ax[0].set(xlabel="Leading EMTopo (Offline) jet pT (GeV)",ylabel='Freq.')
ax[0].legend()

n2_akt,bins,_ = ax[1].hist(antikt_lead_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="gold")
n2_p,_,_ = ax[1].hist(antikt_lead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes',color="red")
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

bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]

ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='gold')
ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes.',color='red')
ax[2].grid()
ax[2].set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency')
ax[2].legend(loc='lower right')
# hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
f.savefig(trig_save_loc + f'/eff_plot_leading{lead_jet_pt_cut:.0f}GeV_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



f,a = plt.subplots(1,1,figsize=(8,8))
a.axvline(x=lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes',color='red')
a.grid()
a.set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency')
a.legend(loc='upper left')
hep.atlas.label(ax=a,label='Work in Progress',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_leading{lead_jet_pt_cut:.0f}GeV.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



################
# Nth Leading Jet
nth_jet = 4
nth_lead_jet_pt_cut = 75 # 400GeV
antikt_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in event_tru_pt])
pred_nlead_pt = np.array([nth_leading_jet_pt(x,nth_jet) for x in event_pred_pt])

trig_decision_akt = np.argwhere(antikt_nlead_pt>nth_lead_jet_pt_cut).T[0]
trig_decision_pred = np.argwhere(pred_nlead_pt>nth_lead_jet_pt_cut).T[0]

start,end = 20,350
step = 10
bins = np.arange(start, end, step)

f,ax = plt.subplots(3,1,figsize=(6.5,12))
n_akt,bins,_ = ax[0].hist(antikt_nlead_pt,bins=bins,histtype='step',label='Anti-kt jetConstitScale')
ax[0].set_title(f'Before {nth_lead_jet_pt_cut:.0f}GeV Cut')
ax[0].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Freq.')
ax[0].grid()
ax[0].legend()

n2_akt,bins,_ = ax[1].hist(antikt_nlead_pt[trig_decision_akt],bins=bins,histtype='step',label='Anti-kt jetConstitScale',color="gold")
n2_pb,_,_ = ax[1].hist(antikt_nlead_pt[trig_decision_pred],bins=bins,histtype='step',label='Pred Boxes Adj',color="red")
ax[1].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
ax[1].set_title(f'After {nth_lead_jet_pt_cut:.0f}GeV Cut',fontsize=16, fontfamily="TeX Gyre Heros")
ax[1].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Freq.')
ax[1].legend()

ax[2].axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
with np.errstate(divide='ignore', invalid='ignore'):
    step_eff = get_ratio(n2_akt,n_akt)
    step_err = get_errorbars(n2_akt,n_akt)
    pred_eff = get_ratio(n2_pb,n_akt)
    pred_err = get_errorbars(n2_pb,n_akt)

bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]

ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,elinewidth=0.4,marker='.',ls='none',label='Anti-kt',color='gold')
ax[2].errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes',color='red')
ax[2].grid()
ax[2].set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
ax[2].legend(loc='lower right')
hep.atlas.label(ax=ax[2],label='Work in Progress',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
f.savefig(trig_save_loc + f'/eff_plot_{nth_jet}leading{nth_lead_jet_pt_cut:.0f}GeV_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")



f,a = plt.subplots(1,1,figsize=(8,8))
a.axvline(x=nth_lead_jet_pt_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
a.errorbar(bin_centers,pred_eff,xerr=bin_width/2,yerr=pred_err,elinewidth=0.4,marker='.',ls='none',label='Pred Boxes',color='red')
a.grid()
a.set(xlabel=f"${{{nth_jet}}}^{{th}}$ Leading jet pT (GeV)",ylabel='Efficiency')
a.legend(loc='upper left')
hep.atlas.label(ax=a,label='Work in Progress',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_{nth_jet}_leading{lead_jet_pt_cut:.0f}GeV.{image_format}',dpi=400,format=image_format,bbox_inches="tight")















def get_eff(leading_jet_pt_cut):
    event_tru_pt = load_object(metrics_folder+"/tboxes_pt.pkl")
    event_pred_pt = load_object(metrics_folder+"/pboxes_pt.pkl")
    antikt_lead_pt = np.array([leading_jet_pt(x) for x in event_tru_pt])
    pred_lead_pt = np.array([leading_jet_pt(x) for x in event_pred_pt])

    #make a trigger decision based on antikt jet
    trig_decision_akt = np.argwhere(antikt_lead_pt>leading_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(pred_lead_pt>leading_jet_pt_cut).T[0]

    start,end = 350,750
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


f,a = plt.subplots(1,1,figsize=(8,8))
a.errorbar(xs,ys,xerr=(xs[1]-xs[0])/2,yerr=yerr,elinewidth=0.4,marker='.',ls='none',label='450 GeV Cut',color='red')
a.errorbar(xs2,ys2,xerr=(xs[1]-xs[0])/2,yerr=yerr2,elinewidth=0.4,marker='.',ls='none',label='550 GeV Cut',color='coral')
a.errorbar(xs3,ys3,xerr=(xs[1]-xs[0])/2,yerr=yerr3,elinewidth=0.4,marker='.',ls='none',label='650 GeV Cut',color='peru')
a.grid()
a.set(xlabel="Leading jet pT (GeV)",ylabel='Efficiency')
a.legend(loc='upper left', bbox_to_anchor=(0.75, 0.99),fontsize=7)
hep.atlas.label(ax=a,label='Work in Progress',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_leading_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")







def get_2eff(subleading_jet_pt_cut):
    event_tru_pt = load_object(metrics_folder+"/tboxes_pt.pkl")
    event_pred_pt = load_object(metrics_folder+"/pboxes_pt.pkl")
    antikt_2lead_pt = np.array([nth_leading_jet_pt(x,2) for x in event_tru_pt])
    pred_2lead_pt = np.array([nth_leading_jet_pt(x,2) for x in event_pred_pt])

    #make a trigger decision based on antikt jet
    trig_decision_akt = np.argwhere(antikt_2lead_pt>subleading_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(pred_2lead_pt>subleading_jet_pt_cut).T[0]

    start,end = 150,650
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

f,a = plt.subplots(1,1,figsize=(8,8))
a.errorbar(xs,ys,xerr=(xs[1]-xs[0])/2,yerr=yerr,elinewidth=0.4,marker='.',ls='none',label='350 GeV Cut',color='red')
a.errorbar(xs2,ys2,xerr=(xs[1]-xs[0])/2,yerr=yerr2,elinewidth=0.4,marker='.',ls='none',label='450 GeV Cut',color='coral')
a.errorbar(xs3,ys3,xerr=(xs[1]-xs[0])/2,yerr=yerr3,elinewidth=0.4,marker='.',ls='none',label='550 GeV Cut',color='peru')
a.grid()
a.set(xlabel="Subleading jet pT (GeV)",ylabel='Efficiency')
a.legend(loc='upper left', bbox_to_anchor=(0.75, 0.99),fontsize=7)
hep.atlas.label(ax=a,label='Work in Progress',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_2leading_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")






def get_3eff(subleading_jet_pt_cut):
    event_tru_pt = load_object(metrics_folder+"/tboxes_pt.pkl")
    event_pred_pt = load_object(metrics_folder+"/pboxes_pt.pkl")
    antikt_2lead_pt = np.array([nth_leading_jet_pt(x,3) for x in event_tru_pt])
    pred_2lead_pt = np.array([nth_leading_jet_pt(x,3) for x in event_pred_pt])

    #make a trigger decision based on antikt jet
    trig_decision_akt = np.argwhere(antikt_2lead_pt>subleading_jet_pt_cut).T[0]
    trig_decision_pred = np.argwhere(pred_2lead_pt>subleading_jet_pt_cut).T[0]

    start,end = 50,450
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
hep.atlas.label(ax=a,label='Work in Progress',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_3leading_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")




def get_Neff(nleading_jet_pt_cut,nthjet):
    event_tru_pt = load_object(metrics_folder+"/tboxes_pt.pkl")
    event_pred_pt = load_object(metrics_folder+"/pboxes_pt.pkl")
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
a.set(xlabel="4th leading jet pT (GeV)",ylabel='Efficiency',ylim=(-0.2,1.2))
a.legend(loc='upper left', bbox_to_anchor=(0.75, 0.99),fontsize=7)
hep.atlas.label(ax=a,label='Work in Progress',data=False,lumi=None,loc=1)
f.subplots_adjust(hspace=0.4)
f.savefig(trig_save_loc + f'/eff_plot_nleading_cuts.{image_format}',dpi=400,format=image_format,bbox_inches="tight")














































