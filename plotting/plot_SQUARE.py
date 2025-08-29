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

MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496

def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)


model_name = "jetSSD_custom_convnext_central_32e"
proc = "JZcomb0_test"

diamond_date = "20250313-06"
square_date  = "20250406-23"

diamond_metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/{proc}/{diamond_date}/box_metrics/"
square_metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/{proc}/{square_date}/box_metrics/"
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/{proc}/{diamond_date}/square_comp/"
if not os.path.exists(save_folder): os.makedirs(save_folder)



print("==================================================================================================")
print(f"Loading DIAMOND matched jets from\n{diamond_metrics_folder}")
print("==================================================================================================\n")

# dR target matched
total_di_dRmatched_tar_pt = np.concatenate(load_object(diamond_metrics_folder+"/tarboxes_dRmatched_pt.pkl"))
total_di_dRmatched_pred_pt = np.concatenate(load_object(diamond_metrics_folder+"/pboxes_dRmatched_pt.pkl"))
total_di_dRmatched_tar_eta = np.concatenate(load_object(diamond_metrics_folder+"/tarboxes_dRmatched_eta.pkl"))
total_di_dRmatched_pred_eta = np.concatenate(load_object(diamond_metrics_folder+"/pboxes_dRmatched_eta.pkl"))

# dR truth matched
total_di_dRtruthmatch_tru_pt    = np.concatenate(load_object(diamond_metrics_folder+"/truboxes_dRtruthmatched_pt.pkl"))
total_di_dRtruthmatch_p_pt    = np.concatenate(load_object(diamond_metrics_folder+"/pboxes_dRtruthmatched_pt.pkl"))
total_di_dRtruthmatch_tru_eta    = np.concatenate(load_object(diamond_metrics_folder+"/truboxes_dRtruthmatched_eta.pkl"))
total_di_dRtruthmatch_p_eta    = np.concatenate(load_object(diamond_metrics_folder+"/pboxes_dRtruthmatched_eta.pkl"))

# TARGET dR matched to TRUTH 
total_di_dRtruthtarmatch_tru_pt    = np.concatenate(load_object(diamond_metrics_folder+"/truboxes_dRtruthtarmatched_pt.pkl"))
total_di_dRtruthtarmatch_tar_pt    = np.concatenate(load_object(diamond_metrics_folder+"/tarboxes_dRtruthtarmatched_pt.pkl"))
total_di_dRtruthtarmatch_tru_eta    = np.concatenate(load_object(diamond_metrics_folder+"/truboxes_dRtruthtarmatched_eta.pkl"))
total_di_dRtruthtarmatch_tar_eta    = np.concatenate(load_object(diamond_metrics_folder+"/tarboxes_dRtruthtarmatched_eta.pkl"))

print("==================================================================================================")
print(f"Loading SQUARE matched jets from\n{square_metrics_folder}")
print("==================================================================================================\n")

# dR target matched
total_sq_dRmatched_tar_pt = np.concatenate(load_object(square_metrics_folder+"/tarboxes_dRmatched_pt.pkl"))
total_sq_dRmatched_pred_pt = np.concatenate(load_object(square_metrics_folder+"/pboxes_dRmatched_pt.pkl"))
total_sq_dRmatched_tar_eta = np.concatenate(load_object(square_metrics_folder+"/tarboxes_dRmatched_eta.pkl"))
total_sq_dRmatched_pred_eta = np.concatenate(load_object(square_metrics_folder+"/pboxes_dRmatched_eta.pkl"))

# dR truth matched
total_sq_dRtruthmatch_tru_pt    = np.concatenate(load_object(square_metrics_folder+"/truboxes_dRtruthmatched_pt.pkl"))
total_sq_dRtruthmatch_p_pt    = np.concatenate(load_object(square_metrics_folder+"/pboxes_dRtruthmatched_pt.pkl"))
total_sq_dRtruthmatch_tru_eta    = np.concatenate(load_object(square_metrics_folder+"/truboxes_dRtruthmatched_eta.pkl"))
total_sq_dRtruthmatch_p_eta    = np.concatenate(load_object(square_metrics_folder+"/pboxes_dRtruthmatched_eta.pkl"))

# TARGET dR matched to TRUTH 
total_sq_dRtruthtarmatch_tru_pt    = np.concatenate(load_object(square_metrics_folder+"/truboxes_dRtruthtarmatched_pt.pkl"))
total_sq_dRtruthtarmatch_tar_pt    = np.concatenate(load_object(square_metrics_folder+"/tarboxes_dRtruthtarmatched_pt.pkl"))
total_sq_dRtruthtarmatch_tru_eta    = np.concatenate(load_object(square_metrics_folder+"/truboxes_dRtruthtarmatched_eta.pkl"))
total_sq_dRtruthtarmatch_tar_eta    = np.concatenate(load_object(square_metrics_folder+"/tarboxes_dRtruthtarmatched_eta.pkl"))





print("==================================================================================================")
print(f"Calculating Jet Energy Response and Resolution")
print("==================================================================================================\n")


# homemade Gaussian to fit
def gaussian(x, a, mean, variance):
    return a * np.exp(-((x - mean)**2 / (2 * variance)))


bin_edges = [30, 40, 55, 80, 110, 150, 200, 300, 400, 550, 800] #[20,35,50,75,100,125,175,225,300,400,500,600]
bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
bin_width = np.diff(bin_edges)



avg_di_resp, std_di_resp = [],[]
fitted_di_mu, fitted_di_mu_unc = [],[]
fitted_di_sig, fitted_di_sig_unc = [],[]
avg_sq_resp, std_sq_resp = [],[]
fitted_sq_mu, fitted_sq_mu_unc = [],[]
fitted_sq_sig, fitted_sq_sig_unc = [],[]
for bin_idx in range(len(bin_edges)-1):

    # find DI matched target jets in each bin
    di_bin_mask = (bin_edges[bin_idx]<total_di_dRmatched_tar_pt) & (total_di_dRmatched_tar_pt<bin_edges[bin_idx+1])

    target_di_jet_pt_in_this_bin = total_di_dRmatched_tar_pt[di_bin_mask]
    pred_di_jet_pt_in_this_bin = total_di_dRmatched_pred_pt[di_bin_mask]
    di_jet_pt_response_bin_i = pred_di_jet_pt_in_this_bin / target_di_jet_pt_in_this_bin
    print(f"Number of DIAMOND jets in bin {bin_idx}: {len(di_jet_pt_response_bin_i)}")
    print(f"pT in [{bin_edges[bin_idx],bin_edges[bin_idx+1]}], np.mean {np.mean(di_jet_pt_response_bin_i):.4f}")

    di_jet_resp_hist, bins = np.histogram(di_jet_pt_response_bin_i, bins=50)
    bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    popt_di_g, pcov_di_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=di_jet_resp_hist, p0=[len(di_jet_pt_response_bin_i),1.0,0.1])
    fit_di_mu = popt_di_g[1]
    fit_di_var = popt_di_g[2] # = V, variance
    fit_di_mu_unc = np.sqrt(np.diag(pcov_di_g))[1]
    fit_di_var_unc = np.sqrt(np.diag(pcov_di_g))[2] # = sigma_V, unc on variance
    print(f"Fit parameters: A = {popt_di_g[0]:.4f}, mu = {fit_di_mu:.4f}, var = {fit_di_var:.4f} ")
    print(f"Fit error:        +- {np.sqrt(np.diag(pcov_di_g))[0]:.3f},    +- {fit_di_mu_unc:.5f},   +- {fit_di_var_unc:.5f}")
    print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!\n")

    avg_di_resp.append(np.mean(di_jet_pt_response_bin_i))
    std_di_resp.append(np.std(di_jet_pt_response_bin_i))
    fitted_di_mu.append(fit_di_mu)
    fitted_di_mu_unc.append(fit_di_mu_unc)
    fitted_di_sig.append(np.sqrt(fit_di_var))
    fitted_di_sig_unc.append((fit_di_var_unc*np.sqrt(fit_di_var)) / (2*fit_di_var))

    # find SQ matched target jets in each bin
    sq_bin_mask = (bin_edges[bin_idx]<total_sq_dRmatched_tar_pt) & (total_sq_dRmatched_tar_pt<bin_edges[bin_idx+1])

    target_sq_jet_pt_in_this_bin = total_sq_dRmatched_tar_pt[sq_bin_mask]
    pred_sq_jet_pt_in_this_bin = total_sq_dRmatched_pred_pt[sq_bin_mask]
    sq_jet_pt_response_bin_i = pred_sq_jet_pt_in_this_bin / target_sq_jet_pt_in_this_bin
    print(f"Number of SQUARE jets in bin {bin_idx}: {len(sq_jet_pt_response_bin_i)}")
    print(f"pT in [{bin_edges[bin_idx],bin_edges[bin_idx+1]}], np.mean {np.mean(sq_jet_pt_response_bin_i):.4f}")

    sq_jet_resp_hist, bins = np.histogram(sq_jet_pt_response_bin_i, bins=50)
    bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    popt_sq_g, pcov_sq_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=sq_jet_resp_hist, p0=[len(sq_jet_pt_response_bin_i),1.0,0.1])
    fit_sq_mu = popt_sq_g[1]
    fit_sq_var = popt_sq_g[2] # = V, variance
    fit_sq_mu_unc = np.sqrt(np.diag(pcov_sq_g))[1]
    fit_sq_var_unc = np.sqrt(np.diag(pcov_sq_g))[2] # = sigma_V, unc on variance
    print(f"Fit parameters: A = {popt_sq_g[0]:.4f}, mu = {fit_sq_mu:.4f}, var = {fit_sq_var:.4f} ")
    print(f"Fit error:        +- {np.sqrt(np.diag(pcov_sq_g))[0]:.3f},    +- {fit_sq_mu_unc:.5f},   +- {fit_sq_var_unc:.5f}")
    print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")

    avg_sq_resp.append(np.mean(sq_jet_pt_response_bin_i))
    std_sq_resp.append(np.std(sq_jet_pt_response_bin_i))
    fitted_sq_mu.append(fit_sq_mu)
    fitted_sq_mu_unc.append(fit_sq_mu_unc)
    fitted_sq_sig.append(np.sqrt(fit_sq_var))
    fitted_sq_sig_unc.append((fit_sq_var_unc*np.sqrt(fit_sq_var)) / (2*fit_sq_var))


    print("---------------------------------------------------------")
    plt.figure()
    plt.stairs(di_jet_resp_hist, bins, fill=True, color='lightskyblue',alpha=0.5)
    plt.hist(di_jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='darkturquoise')
    x = np.linspace(0,di_jet_pt_response_bin_i.max(),100)
    plt.plot(x, gaussian(x, *popt_di_g), linewidth=2.5)
    plt.stairs(sq_jet_resp_hist, bins, fill=True, color='wheat',alpha=0.5)
    plt.hist(sq_jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='darkorange')
    x_sq = np.linspace(0,sq_jet_pt_response_bin_i.max(),100)
    plt.plot(x_sq, gaussian(x_sq, *popt_sq_g), linewidth=2.5)
    plt.xlabel('reco/target jet pt')
    plt.ylabel(f'jets in bin {bin_idx}')
    # plt.yscale('log')
    ax = plt.gca()
    di_text = (f'mu = {popt_di_g[1]:.4f} +- {np.sqrt(np.diag(pcov_di_g))[1]:.4f}  \n'
            f'std = {np.sqrt(fit_di_var):.4f} +- {(fit_di_var_unc*np.sqrt(fit_di_var)) / (2*fit_di_var):.5f}')
    ax.text(0.95, 0.95, f'Fit parameters:\n'+di_text, transform=ax.transAxes, va='top', ha='right',color='darkturquoise')
    sq_text = (f'mu = {popt_sq_g[1]:.4f} +- {np.sqrt(np.diag(pcov_sq_g))[1]:.4f}  \n'
            f'std = {np.sqrt(fit_sq_var):.4f} +- {(fit_sq_var_unc*np.sqrt(fit_sq_var)) / (2*fit_sq_var):.5f}')
    ax.text(0.95, 0.65, f'Fit parameters:\n'+sq_text, transform=ax.transAxes, va='top', ha='right',color='orange')
    plt.title(f'Jet pT in [{bin_edges[bin_idx]},{bin_edges[bin_idx+1]}]')
    plt.savefig(save_folder + f'_{bin_idx}_pt_response_bin.png')
    plt.close()






print("==================================================================================================")
print(f"Plotting Jet Energy Response and Resolution w.r.t target jets")
print("==================================================================================================\n")



plt.figure()
plt.errorbar(bin_centers, fitted_sq_mu, xerr=bin_width/2, yerr=fitted_sq_mu_unc, marker='s', capsize=5, color='tomato',label='Square kernel')
plt.errorbar(bin_centers, fitted_di_mu, xerr=bin_width/2, yerr=fitted_di_mu_unc, marker='D', capsize=5, color='teal',label='Diamond kernel')
plt.axhline(y=1, color='black', linestyle='--', linewidth=2)
plt.xlabel(r'AntiKt4EMTopo Jet $p_T$ (jet constituent scale) [GeV]')
# plt.xlabel(r'Uncalibrated jet $p_T$')
# plt.ylabel('Jet Energy Response')
plt.ylabel(r'$\frac{\text{Predicted jet}\,\, p_T}{\text{Target jet}\,\,\, p_T}$')
# plt.ylabel(r'$\mu \left( \frac{\text{Predicted jet}\,\, p_T}{\text{Target jet}\,\,\, p_T} \right) $')
plt.ylim(0.9,1.65)
plt.xlim(0,400)
plt.legend(loc='lower left',bbox_to_anchor=(0.56, 0.8),fontsize="small",handletextpad=0.8)
plt.text(10,1.6, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
# plt.text(75,1.6, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
plt.text(75,1.6, "Simulation Preliminary",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
plt.text(10,1.564, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=11)
# plt.text(20,1.56, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
# plt.text(140,1.56, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
# plt.text(20,1.535, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=11)
named_proc = "dijet"
# plt.text(20,1.515, f"MC {named_proc} " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=11)
plt.text(10,1.54, f"MC {named_proc} " +  r"$p^{uncalib}_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=11)
plt.savefig(save_folder+'jet_response_simple.png',dpi=400)

plt.figure()
plt.errorbar(bin_centers, fitted_sq_sig, xerr=bin_width/2, yerr=fitted_sq_sig_unc, marker='s', capsize=5, color='tomato',label='Square kernel')
plt.errorbar(bin_centers, fitted_di_sig, xerr=bin_width/2, yerr=fitted_di_sig_unc, marker='D', capsize=5, color='teal',label='Diamond kernel')
# plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.xlabel(r'AntiKt4EMTopo Jet $p_T$ (jet constituent scale) [GeV]')
# plt.xlabel(r'Uncalibrated jet $p_T$')
plt.ylabel(r'$\sigma \left( \frac{\text{Predicted jet}\,\, p_T}{\text{Target jet}\,\,\, p_T} \right) $')
# plt.ylabel('Relative Jet Energy Resolution')
plt.xlim(0,400)
plt.ylim(0.0,0.40)
plt.legend(loc='lower left',bbox_to_anchor=(0.56, 0.8),fontsize="small",handletextpad=0.8)
plt.text(10,0.37, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
# plt.text(75,0.37, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
plt.text(75,0.37, "Simulation Preliminary",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
plt.text(10,0.353, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
named_proc = "dijet"
plt.text(10,0.335, f"MC {named_proc} " +  r"$p^{uncalib}_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)
plt.savefig(save_folder+'jet_resolution_simple.png',dpi=400)






print("==================================================================================================")
print(f"Calculating Jet Energy Response and Resolution w.r.t TRUTH")
print("==================================================================================================\n")

















avg_di_resp, std_di_resp = [],[]
fitted_di_mu, fitted_di_mu_unc = [],[]
fitted_di_sig, fitted_di_sig_unc = [],[]
avg_sq_resp, std_sq_resp = [],[]
fitted_sq_mu, fitted_sq_mu_unc = [],[]
fitted_sq_sig, fitted_sq_sig_unc = [],[]
avg_tar_resp, std_tar_resp = [],[]
fitted_tar_mu, fitted_tar_mu_unc = [],[]
fitted_tar_sig, fitted_tar_sig_unc = [],[]
for bin_idx in range(len(bin_edges)-1):

    # find DI matched TRUTH jets in each bin
    di_bin_mask = (bin_edges[bin_idx]<total_di_dRtruthmatch_tru_pt) & (total_di_dRtruthmatch_tru_pt<bin_edges[bin_idx+1])

    truth_di_jet_pt_in_this_bin = total_di_dRtruthmatch_tru_pt[di_bin_mask]
    pred_di_jet_pt_in_this_bin = total_di_dRtruthmatch_p_pt[di_bin_mask]
    di_jet_pt_response_bin_i = pred_di_jet_pt_in_this_bin / truth_di_jet_pt_in_this_bin
    print(f"Number of DIAMOND jets in bin {bin_idx}: {len(di_jet_pt_response_bin_i)} (matched to truth jets)")
    print(f"pT in [{bin_edges[bin_idx],bin_edges[bin_idx+1]}], np.mean {np.mean(di_jet_pt_response_bin_i):.4f}")

    di_jet_resp_hist, bins = np.histogram(di_jet_pt_response_bin_i, bins=50)
    bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    popt_di_g, pcov_di_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=di_jet_resp_hist, p0=[len(di_jet_pt_response_bin_i),1.0,0.1])
    fit_di_mu = popt_di_g[1]
    fit_di_var = popt_di_g[2] # = V, variance
    fit_di_mu_unc = np.sqrt(np.diag(pcov_di_g))[1]
    fit_di_var_unc = np.sqrt(np.diag(pcov_di_g))[2] # = sigma_V, unc on variance
    print(f"Fit parameters: A = {popt_di_g[0]:.4f}, mu = {fit_di_mu:.4f}, var = {fit_di_var:.4f} ")
    print(f"Fit error:        +- {np.sqrt(np.diag(pcov_di_g))[0]:.3f},    +- {fit_di_mu_unc:.5f},   +- {fit_di_var_unc:.5f}")
    print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!\n")

    avg_di_resp.append(np.mean(di_jet_pt_response_bin_i))
    std_di_resp.append(np.std(di_jet_pt_response_bin_i))
    fitted_di_mu.append(fit_di_mu)
    fitted_di_mu_unc.append(fit_di_mu_unc)
    fitted_di_sig.append(np.sqrt(fit_di_var))
    fitted_di_sig_unc.append((fit_di_var_unc*np.sqrt(fit_di_var)) / (2*fit_di_var))

    # find SQ matched TRUTH jets in each bin
    sq_bin_mask = (bin_edges[bin_idx]<total_sq_dRtruthmatch_tru_pt) & (total_sq_dRtruthmatch_tru_pt<bin_edges[bin_idx+1])

    truth_sq_jet_pt_in_this_bin = total_sq_dRtruthmatch_tru_pt[sq_bin_mask]
    pred_sq_jet_pt_in_this_bin = total_sq_dRtruthmatch_p_pt[sq_bin_mask]
    sq_jet_pt_response_bin_i = pred_sq_jet_pt_in_this_bin / truth_sq_jet_pt_in_this_bin
    print(f"Number of SQUARE jets in bin {bin_idx}: {len(sq_jet_pt_response_bin_i)} (matched to truth jets)")
    print(f"pT in [{bin_edges[bin_idx],bin_edges[bin_idx+1]}], np.mean {np.mean(sq_jet_pt_response_bin_i):.4f}")

    sq_jet_resp_hist, bins = np.histogram(sq_jet_pt_response_bin_i, bins=50)
    bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    popt_sq_g, pcov_sq_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=sq_jet_resp_hist, p0=[len(sq_jet_pt_response_bin_i),1.0,0.1])
    fit_sq_mu = popt_sq_g[1]
    fit_sq_var = popt_sq_g[2] # = V, variance
    fit_sq_mu_unc = np.sqrt(np.diag(pcov_sq_g))[1]
    fit_sq_var_unc = np.sqrt(np.diag(pcov_sq_g))[2] # = sigma_V, unc on variance
    print(f"Fit parameters: A = {popt_sq_g[0]:.4f}, mu = {fit_sq_mu:.4f}, var = {fit_sq_var:.4f} ")
    print(f"Fit error:        +- {np.sqrt(np.diag(pcov_sq_g))[0]:.3f},    +- {fit_sq_mu_unc:.5f},   +- {fit_sq_var_unc:.5f}")
    print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")

    avg_sq_resp.append(np.mean(sq_jet_pt_response_bin_i))
    std_sq_resp.append(np.std(sq_jet_pt_response_bin_i))
    fitted_sq_mu.append(fit_sq_mu)
    fitted_sq_mu_unc.append(fit_sq_mu_unc)
    fitted_sq_sig.append(np.sqrt(fit_sq_var))
    fitted_sq_sig_unc.append((fit_sq_var_unc*np.sqrt(fit_sq_var)) / (2*fit_sq_var))

    # find TARGET matched TRUTH jets in each bin
    tar_bin_mask = (bin_edges[bin_idx]<total_di_dRtruthtarmatch_tru_pt) & (total_di_dRtruthtarmatch_tru_pt<bin_edges[bin_idx+1])

    tru_jet_pt_in_this_bin = total_di_dRtruthtarmatch_tru_pt[tar_bin_mask]
    tar_jet_pt_in_this_bin = total_di_dRtruthtarmatch_tar_pt[tar_bin_mask]
    tar_jet_pt_response_bin_i = tar_jet_pt_in_this_bin / tru_jet_pt_in_this_bin
    print(f"Number of TARGET jets in bin {bin_idx}: {len(tar_jet_pt_response_bin_i)}")
    print(f"pT in [{bin_edges[bin_idx],bin_edges[bin_idx+1]}], np.mean {np.mean(tar_jet_pt_response_bin_i):.4f}")
    
    tar_jet_resp_hist, bins = np.histogram(tar_jet_pt_response_bin_i, bins=50)
    bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=tar_jet_resp_hist, p0=[len(tar_jet_pt_response_bin_i),1.0,0.1])
    fit_mu = popt_g[1]
    fit_var = popt_g[2]
    fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
    fit_var_unc = np.sqrt(np.diag(pcov_g))[2]
    print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_var:.4f} ")
    print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_var_unc:.5f}")

    avg_tar_resp.append(np.mean(tar_jet_pt_response_bin_i))
    std_tar_resp.append(np.std(tar_jet_pt_response_bin_i))
    fitted_tar_mu.append(fit_mu)
    fitted_tar_mu_unc.append(fit_mu_unc)
    fitted_tar_sig.append(np.sqrt(fit_var))
    fitted_tar_sig_unc.append((fit_var_unc*np.sqrt(fit_var)) / (2*fit_var))





    print("---------------------------------------------------------")
    plt.figure()
    plt.stairs(di_jet_resp_hist, bins, fill=True, color='lightskyblue',alpha=0.5)
    plt.hist(di_jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='darkturquoise')
    x = np.linspace(0,di_jet_pt_response_bin_i.max(),100)
    plt.plot(x, gaussian(x, *popt_di_g), linewidth=2.5)
    #
    plt.stairs(sq_jet_resp_hist, bins, fill=True, color='wheat',alpha=0.5)
    plt.hist(sq_jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='darkorange')
    x_sq = np.linspace(0,sq_jet_pt_response_bin_i.max(),100)
    plt.plot(x_sq, gaussian(x_sq, *popt_sq_g), linewidth=2.5)
    #
    plt.stairs(tar_jet_resp_hist, bins, fill=True, color='yellow',alpha=0.5)
    plt.hist(tar_jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='gold')
    x_tar = np.linspace(0,tar_jet_pt_response_bin_i.max(),100)
    plt.plot(x_tar, gaussian(x_tar, *popt_g), linewidth=2.5)
    #
    plt.xlabel('reco/target jet pt')
    plt.ylabel(f'jets in bin {bin_idx}')
    # plt.yscale('log')
    ax = plt.gca()
    di_text = (f'mu = {popt_di_g[1]:.4f} +- {np.sqrt(np.diag(pcov_di_g))[1]:.4f}  \n'
            f'std = {np.sqrt(fit_di_var):.4f} +- {(fit_di_var_unc*np.sqrt(fit_di_var)) / (2*fit_di_var):.5f}')
    ax.text(0.95, 0.95, f'Fit parameters:\n'+di_text, transform=ax.transAxes, va='top', ha='right',color='darkturquoise')
    sq_text = (f'mu = {popt_sq_g[1]:.4f} +- {np.sqrt(np.diag(pcov_sq_g))[1]:.4f}  \n'
            f'std = {np.sqrt(fit_sq_var):.4f} +- {(fit_sq_var_unc*np.sqrt(fit_sq_var)) / (2*fit_sq_var):.5f}')
    ax.text(0.95, 0.75, f'Fit parameters:\n'+sq_text, transform=ax.transAxes, va='top', ha='right',color='orange')
    tar_text = (f'mu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.4f}  \n'
            f'std = {np.sqrt(fit_var):.4f} +- {(fit_var_unc*np.sqrt(fit_var)) / (2*fit_var):.5f}')
    ax.text(0.95, 0.55, f'Fit parameters:\n'+tar_text, transform=ax.transAxes, va='top', ha='right',color='gold')
    plt.title(f'Jet pT in [{bin_edges[bin_idx]},{bin_edges[bin_idx+1]}]')
    plt.savefig(save_folder + f'_{bin_idx}_truth_pt_response_bin.png')
    plt.close()








print("==================================================================================================")
print(f"Plotting Jet Energy Response and Resolution w.r.t TRUTH jets")
print("==================================================================================================\n")

plt.figure()
plt.errorbar(bin_centers, fitted_sq_mu, xerr=bin_width/2, yerr=fitted_sq_mu_unc, marker='s', capsize=5, color='tomato',label='Square kernel')
plt.errorbar(bin_centers, fitted_di_mu, xerr=bin_width/2, yerr=fitted_di_mu_unc, marker='D', capsize=5, color='teal',label='Diamond kernel')
plt.errorbar(bin_centers, fitted_tar_mu, xerr=bin_width/2, yerr=fitted_tar_mu_unc, marker='*', capsize=5, color='gold',label='AKT4EMTopo')
plt.axhline(y=1, color='black', linestyle='--', linewidth=2)
plt.xlabel(r'Truth Jet $p_T$')
plt.ylabel('Jet Energy Response')
plt.legend()
# hep.atlas.label(ax=plt.gca(),label='Work in Progress',data=False,lumi=None,loc=0)
# plt.text(60,1.23, r"MC21, $\sqrt{s}=14\,$TeV $<\mu >=200$")
# plt.text(60,1.21, r"Dijet JZ1-4")
plt.savefig(save_folder+'jet_response_truth.png')

plt.figure()
plt.errorbar(bin_centers, fitted_sq_sig, xerr=bin_width/2, yerr=fitted_sq_sig_unc, marker='s', capsize=5, color='tomato',label='Square kernel')
plt.errorbar(bin_centers, fitted_di_sig, xerr=bin_width/2, yerr=fitted_di_sig_unc, marker='D', capsize=5, color='teal',label='Diamond kernel')
plt.errorbar(bin_centers, fitted_tar_sig, xerr=bin_width/2, yerr=fitted_tar_sig_unc, marker='*', capsize=5, color='gold',label='AKT4EMTopo')
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.xlabel(r'Truth Jet $p_T$')
plt.ylabel('Jet Energy Resolution')
plt.legend()
# hep.atlas.label(ax=plt.gca(),label='Work in Progress',data=False,lumi=None,loc=0)
# plt.text(80,0.23, r"MC21, $\sqrt{s}=14\,$TeV $<\mu >=200$")
# plt.text(80,0.21, r"Dijet JZ1-4")
plt.savefig(save_folder+'jet_resolution_truth.png')


