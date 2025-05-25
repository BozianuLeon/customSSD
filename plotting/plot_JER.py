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
date = "20250313-06"
# date = "20250406-23"
# proc = "ttbar_test"
# date = "20250407-14"

metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/{proc}/{date}/box_metrics/"
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/{proc}/{date}/jet_res/"
if not os.path.exists(save_folder): os.makedirs(save_folder)

# plt.figure()
# plt.text(10,0.65, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
# plt.text(48,0.65, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
# plt.text(80,0.25, r"MC21, $\sqrt{s}=14\,$TeV $<\mu >=200$")
# plt.text(80,0.21, r"Dijet JZ1-4")
# plt.xlim(0,250)
# plt.ylim(0,1.2)
# plt.savefig(save_folder+"/text_test.png")


# load in 
print("==================================================================================================")
print(f"Loading matched jets from\n{metrics_folder}")
print("==================================================================================================\n")
total_matched_tar_pt = np.concatenate(load_object(metrics_folder+"/tarboxes_matched_pt.pkl"))
total_matched_pred_pt = np.concatenate(load_object(metrics_folder+"/pboxes_matched_pt.pkl"))
total_matched_tar_eta = np.concatenate(load_object(metrics_folder+"/tarboxes_matched_eta.pkl"))
total_matched_pred_eta = np.concatenate(load_object(metrics_folder+"/pboxes_matched_eta.pkl"))

total_dRmatched_tar_pt = np.concatenate(load_object(metrics_folder+"/tarboxes_dRmatched_pt.pkl"))
total_dRmatched_pred_pt = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_pt.pkl"))
total_dRmatched_tar_eta = np.concatenate(load_object(metrics_folder+"/tarboxes_dRmatched_eta.pkl"))
total_dRmatched_pred_eta = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_eta.pkl"))

# dR truth matched
total_dRtruthmatch_tru_pt    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthmatched_pt.pkl"))
total_dRtruthmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_pt.pkl"))
total_dRtruthmatch_tru_eta    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthmatched_eta.pkl"))
total_dRtruthmatch_p_eta    = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_eta.pkl"))

# TARGET dR matched to TRUTH 
dRtruthtarmatch_tru_pt    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthtarmatched_pt.pkl"))
dRtruthtarmatch_tar_pt    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRtruthtarmatched_pt.pkl"))
dRtruthtarmatch_tru_eta    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthtarmatched_eta.pkl"))
dRtruthtarmatch_tar_eta    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRtruthtarmatched_eta.pkl"))



# homemade Gaussian to fit
def gaussian(x, a, mean, variance):
    return a * np.exp(-((x - mean)**2 / (2 * variance)))


# which plots to make
inclusive_target = False
central_target   = False
inclusive_truth  = False
central_truth    = False
eta_target       = False
eta_target_var   = False
eta_truth        = False
phi_target       = True
phi_truth        = False





bin_edges = [30, 40, 55, 80, 110, 150, 200, 300, 400, 550, 800] #[20,35,50,75,100,125,175,225,300,400,500,600]
bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
bin_width = np.diff(bin_edges)

if inclusive_target:
    if not os.path.exists(save_folder + "/inclusive/"): os.makedirs(save_folder + "/inclusive/")
    average_response, std_response = list(), list()
    fitted_mu, fitted_mu_unc = list(), list()
    fitted_sigma, fitted_sigma_unc = list(), list()
    for bin_idx in range(len(bin_edges)-1):
        # find jets in each bin
        bin_mask = (bin_edges[bin_idx]<total_dRmatched_tar_pt) & (total_dRmatched_tar_pt<bin_edges[bin_idx+1])

        target_jet_pt_in_this_bin = total_dRmatched_tar_pt[bin_mask]
        pred_jet_pt_in_this_bin = total_dRmatched_pred_pt[bin_mask]
        jet_pt_response_bin_i = pred_jet_pt_in_this_bin / target_jet_pt_in_this_bin
        print(f"Number of jets in bin {bin_idx}: {len(jet_pt_response_bin_i)}")
        print(f"pT in [{bin_edges[bin_idx],bin_edges[bin_idx+1]}], np.mean {np.mean(jet_pt_response_bin_i):.4f}")
        
        jet_resp_hist, bins = np.histogram(jet_pt_response_bin_i, bins=50)
        bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[1.0,1.0,0.1])
        popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[len(jet_pt_response_bin_i),1.0,0.1])
        fit_mu = popt_g[1]
        fit_var = popt_g[2]
        fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
        fit_var_unc = np.sqrt(np.diag(pcov_g))[2]
        print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_var:.4f} ")
        print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_var_unc:.5f}")


        average_response.append(np.mean(jet_pt_response_bin_i))
        std_response.append(np.std(jet_pt_response_bin_i))
        fitted_mu.append(popt_g[1])
        fitted_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
        print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")
        V = popt_g[2]
        sigma_V = np.sqrt(np.diag(pcov_g))[2]

        fitted_sigma.append(np.sqrt(V))
        fitted_sigma_unc.append((sigma_V*np.sqrt(V)) / (2*V))
        print("---------------------------------------------------------")
        
        plt.figure()
        plt.stairs(jet_resp_hist, bins, fill=True, color='orange',alpha=0.5)
        plt.hist(jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='blue')
        x = np.linspace(0,jet_pt_response_bin_i.max(),100)
        plt.plot(x, gaussian(x, *popt_g), linewidth=2.5)
        plt.xlabel('reco/target jet pt')
        plt.ylabel(f'jets in bin {bin_idx}')
        # plt.yscale('log')
        ax = plt.gca()
        text = (f'mu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.4f}  \n'
                f'std = {np.sqrt(V):.4f} +- {(sigma_V*np.sqrt(V)) / (2*V):.5f}\n'
                f'std / pT = {np.sqrt(V) / bin_centers[bin_idx]:.6f}')
        ax.text(0.95, 0.95, f'Fit parameters:\n'+text, transform=ax.transAxes, va='top', ha='right')
        plt.title(f'Jet pT in [{bin_edges[bin_idx]},{bin_edges[bin_idx+1]}]')
        plt.savefig(save_folder + f'/inclusive/pt_response_bin_{bin_idx}.png')
        plt.close()
        
        print()


    plt.figure()
    plt.errorbar(bin_centers, fitted_mu, xerr=bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5, color='tomato',label='CNN jets')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'AntiKt4EMTopo Jet $p_T$ (jet constituent scale)')
    plt.ylabel('Jet Energy Response')
    plt.legend()
    # hep.atlas.label(ax=plt.gca(),label='Work in Progress',data=False,lumi=None,loc=0)
    plt.text(60,1.23, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(185,1.23, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(60,1.215, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(60,1.202, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(save_folder+'/inclusive/jet_response_simple.png')

    plt.figure()
    # plt.errorbar(bin_centers, std_response, xerr=bin_width/2, fmt='o', capsize=5, color='blue',label='Simple np.std')
    plt.errorbar(bin_centers, fitted_sigma, xerr=bin_width/2, yerr=fitted_sigma_unc, fmt='o', capsize=5, color='tomato',label='CNN jets')
    # plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'AntiKt4EMTopo Jet $p_T$ (jet constituent scale)')
    plt.ylabel('Jet Energy Resolution')
    plt.legend()
    # hep.atlas.label(ax=plt.gca(),label='Work in Progress',data=False,lumi=None,loc=0)
    plt.text(80,0.23, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(200,0.23, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(80,0.2175, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(80,0.2055, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(save_folder+'/inclusive/jet_resolution_simple.png')

####################################################################################################################################################################################


if central_target:
    print()
    print("Same as above but now central jets only")
    central_mask = (np.abs(total_dRmatched_tar_eta)>0.2) & (np.abs(total_dRmatched_tar_eta)<0.7)
    sup_central_dRmatch_tar_pt = total_dRmatched_tar_pt[central_mask]
    sup_central_dRmatch_pre_pt = total_dRmatched_pred_pt[central_mask]
    print("Important that these shapes match: ",sup_central_dRmatch_tar_pt.shape, sup_central_dRmatch_pre_pt.shape, total_dRmatched_pred_pt[np.abs(total_dRmatched_pred_eta)<0.7].shape)
    super_central_save_folder = save_folder + "/central/"
    if not os.path.exists(super_central_save_folder): os.makedirs(super_central_save_folder)

    average_response, std_response = list(), list()
    fitted_mu, fitted_mu_unc = list(), list()
    fitted_sigma, fitted_sigma_unc = list(), list()
    for bin_idx in range(len(bin_edges)-1):
        # find jets in each bin
        bin_mask = (bin_edges[bin_idx]<sup_central_dRmatch_tar_pt) & (sup_central_dRmatch_tar_pt<bin_edges[bin_idx+1])

        target_jet_pt_in_this_bin = sup_central_dRmatch_tar_pt[bin_mask]
        pred_jet_pt_in_this_bin = sup_central_dRmatch_pre_pt[bin_mask]
        jet_pt_response_bin_i = pred_jet_pt_in_this_bin / target_jet_pt_in_this_bin
        print(f"Number of jets in bin {bin_idx}: {len(jet_pt_response_bin_i)}")
        print(f"pT in [{bin_edges[bin_idx],bin_edges[bin_idx+1]}], np.mean {np.mean(jet_pt_response_bin_i):.4f}")
        
        jet_resp_hist, bins = np.histogram(jet_pt_response_bin_i, bins=50)
        bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[1.0,1.0,0.1])
        popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[len(jet_pt_response_bin_i),1.0,0.1])
        fit_mu = popt_g[1]
        fit_var = popt_g[2]
        fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
        fit_var_unc = np.sqrt(np.diag(pcov_g))[2]
        print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_var:.4f} ")
        print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_var_unc:.5f}")

        average_response.append(np.mean(jet_pt_response_bin_i))
        std_response.append(np.std(jet_pt_response_bin_i))
        fitted_mu.append(popt_g[1])
        fitted_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
        print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")
        V = popt_g[2]
        sigma_V = np.sqrt(np.diag(pcov_g))[2]

        fitted_sigma.append(np.sqrt(V))
        fitted_sigma_unc.append((sigma_V*np.sqrt(V)) / (2*V))
        print("---------------------------------------------------------")
        
        plt.figure()
        plt.stairs(jet_resp_hist, bins, fill=True, color='orange',alpha=0.5)
        plt.hist(jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='blue')
        x = np.linspace(0,jet_pt_response_bin_i.max(),100)
        plt.plot(x, gaussian(x, *popt_g), linewidth=2.5)
        plt.xlabel('reco/target jet pt')
        plt.ylabel(f'jets in bin {bin_idx}')
        # plt.yscale('log')
        ax = plt.gca()
        text = (f'mu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.4f}  \n'
                f'std = {np.sqrt(V):.4f} +- {(sigma_V*np.sqrt(V)) / (2*V):.5f}\n'
                f'std / pT = {np.sqrt(V) / bin_centers[bin_idx]:.6f}')
        ax.text(0.95, 0.95, f'Fit parameters:\n'+text, transform=ax.transAxes, va='top', ha='right')
        plt.title(f'Jet pT in [{bin_edges[bin_idx]},{bin_edges[bin_idx+1]}]')
        plt.savefig(super_central_save_folder + f'tar_pt_response_bin_{bin_idx}.png')
        plt.close()
        print()

    plt.figure()
    plt.errorbar(bin_centers, fitted_mu, xerr=bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'AntiKt4EMTopo Jet $p_T$ (jet constituent scale)')
    plt.ylabel('Jet Energy Response')
    plt.legend()
    plt.text(80,1.22, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(200,1.22, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(80,1.205, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(80,1.192, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV, $0.2<|\eta|<0.7$",fontfamily='sans-serif',fontsize=12)
    plt.savefig(super_central_save_folder+'central_jet_response_simple.png')

    plt.figure()
    plt.errorbar(bin_centers, fitted_sigma, xerr=bin_width/2, yerr=fitted_sigma_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    plt.xlabel(r'AntiKt4EMTopo Jet $p_T$ (jet constituent scale)')
    plt.ylabel('Jet Energy Resolution')
    plt.legend()
    plt.text(80,0.18, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(200,0.18, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(80,0.169, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(80,0.159, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV, $0.2<|\eta|<0.7$",fontfamily='sans-serif',fontsize=12)
    plt.savefig(super_central_save_folder+'central_jet_resolution_simple.png')


    



####################################################################################################################################################################################
if inclusive_truth:
    print()
    print("\nNow comparing to truth jets")
    if not os.path.exists(save_folder + "/inclusive_truth/"): os.makedirs(save_folder + "/inclusive_truth/")

    average_response, std_response = list(), list()
    fitted_mu, fitted_mu_unc = list(), list()
    fitted_sigma, fitted_sigma_unc = list(), list()
    for bin_idx in range(len(bin_edges)-1):
        # find jets in each bin
        bin_mask = (bin_edges[bin_idx]<total_dRtruthmatch_tru_pt) & (total_dRtruthmatch_tru_pt<bin_edges[bin_idx+1])

        target_jet_pt_in_this_bin = total_dRtruthmatch_tru_pt[bin_mask]
        pred_jet_pt_in_this_bin = total_dRtruthmatch_p_pt[bin_mask]
        jet_pt_response_bin_i = pred_jet_pt_in_this_bin / target_jet_pt_in_this_bin
        print(f"Number of jets in bin {bin_idx}: {len(jet_pt_response_bin_i)}")
        print(f"pT in [{bin_edges[bin_idx],bin_edges[bin_idx+1]}], np.mean {np.mean(jet_pt_response_bin_i):.4f}")
        
        jet_resp_hist, bins = np.histogram(jet_pt_response_bin_i, bins=50)
        bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[1.0,1.0,0.1])
        popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[len(jet_pt_response_bin_i),1.0,0.1])
        fit_mu = popt_g[1]
        fit_var = popt_g[2]
        fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
        fit_var_unc = np.sqrt(np.diag(pcov_g))[2]
        print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_var:.4f} ")
        print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_var_unc:.5f}")


        average_response.append(np.mean(jet_pt_response_bin_i))
        std_response.append(np.std(jet_pt_response_bin_i))
        fitted_mu.append(popt_g[1])
        fitted_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
        print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")
        V = popt_g[2]
        sigma_V = np.sqrt(np.diag(pcov_g))[2]

        fitted_sigma.append(np.sqrt(V))
        fitted_sigma_unc.append((sigma_V*np.sqrt(V)) / (2*V))
        print("---------------------------------------------------------")
        
        plt.figure()
        plt.stairs(jet_resp_hist, bins, fill=True, color='orange',alpha=0.5)
        plt.hist(jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='blue')
        x = np.linspace(0,jet_pt_response_bin_i.max(),100)
        plt.plot(x, gaussian(x, *popt_g), linewidth=2.5)
        plt.xlabel('reco/target jet pt')
        plt.ylabel(f'jets in bin {bin_idx}')
        # plt.yscale('log')
        ax = plt.gca()
        text = (f'mu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.4f}  \n'
                f'std = {np.sqrt(V):.4f} +- {(sigma_V*np.sqrt(V)) / (2*V):.5f}\n'
                f'std / pT = {np.sqrt(V) / bin_centers[bin_idx]:.6f}')
        ax.text(0.95, 0.95, f'Fit parameters:\n'+text, transform=ax.transAxes, va='top', ha='right')
        plt.title(f'Jet pT in [{bin_edges[bin_idx]},{bin_edges[bin_idx+1]}]')
        plt.savefig(save_folder + f'/inclusive_truth/truth_pt_response_bin_{bin_idx}.png')
        plt.close()
        
        print()



    plt.figure()
    # plt.errorbar(bin_centers, average_response, xerr=bin_width/2, fmt='o', capsize=5, color='blue',label='Simple mean')
    plt.errorbar(bin_centers, fitted_mu, xerr=bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'Truth Jet $p_T$')
    plt.ylabel('Jet Energy Response')
    plt.legend()
    plt.text(80,1.07, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(200,1.07, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(80,1.055, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(80,1.041, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(save_folder+'/inclusive_truth/truth_jet_response_simple.png')

    plt.figure()
    # plt.errorbar(bin_centers, std_response, xerr=bin_width/2, fmt='o', capsize=5, color='blue',label='Simple np.std')
    plt.errorbar(bin_centers, fitted_sigma, xerr=bin_width/2, yerr=fitted_sigma_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    # plt.errorbar(bin_centers, abs(np.array(fitted_sigma)), xerr=bin_width/2, yerr=fitted_sigma_unc,alpha=0.5, capsize=5, color='pink')
    plt.xlabel(r'Truth Jet $p_T$')
    plt.ylabel('Jet Energy Resolution')
    plt.legend()
    plt.text(80,0.32, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(200,0.32, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(80,0.303, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(80,0.289, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(save_folder+'/inclusive_truth/truth_jet_resolution_simple.png')


    print()
    print("\nNow comparing targets to truth jets")

    average_tar_response, std_tar_response = list(), list()
    fitted_tar_mu, fitted_tar_mu_unc = list(), list()
    fitted_tar_sigma, fitted_tar_sigma_unc = list(), list()
    for bin_idx in range(len(bin_edges)-1):
        # find jets in each bin
        bin_mask = (bin_edges[bin_idx]<dRtruthtarmatch_tru_pt) & (dRtruthtarmatch_tru_pt<bin_edges[bin_idx+1])

        tru_jet_pt_in_this_bin = dRtruthtarmatch_tru_pt[bin_mask]
        tar_jet_pt_in_this_bin = dRtruthtarmatch_tar_pt[bin_mask]
        tar_jet_pt_response_bin_i = tar_jet_pt_in_this_bin / tru_jet_pt_in_this_bin
        print(f"Number of jets in bin {bin_idx}: {len(tar_jet_pt_response_bin_i)}")
        print(f"pT in [{bin_edges[bin_idx],bin_edges[bin_idx+1]}], np.mean {np.mean(tar_jet_pt_response_bin_i):.4f}")
        
        tar_jet_resp_hist, bins = np.histogram(tar_jet_pt_response_bin_i, bins=50)
        bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=tar_jet_resp_hist, p0=[1.0,1.0,0.1])
        popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=tar_jet_resp_hist, p0=[len(tar_jet_pt_response_bin_i),1.0,0.1])
        fit_mu = popt_g[1]
        fit_var = popt_g[2]
        fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
        fit_var_unc = np.sqrt(np.diag(pcov_g))[2]
        print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_var:.4f} ")
        print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_var_unc:.5f}")


        average_tar_response.append(np.mean(tar_jet_pt_response_bin_i))
        std_tar_response.append(np.std(tar_jet_pt_response_bin_i))
        fitted_tar_mu.append(popt_g[1])
        fitted_tar_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
        print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")
        V = popt_g[2]
        sigma_V = np.sqrt(np.diag(pcov_g))[2]

        fitted_tar_sigma.append(np.sqrt(V))
        fitted_tar_sigma_unc.append((sigma_V*np.sqrt(V)) / (2*V))
        print("---------------------------------------------------------")
        plt.figure()
        plt.stairs(tar_jet_resp_hist, bins, fill=True, color='gold',alpha=0.5)
        plt.hist(tar_jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='mediumslateblue')
        x = np.linspace(0,tar_jet_pt_response_bin_i.max(),100)
        plt.plot(x, gaussian(x, *popt_g), linewidth=2.5)
        plt.xlabel('reco/target jet pt')
        plt.ylabel(f'jets in bin {bin_idx}')
        # plt.yscale('log')
        ax = plt.gca()
        text = (f'mu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.4f}  \n'
                f'std = {np.sqrt(V):.4f} +- {(sigma_V*np.sqrt(V)) / (2*V):.5f}\n'
                f'std / pT = {np.sqrt(V) / bin_centers[bin_idx]:.6f}')
        ax.text(0.95, 0.95, f'Fit parameters:\n'+text, transform=ax.transAxes, va='top', ha='right')
        plt.title(f'Jet pT in [{bin_edges[bin_idx]},{bin_edges[bin_idx+1]}]')
        plt.savefig(save_folder + f'/inclusive_truth/tar_truth_pt_response_bin_{bin_idx}.png')
        plt.close()
        
        print()

        print()


    plt.figure()
    plt.errorbar(bin_centers, fitted_tar_mu, xerr=bin_width/2, yerr=fitted_tar_mu_unc, fmt='o', capsize=5, color='green',label='AKT4EMTopo jets')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'Truth Jet $p_T$')
    plt.ylabel('Jet Energy Response')
    plt.legend()
    hep.atlas.label(ax=plt.gca(),label='Work in Progress',data=False,lumi=None,loc=0)
    plt.savefig(save_folder+'/inclusive_truth/tar_truth_jet_response_simple.png')

    plt.figure()
    plt.errorbar(bin_centers, fitted_tar_sigma, xerr=bin_width/2, yerr=fitted_tar_sigma_unc, fmt='o', capsize=5, color='green',label='AKT4EMTopo jets')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'Truth Jet $p_T$')
    plt.ylabel('Jet Energy Resolution')
    plt.legend()
    hep.atlas.label(ax=plt.gca(),label='Work in Progress',data=False,lumi=None,loc=0)
    plt.savefig(save_folder+'/inclusive_truth/tar_truth_jet_resolution_simple.png')



    plt.figure()
    plt.errorbar(bin_centers, fitted_tar_mu, xerr=bin_width/2, yerr=fitted_tar_mu_unc, fmt='o', capsize=5, color='green',label='AKT4EMTopo jets')
    plt.errorbar(bin_centers, fitted_mu, xerr=bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'Truth Jet $p_T$')
    plt.ylabel('Jet Energy Response')
    plt.legend()
    plt.text(80,1.07, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(200,1.07, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(80,1.052, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(80,1.036, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(save_folder+'/inclusive_truth/comp_truth_jet_response_simple.png')

    plt.figure()
    plt.errorbar(bin_centers, fitted_tar_sigma, xerr=bin_width/2, yerr=fitted_tar_sigma_unc, fmt='o', capsize=5, color='green',label='AKT4EMTopo jets')
    plt.errorbar(bin_centers, fitted_sigma, xerr=bin_width/2, yerr=fitted_sigma_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    plt.xlabel(r'Truth Jet $p_T$')
    plt.ylabel('Jet Energy Resolution')
    plt.legend()
    plt.text(80,0.32, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(200,0.32, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(80,0.303, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(80,0.289, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(save_folder+'/inclusive_truth/comp_truth_jet_resolution_simple.png')








####################################################################################################################################################################################

if central_truth:
    print()
    print("\nNow comparing to truth jets BUT ONLY SUPER CENTRAL JETS")

    super_central_truth_save_folder = save_folder + "/central_truth/"
    if not os.path.exists(super_central_truth_save_folder): os.makedirs(super_central_truth_save_folder)

    sup_pre_central_mask = (np.abs(total_dRtruthmatch_tru_eta)>0.2) & (np.abs(total_dRtruthmatch_tru_eta)<0.7)
    sup_central_dRtruthmatch_tru_pt = total_dRtruthmatch_tru_pt[sup_pre_central_mask]
    sup_central_dRtruthmatch_p_pt = total_dRtruthmatch_p_pt[sup_pre_central_mask]
    print("Important that these shapes match: ",sup_central_dRtruthmatch_tru_pt.shape, sup_central_dRtruthmatch_p_pt.shape, total_dRtruthmatch_p_pt[np.abs(total_dRtruthmatch_p_eta)<0.7].shape)

    average_response, std_response = list(), list()
    fitted_mu, fitted_mu_unc = list(), list()
    fitted_sigma, fitted_sigma_unc = list(), list()
    for bin_idx in range(len(bin_edges)-1):
        # find jets in each bin
        bin_mask = (bin_edges[bin_idx]<sup_central_dRtruthmatch_tru_pt) & (sup_central_dRtruthmatch_tru_pt<bin_edges[bin_idx+1])

        target_jet_pt_in_this_bin = sup_central_dRtruthmatch_tru_pt[bin_mask]
        pred_jet_pt_in_this_bin = sup_central_dRtruthmatch_p_pt[bin_mask]
        jet_pt_response_bin_i = pred_jet_pt_in_this_bin / target_jet_pt_in_this_bin
        print(f"Number of jets in bin {bin_idx}: {len(jet_pt_response_bin_i)}")
        print(f"pT in [{bin_edges[bin_idx],bin_edges[bin_idx+1]}], np.mean {np.mean(jet_pt_response_bin_i):.4f}")
        
        jet_resp_hist, bins = np.histogram(jet_pt_response_bin_i, bins=50)
        bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[1.0,1.0,0.1])
        popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[len(jet_pt_response_bin_i),1.0,0.1])
        fit_mu = popt_g[1]
        fit_var = popt_g[2]
        fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
        fit_var_unc = np.sqrt(np.diag(pcov_g))[2]
        print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_var:.4f} ")
        print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_var_unc:.5f}")


        average_response.append(np.mean(jet_pt_response_bin_i))
        std_response.append(np.std(jet_pt_response_bin_i))
        fitted_mu.append(popt_g[1])
        fitted_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
        print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")
        V = popt_g[2]
        sigma_V = np.sqrt(np.diag(pcov_g))[2]

        fitted_sigma.append(np.sqrt(V))
        fitted_sigma_unc.append((sigma_V*np.sqrt(V)) / (2*V))
        print("---------------------------------------------------------")
        
        plt.figure()
        plt.stairs(jet_resp_hist, bins, fill=True, color='orange',alpha=0.5)
        plt.hist(jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='blue')
        x = np.linspace(0,jet_pt_response_bin_i.max(),100)
        plt.plot(x, gaussian(x, *popt_g), linewidth=2.5)
        plt.xlabel('reco/target jet pt')
        plt.ylabel(f'jets in bin {bin_idx}')
        # plt.yscale('log')
        ax = plt.gca()
        text = (f'mu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.4f}  \n'
                f'std = {np.sqrt(V):.4f} +- {(sigma_V*np.sqrt(V)) / (2*V):.5f}\n'
                f'std / pT = {np.sqrt(V) / bin_centers[bin_idx]:.6f}')
        ax.text(0.95, 0.95, f'Fit parameters:\n'+text, transform=ax.transAxes, va='top', ha='right')
        plt.title(f'Jet pT in [{bin_edges[bin_idx]},{bin_edges[bin_idx+1]}]')
        plt.savefig(super_central_truth_save_folder + f'truth_pt_response_bin_{bin_idx}.png')
        plt.close()
        
        print()




    print("Now match truth jets to targets. SUPER CENTRAL")

    sup_tar_central_mask = (np.abs(dRtruthtarmatch_tru_eta)>0.2) & (np.abs(dRtruthtarmatch_tru_eta)<0.7)
    sup_central_dRtruthtarmatch_tru_pt = dRtruthtarmatch_tru_pt[sup_tar_central_mask]
    sup_central_dRtruthtarmatch_tar_pt = dRtruthtarmatch_tar_pt[sup_tar_central_mask]
    print("Important that these shapes match: ",sup_central_dRtruthtarmatch_tru_pt.shape, sup_central_dRtruthtarmatch_tar_pt.shape, dRtruthtarmatch_tar_pt[np.abs(dRtruthtarmatch_tar_eta)<0.7].shape)

    average_tar_response, std_tar_response = list(), list()
    fitted_tar_mu, fitted_tar_mu_unc = list(), list()
    fitted_tar_sigma, fitted_tar_sigma_unc = list(), list()
    for bin_idx in range(len(bin_edges)-1):
        # find jets in each bin
        bin_mask = (bin_edges[bin_idx]<sup_central_dRtruthtarmatch_tru_pt) & (sup_central_dRtruthtarmatch_tru_pt<bin_edges[bin_idx+1])

        tru_jet_pt_in_this_bin = sup_central_dRtruthtarmatch_tru_pt[bin_mask]
        tar_jet_pt_in_this_bin = sup_central_dRtruthtarmatch_tar_pt[bin_mask]
        tar_jet_pt_response_bin_i = tar_jet_pt_in_this_bin / tru_jet_pt_in_this_bin
        print(f"Number of jets in bin {bin_idx}: {len(tar_jet_pt_response_bin_i)}")
        print(f"pT in [{bin_edges[bin_idx],bin_edges[bin_idx+1]}], np.mean {np.mean(tar_jet_pt_response_bin_i):.4f}")
        
        tar_jet_resp_hist, bins = np.histogram(tar_jet_pt_response_bin_i, bins=50)
        bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=tar_jet_resp_hist, p0=[1.0,1.0,0.1])
        popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=tar_jet_resp_hist, p0=[len(tar_jet_pt_response_bin_i),1.0,0.1])
        fit_mu = popt_g[1]
        fit_var = popt_g[2]
        fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
        fit_var_unc = np.sqrt(np.diag(pcov_g))[2]
        print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_var:.4f} ")
        print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_var_unc:.5f}")


        average_tar_response.append(np.mean(tar_jet_pt_response_bin_i))
        std_tar_response.append(np.std(tar_jet_pt_response_bin_i))
        fitted_tar_mu.append(popt_g[1])
        fitted_tar_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
        print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")
        V = popt_g[2]
        sigma_V = np.sqrt(np.diag(pcov_g))[2]

        fitted_tar_sigma.append(np.sqrt(V))
        fitted_tar_sigma_unc.append((sigma_V*np.sqrt(V)) / (2*V))
        print("---------------------------------------------------------")
        plt.figure()
        plt.stairs(tar_jet_resp_hist, bins, fill=True, color='gold',alpha=0.5)
        plt.hist(tar_jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='mediumslateblue')
        x = np.linspace(0,tar_jet_pt_response_bin_i.max(),100)
        plt.plot(x, gaussian(x, *popt_g), linewidth=2.5)
        plt.xlabel('reco/target jet pt')
        plt.ylabel(f'jets in bin {bin_idx}')
        # plt.yscale('log')
        ax = plt.gca()
        text = (f'mu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.4f}  \n'
                f'std = {np.sqrt(V):.4f} +- {(sigma_V*np.sqrt(V)) / (2*V):.5f}\n'
                f'std / pT = {np.sqrt(V) / bin_centers[bin_idx]:.6f}')
        ax.text(0.95, 0.95, f'Fit parameters:\n'+text, transform=ax.transAxes, va='top', ha='right')
        plt.title(f'Jet pT in [{bin_edges[bin_idx]},{bin_edges[bin_idx+1]}]')
        plt.savefig(super_central_truth_save_folder + f'tar_truth_pt_response_bin_{bin_idx}.png')
        plt.close()
        
        print()
    


    plt.figure()
    plt.errorbar(bin_centers, fitted_tar_mu, xerr=bin_width/2, yerr=fitted_tar_mu_unc, fmt='o', capsize=5, color='green',label='AKT4EMTopo jets')
    plt.errorbar(bin_centers, fitted_mu, xerr=bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'Truth Jet $p_T$')
    plt.ylabel('Jet Energy Response')
    plt.legend()
    plt.text(80,1.15, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(200,1.15, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(80,1.12, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(80,1.095, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV, $0.2<|\eta|<0.7$",fontfamily='sans-serif',fontsize=12)
    plt.savefig(super_central_truth_save_folder+'comp_central_truth_jet_response_simple.png')

    plt.figure()
    plt.errorbar(bin_centers, fitted_tar_sigma, xerr=bin_width/2, yerr=fitted_tar_sigma_unc, fmt='o', capsize=5, color='green',label='AKT4EMTopo jets')
    plt.errorbar(bin_centers, fitted_sigma, xerr=bin_width/2, yerr=fitted_sigma_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'Truth Jet $p_T$')
    plt.ylabel('Jet Energy Resolution')
    plt.legend()
    plt.text(80,0.28, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(200,0.28, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(80,0.264, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(80,0.251, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV, $0.2<|\eta|<0.7$",fontfamily='sans-serif',fontsize=12)
    plt.savefig(super_central_truth_save_folder+'comp_central_truth_jet_resolution_simple.png')











####################################################################################################################################################################################




def fit_response_hist(jet_pt_response_i,nbins=100):

    if len(jet_pt_response_i)==0:
        avg_response = 0
        std_response = 0
        fit_mu = 0
        fit_mu_unc = 0
        fit_sig = 0
        fit_sig_unc = 0
        return avg_response, std_response, fit_mu, fit_mu_unc, fit_sig, fit_sig_unc

    else:
        jet_resp_hist, bins = np.histogram(jet_pt_response_i, bins=nbins)
        bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gausmyfunc, xdata=bin_centres, ydata=jet_resp_hist, bounds=[(0.5,0.0, -np.inf),(np.inf, np.inf, np.inf)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[1.0,1.0,0.1])
        popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[len(jet_pt_response_i),1.0,0.1])
        V = popt_g[2]
        sigma_V = np.sqrt(np.diag(pcov_g))[2]
        avg_response = np.mean(jet_pt_response_i)
        std_response = np.std(jet_pt_response_i)
        fit_mu = popt_g[1]
        fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
        fit_sig = np.sqrt(V)
        fit_sig_unc = (sigma_V*np.sqrt(V)) / (2*V)
        print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_sig:.4f} ")
        print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_sig_unc:.5f}")
        print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")
        return avg_response, std_response, fit_mu, fit_mu_unc, fit_sig, fit_sig_unc



if eta_target_var:
    print()
    print("\nNow binning in eta variable pT")
    eta_save_folder = save_folder + "/eta_target_var/"
    if not os.path.exists(eta_save_folder): os.makedirs(eta_save_folder)

    # eta_bins = [-2.5,-1.8,-1.3,-1.0,-0.7,-0.2,0.0,0.2,0.7,1.0,1.3,1.8,2.5] #
    eta_bins = np.arange(-2.1,2.2,step=0.1)
    eta_bin_centers = eta_bins[:-1] + 0.5 * np.diff(eta_bins)
    eta_bin_width = np.diff(eta_bins)

    low_pt_mask = (total_dRmatched_tar_pt > 20) & (total_dRmatched_tar_pt < 60)
    mid_pt_mask = (total_dRmatched_tar_pt > 60) & (total_dRmatched_tar_pt < 100)
    hi_pt_mask = (total_dRmatched_tar_pt > 100) & (total_dRmatched_tar_pt < 150)
    top_pt_mask = (total_dRmatched_tar_pt > 150)


    low_pt_dRmatch_tar_pt = total_dRmatched_tar_pt[low_pt_mask]
    low_pt_dRmatch_tar_eta = total_dRmatched_tar_eta[low_pt_mask]
    low_pt_dRmatch_pre_pt = total_dRmatched_pred_pt[low_pt_mask]

    mid_pt_dRmatch_tar_pt = total_dRmatched_tar_pt[mid_pt_mask]
    mid_pt_dRmatch_tar_eta = total_dRmatched_tar_eta[mid_pt_mask]
    mid_pt_dRmatch_pre_pt = total_dRmatched_pred_pt[mid_pt_mask]

    hi_pt_dRmatch_tar_pt = total_dRmatched_tar_pt[hi_pt_mask]
    hi_pt_dRmatch_tar_eta = total_dRmatched_tar_eta[hi_pt_mask]
    hi_pt_dRmatch_pre_pt = total_dRmatched_pred_pt[hi_pt_mask]

    top_pt_dRmatch_tar_pt = total_dRmatched_tar_pt[top_pt_mask]
    top_pt_dRmatch_tar_eta = total_dRmatched_tar_eta[top_pt_mask]
    top_pt_dRmatch_pre_pt = total_dRmatched_pred_pt[top_pt_mask]
    print("These shapes, in total",low_pt_dRmatch_tar_pt.shape, mid_pt_dRmatch_tar_pt.shape,hi_pt_dRmatch_tar_pt.shape,top_pt_dRmatch_tar_pt.shape, total_dRmatched_tar_pt.shape)
    

    low_fit_mu,low_fit_mu_unc,low_fit_sigma,low_fit_sigma_unc = [],[],[],[]
    mid_fit_mu,mid_fit_mu_unc,mid_fit_sigma,mid_fit_sigma_unc = [],[],[],[]
    hi_fit_mu,hi_fit_mu_unc,hi_fit_sigma,hi_fit_sigma_unc = [],[],[],[]
    top_fit_mu,top_fit_mu_unc,top_fit_sigma,top_fit_sigma_unc = [],[],[],[]
    low_avg, low_std, mid_avg, mid_std, hi_avg, hi_std, top_avg, top_std = [],[],[],[],[],[],[],[]
    for bin_idx in range(len(eta_bins)-1):
        low_bin_mask = (eta_bins[bin_idx]<low_pt_dRmatch_tar_eta) & (low_pt_dRmatch_tar_eta<eta_bins[bin_idx+1])
        mid_bin_mask = (eta_bins[bin_idx]<mid_pt_dRmatch_tar_eta) & (mid_pt_dRmatch_tar_eta<eta_bins[bin_idx+1])
        hi_bin_mask = (eta_bins[bin_idx]<hi_pt_dRmatch_tar_eta) & (hi_pt_dRmatch_tar_eta<eta_bins[bin_idx+1])
        top_bin_mask = (eta_bins[bin_idx]<top_pt_dRmatch_tar_eta) & (top_pt_dRmatch_tar_eta<eta_bins[bin_idx+1])

        low_jet_pt_response_bin_i = low_pt_dRmatch_pre_pt[low_bin_mask] / low_pt_dRmatch_tar_pt[low_bin_mask] 
        mid_jet_pt_response_bin_i = mid_pt_dRmatch_pre_pt[mid_bin_mask] / mid_pt_dRmatch_tar_pt[mid_bin_mask] 
        hi_jet_pt_response_bin_i  = hi_pt_dRmatch_pre_pt[hi_bin_mask] / hi_pt_dRmatch_tar_pt[hi_bin_mask] 
        top_jet_pt_response_bin_i = top_pt_dRmatch_pre_pt[top_bin_mask] / top_pt_dRmatch_tar_pt[top_bin_mask] 
        print(f"eta in [{eta_bins[bin_idx]:.2f},{eta_bins[bin_idx+1]:.2f}], low {np.mean(low_jet_pt_response_bin_i):.3f}, mid {np.mean(mid_jet_pt_response_bin_i):.3f}, high {np.mean(hi_jet_pt_response_bin_i):.3f}, top {np.mean(top_jet_pt_response_bin_i):.3f}")

        
        lo_avg_resp_i, lo_std_resp_i, lo_fit_mu_i, lo_fit_mu_unc_i, lo_fit_sig_i, lo_fit_sig_unc_i = fit_response_hist(low_jet_pt_response_bin_i,nbins=100)
        mid_avg_resp_i, mid_std_resp_i, mid_fit_mu_i, mid_fit_mu_unc_i, mid_fit_sig_i, mid_fit_sig_unc_i = fit_response_hist(mid_jet_pt_response_bin_i,nbins=100)
        hi_avg_resp_i, hi_std_resp_i, hi_fit_mu_i, hi_fit_mu_unc_i, hi_fit_sig_i, hi_fit_sig_unc_i = fit_response_hist(hi_jet_pt_response_bin_i,nbins=100)
        top_avg_resp_i, top_std_resp_i, top_fit_mu_i, top_fit_mu_unc_i, top_fit_sig_i, top_fit_sig_unc_i = fit_response_hist(top_jet_pt_response_bin_i,nbins=100)
    

        low_fit_mu.append(lo_fit_mu_i)
        low_fit_mu_unc.append(lo_fit_mu_unc_i)
        low_fit_sigma.append(lo_fit_sig_i)
        low_fit_sigma_unc.append(lo_fit_sig_unc_i) 
        mid_fit_mu.append(mid_fit_mu_i)
        mid_fit_mu_unc.append(mid_fit_mu_unc_i)
        mid_fit_sigma.append(mid_fit_sig_i)
        mid_fit_sigma_unc.append(mid_fit_sig_unc_i) 
        hi_fit_mu.append(hi_fit_mu_i)
        hi_fit_mu_unc.append(hi_fit_mu_unc_i)
        hi_fit_sigma.append(hi_fit_sig_i)
        hi_fit_sigma_unc.append(hi_fit_sig_unc_i) 
        top_fit_mu.append(top_fit_mu_i)
        top_fit_mu_unc.append(top_fit_mu_unc_i)
        top_fit_sigma.append(top_fit_sig_i)
        top_fit_sigma_unc.append(top_fit_sig_unc_i) 
        low_avg.append(lo_avg_resp_i)
        low_std.append(lo_std_resp_i)
        mid_avg.append(mid_avg_resp_i)
        mid_std.append(mid_std_resp_i)
        hi_avg.append(hi_avg_resp_i)
        hi_std.append(hi_std_resp_i)
        top_avg.append(top_avg_resp_i)
        top_std.append(top_std_resp_i) 

    plt.figure()
    plt.errorbar(eta_bin_centers, low_fit_mu, xerr=eta_bin_width/2, yerr=low_fit_mu_unc, fmt='o', capsize=5, color='red',label='CNN Jets (pT < 60 GeV)')
    plt.errorbar(eta_bin_centers, mid_fit_mu, xerr=eta_bin_width/2, yerr=mid_fit_mu_unc, fmt='o', capsize=5, color='orange',label='CNN Jets (60 GeV < pT < 100 GeV)')
    plt.errorbar(eta_bin_centers, hi_fit_mu, xerr=eta_bin_width/2, yerr=hi_fit_mu_unc, fmt='o', capsize=5, color='yellow',label='CNN Jets (100 GeV < pT < 150 GeV)')
    plt.errorbar(eta_bin_centers, top_fit_mu, xerr=eta_bin_width/2, yerr=top_fit_mu_unc, fmt='o', capsize=5, color='blue',label='CNN Jets (150 GeV < pT)')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'AntiKt4EMTopo Jet $\eta$ (jet constituent scale)')
    plt.ylabel('Jet Energy Response')
    plt.xlim(-2.5,2.5)
    plt.ylim(0.8,1.5)
    plt.legend(fontsize='x-small')
    # plt.text(-2.1,1.45, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    # plt.text(-1.35,1.45, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    # plt.text(-2.1,1.426, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    # plt.text(-2.1,1.407, r"MC21 Dijet JZ1-4, $p_T > 100\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(eta_save_folder+'comp_jet_pt_response_eta.png')

    plt.figure()
    plt.errorbar(eta_bin_centers, low_fit_mu, xerr=eta_bin_width/2, yerr=low_fit_mu_unc, fmt='o', capsize=5, color='red',label='CNN Jets (pT < 60 GeV)')
    plt.plot(eta_bin_centers, low_avg, marker='x', color='red',label='(pT < 60 GeV)')
    plt.errorbar(eta_bin_centers, mid_fit_mu, xerr=eta_bin_width/2, yerr=mid_fit_mu_unc, fmt='o', capsize=5, color='orange',label='CNN Jets (60 GeV < pT < 100 GeV)')
    plt.plot(eta_bin_centers, mid_avg, marker='x', color='orange',label='(60 GeV < pT < 100 GeV)')
    plt.errorbar(eta_bin_centers, hi_fit_mu, xerr=eta_bin_width/2, yerr=hi_fit_mu_unc, fmt='o', capsize=5, color='yellow',label='CNN Jets (100 GeV < pT < 150 GeV)')
    plt.plot(eta_bin_centers, hi_avg, marker='x', color='yellow',label='(100 GeV < pT < 150 GeV)')
    plt.errorbar(eta_bin_centers, top_fit_mu, xerr=eta_bin_width/2, yerr=top_fit_mu_unc, fmt='o', capsize=5, color='blue',label='CNN Jets (150 GeV < pT)')
    plt.plot(eta_bin_centers, top_avg, marker='x', color='blue',label='(100 GeV < pT)')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'AntiKt4EMTopo Jet $\eta$ (jet constituent scale)')
    plt.ylabel('Jet Energy Response')
    plt.xlim(-2.5,2.5)
    plt.ylim(0.8,1.5)
    plt.legend(fontsize='x-small')
    plt.savefig(eta_save_folder+'mean_std_jet_pt_response_eta.png')


    plt.figure()
    plt.errorbar(eta_bin_centers, low_fit_sigma, xerr=eta_bin_width/2, yerr=low_fit_sigma_unc, fmt='o', capsize=5, color='red',label='CNN Jets (pT < 60 GeV)')
    plt.errorbar(eta_bin_centers, mid_fit_sigma, xerr=eta_bin_width/2, yerr=mid_fit_sigma_unc, fmt='o', capsize=5, color='orange',label='CNN Jets (60 GeV < pT < 100 GeV)')
    plt.errorbar(eta_bin_centers, hi_fit_sigma, xerr=eta_bin_width/2, yerr=hi_fit_sigma_unc, fmt='o', capsize=5, color='yellow',label='CNN Jets (100 GeV < pT < 150 GeV)')
    plt.errorbar(eta_bin_centers, top_fit_sigma, xerr=eta_bin_width/2, yerr=top_fit_sigma_unc, fmt='o', capsize=5, color='blue',label='CNN Jets (150 GeV < pT)')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'AntiKt4EMTopo Jet $\eta$ (jet constituent scale)')
    plt.ylabel('Jet Energy Resolution')
    plt.xlim(-2.5,2.5)
    plt.legend(fontsize='x-small')
    plt.savefig(eta_save_folder+'comp_jet_pt_resolution_eta.png')

    plt.figure()
    plt.errorbar(eta_bin_centers, low_fit_sigma, xerr=eta_bin_width/2, yerr=low_fit_sigma_unc, fmt='o', capsize=5, color='red',label='CNN Jets (pT < 60 GeV)')
    plt.plot(eta_bin_centers, low_std, marker='x', color='red',label='(pT < 60 GeV)')
    plt.errorbar(eta_bin_centers, mid_fit_sigma, xerr=eta_bin_width/2, yerr=mid_fit_sigma_unc, fmt='o', capsize=5, color='orange',label='CNN Jets (60 GeV < pT < 100 GeV)')
    plt.plot(eta_bin_centers, mid_std, marker='x', color='orange',label='(60 GeV < pT < 100 GeV)')
    plt.errorbar(eta_bin_centers, hi_fit_sigma, xerr=eta_bin_width/2, yerr=hi_fit_sigma_unc, fmt='o', capsize=5, color='yellow',label='CNN Jets (100 GeV < pT < 150 GeV)')
    plt.plot(eta_bin_centers, hi_std, marker='x', color='yellow',label='(100 GeV < pT < 150 GeV)')
    plt.errorbar(eta_bin_centers, top_fit_sigma, xerr=eta_bin_width/2, yerr=top_fit_sigma_unc, fmt='o', capsize=5, color='blue',label='CNN Jets (150 GeV < pT)')
    plt.plot(eta_bin_centers, top_std, marker='x', color='blue',label='(100 GeV < pT)')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'AntiKt4EMTopo Jet $\eta$ (jet constituent scale)')
    plt.ylabel('Jet Energy Resolution')
    plt.xlim(-2.5,2.5)
    plt.legend(fontsize='x-small')
    plt.savefig(eta_save_folder+'mean_std_jet_pt_resolution_eta.png')






if eta_target:
    print()
    print("\nNow binning in eta")
    eta_save_folder = save_folder + "/eta_target/"
    if not os.path.exists(eta_save_folder): os.makedirs(eta_save_folder)

    # eta_bins = [-2.5,-1.8,-1.3,-1.0,-0.7,-0.2,0.0,0.2,0.7,1.0,1.3,1.8,2.5] #
    eta_bins = np.arange(-2.1,2.2,step=0.1)
    eta_bin_centers = eta_bins[:-1] + 0.5 * np.diff(eta_bins)
    eta_bin_width = np.diff(eta_bins)


    high_pt_mask = (total_dRmatched_tar_pt>0)
    high_pt_dRmatch_tar_pt = total_dRmatched_tar_pt[high_pt_mask]
    high_pt_dRmatch_tar_eta = total_dRmatched_tar_eta[high_pt_mask]
    high_pt_dRmatch_pre_pt = total_dRmatched_pred_pt[high_pt_mask]
    print("Important that these shapes match: ",high_pt_dRmatch_tar_pt.shape, high_pt_dRmatch_pre_pt.shape)

    average_response, std_response = list(), list()
    fitted_mu, fitted_mu_unc = list(), list()
    fitted_sigma, fitted_sigma_unc = list(), list()
    for bin_idx in range(len(eta_bins)-1):
        bin_mask = (eta_bins[bin_idx]<high_pt_dRmatch_tar_eta) & (high_pt_dRmatch_tar_eta<eta_bins[bin_idx+1])

        target_jet_pt_in_this_bin = high_pt_dRmatch_tar_pt[bin_mask]
        pred_jet_pt_in_this_bin = high_pt_dRmatch_pre_pt[bin_mask]
        jet_pt_response_bin_i = pred_jet_pt_in_this_bin / target_jet_pt_in_this_bin
        print(f"Number of jets in bin {bin_idx}: {len(jet_pt_response_bin_i)}")
        print(f"eta in [{eta_bins[bin_idx],eta_bins[bin_idx+1]}], np.mean {np.mean(jet_pt_response_bin_i):.4f}")

        if len(jet_pt_response_bin_i)==0:
            average_response.append(0)
            std_response.append(0)
            fitted_mu.append(0)
            fitted_mu_unc.append(0)
            fitted_sigma.append(0)
            fitted_sigma_unc.append(0)
            continue

        jet_resp_hist, bins = np.histogram(jet_pt_response_bin_i, bins=100)
        bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gausmyfunc, xdata=bin_centres, ydata=jet_resp_hist, bounds=[(0.5,0.0, -np.inf),(np.inf, np.inf, np.inf)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[1.0,1.0,0.1])
        popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[len(jet_pt_response_bin_i),1.0,0.1])
        fit_mu = popt_g[1]
        fit_var = popt_g[2]
        fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
        fit_var_unc = np.sqrt(np.diag(pcov_g))[2]
        print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_var:.4f} ")
        print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_var_unc:.5f}")

        average_response.append(np.mean(jet_pt_response_bin_i))
        std_response.append(np.std(jet_pt_response_bin_i))
        fitted_mu.append(popt_g[1])
        fitted_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
        print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")
        V = popt_g[2]
        sigma_V = np.sqrt(np.diag(pcov_g))[2]

        fitted_sigma.append(np.sqrt(V))
        fitted_sigma_unc.append((sigma_V*np.sqrt(V)) / (2*V))
        print("---------------------------------------------------------")
        plt.figure()
        plt.stairs(jet_resp_hist, bins, fill=True, color='orange',alpha=0.5)
        plt.hist(jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='blue')
        x = np.linspace(0,jet_pt_response_bin_i.max(),100)
        plt.plot(x, gaussian(x, *popt_g), linewidth=2.5, label='Custom gausmyfunc')
        plt.xlabel('reco/target jet pt')
        plt.ylabel(f'jets in bin {bin_idx}')
        ax = plt.gca()
        # ax.text(0.95, 0.95, f'Fit parameters:\nmu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.3f}\nstd = {popt_g[2]:.4f} +- {np.sqrt(np.diag(pcov_g))[2]:.5f}', transform=ax.transAxes, va='top', ha='right')
        text = (f'mu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.4f}  \n'
                f'std = {np.sqrt(V):.4f} +- {(sigma_V*np.sqrt(V)) / (2*V):.5f}\n'
                f'std / pT = {np.sqrt(V) / eta_bin_centers[bin_idx]:.6f}')
        ax.text(0.95, 0.95, f'Fit parameters:\n'+text, transform=ax.transAxes, va='top', ha='right')
        plt.title(f'Jet eta in [{eta_bins[bin_idx]:.3f},{eta_bins[bin_idx+1]:.3f}], {len(jet_pt_response_bin_i)} jets')
        plt.savefig(eta_save_folder + f'pt_response_eta_bin_{bin_idx}.png')
        plt.close()
        print()


    plt.figure()
    plt.errorbar(eta_bin_centers, fitted_mu, xerr=eta_bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5, color='orange',label='Fit param')
    plt.errorbar(eta_bin_centers, average_response, xerr=eta_bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5,alpha=0.5, color='blue',label='np.mean')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'AntiKt4EMTopo Jet $\eta$ (jet constituent scale)')
    plt.ylabel('Jet Energy Response')
    plt.xlim(-2.5,2.5)
    plt.ylim(0.8,1.5)
    plt.legend()
    plt.text(-2.1,1.45, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(-1.35,1.45, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(-2.1,1.426, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(-2.1,1.407, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(eta_save_folder+'jet_response_eta_simple.png')

    plt.figure()
    # plt.errorbar(eta_bin_centers, std_response, xerr=eta_bin_width/2, fmt='o', capsize=5, color='blue',label='Simple np.std')
    plt.errorbar(eta_bin_centers, fitted_sigma, xerr=eta_bin_width/2, yerr=fitted_sigma_unc, fmt='o', capsize=5, color='orange',label='Fit param')
    plt.errorbar(eta_bin_centers, abs(np.array(fitted_sigma)), xerr=eta_bin_width/2, yerr=fitted_sigma_unc,alpha=0.5, capsize=5, color='pink')
    plt.errorbar(eta_bin_centers, std_response, xerr=eta_bin_width/2, yerr=fitted_sigma_unc,alpha=0.5, capsize=5, color='blue')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'AntiKt4EMTopo Jet $\eta$ (jet constituent scale)')
    plt.ylabel('Jet Energy Resolution')
    plt.xlim(-2.5,2.5)
    plt.legend()
    plt.savefig(eta_save_folder+'jet_resolution_eta_simple.png')






if eta_truth:
    print()
    print("Eta binning compare to truth")

    eta_save_folder = save_folder + "/eta_truth/"
    if not os.path.exists(eta_save_folder): os.makedirs(eta_save_folder)

    # eta_bins = [-2.5,-1.8,-1.3,-1.0,-0.7,-0.2,0.0,0.2,0.7,1.0,1.3,1.8,2.5] #
    eta_bins = np.arange(-2.1,2.2,step=0.1)
    eta_bin_centers = eta_bins[:-1] + 0.5 * np.diff(eta_bins)
    eta_bin_width = np.diff(eta_bins)

    average_response, std_response = list(), list()
    fitted_mu, fitted_mu_unc = list(), list()
    fitted_sigma, fitted_sigma_unc = list(), list()
    for bin_idx in range(len(eta_bins)-1):
        bin_mask = (eta_bins[bin_idx]<total_dRtruthmatch_tru_eta) & (total_dRtruthmatch_tru_eta<eta_bins[bin_idx+1])

        target_jet_pt_in_this_bin = total_dRtruthmatch_tru_pt[bin_mask]
        pred_jet_pt_in_this_bin = total_dRtruthmatch_p_pt[bin_mask]
        jet_pt_response_bin_i = pred_jet_pt_in_this_bin / target_jet_pt_in_this_bin
        print(f"Number of jets in bin {bin_idx}: {len(jet_pt_response_bin_i)}")
        print(f"eta in [{eta_bins[bin_idx]:.3f},{eta_bins[bin_idx+1]:.3f}], np.mean {np.mean(jet_pt_response_bin_i):.4f}")

        if len(jet_pt_response_bin_i)==0:
            average_response.append(0)
            std_response.append(0)
            fitted_mu.append(np.nan)
            fitted_mu_unc.append(0)
            fitted_sigma.append(np.nan)
            fitted_sigma_unc.append(0)
            continue

        jet_resp_hist, bins = np.histogram(jet_pt_response_bin_i, bins=100)
        bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        # # popt_g, pcov_g = scipy.optimize.curve_fit(gausmyfunc, xdata=bin_centres, ydata=jet_resp_hist, bounds=[(0.5,0.0, -np.inf),(np.inf, np.inf, np.inf)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[len(jet_pt_response_bin_i),1.0,0.1])
        try:
            # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[1.0,1.0,0.1])
            popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[len(jet_pt_response_bin_i),1.0,0.1])
            # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, bounds=[(-np.inf,0.0,-np.inf),(np.inf, np.inf, np.inf)], maxfev=1200)
        except RuntimeError:
            popt_g = np.array([0,0,0])
            pcov_g = np.array([[0,0,0],[0,0,0],[0,0,0]])
        fit_mu = popt_g[1]
        fit_var = popt_g[2]
        fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
        fit_var_unc = np.sqrt(np.diag(pcov_g))[2]
        print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_var:.4f} ")
        print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_var_unc:.5f}")

        average_response.append(np.mean(jet_pt_response_bin_i))
        std_response.append(np.std(jet_pt_response_bin_i))
        fitted_mu.append(popt_g[1])
        fitted_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
        print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")
        V = popt_g[2]
        sigma_V = np.sqrt(np.diag(pcov_g))[2]

        fitted_sigma.append(np.sqrt(V))
        fitted_sigma_unc.append((sigma_V*np.sqrt(V)) / (2*V))
        print("---------------------------------------------------------")


        plt.figure()
        plt.stairs(jet_resp_hist, bins, fill=True, color='orange',alpha=0.5)
        plt.hist(jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='blue')
        x = np.linspace(0,jet_pt_response_bin_i.max(),100)
        plt.plot(x, gaussian(x, *popt_g), linewidth=2.5, label='Custom gausmyfunc')
        plt.xlabel('reco/target jet pt')
        plt.ylabel(f'jets in bin {bin_idx}')
        ax = plt.gca()
        text = (f'mu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.4f}  \n'
                f'std = {np.sqrt(V):.4f} +- {(sigma_V*np.sqrt(V)) / (2*V):.5f}\n'
                f'std / pT = {np.sqrt(V) / eta_bin_centers[bin_idx]:.6f}')
        ax.text(0.95, 0.95, f'Fit parameters:\n'+text, transform=ax.transAxes, va='top', ha='right')
        plt.title(f'Jet eta in [{eta_bins[bin_idx]:.3f},{eta_bins[bin_idx+1]:.3f}], {len(jet_pt_response_bin_i)} jets')
        plt.savefig(eta_save_folder + f'truth_pt_response_eta_bin_{bin_idx}.png')
        plt.close()
        print()

    print()
    print("\nNow comparing targets to truth jets in eta")
    average_tar_response, std_tar_response = list(), list()
    fitted_tar_mu, fitted_tar_mu_unc = list(), list()
    fitted_tar_sigma, fitted_tar_sigma_unc = list(), list()
    for bin_idx in range(len(eta_bins)-1):
        # find jets in each bin
        bin_mask = (eta_bins[bin_idx]<dRtruthtarmatch_tru_eta) & (dRtruthtarmatch_tru_eta<eta_bins[bin_idx+1])

        tru_jet_pt_in_this_bin = dRtruthtarmatch_tru_pt[bin_mask]
        tar_jet_pt_in_this_bin = dRtruthtarmatch_tar_pt[bin_mask]
        tar_jet_pt_response_bin_i = tar_jet_pt_in_this_bin / tru_jet_pt_in_this_bin
        print(f"Number of jets in bin {bin_idx}: {len(tar_jet_pt_response_bin_i)}")
        print(f"eta in [{eta_bins[bin_idx]:.3f},{eta_bins[bin_idx+1]:.3f}], np.mean {np.mean(tar_jet_pt_response_bin_i):.4f}")
        if len(tar_jet_pt_response_bin_i)==0:
            average_tar_response.append(0)
            std_tar_response.append(0)
            fitted_tar_mu.append(np.nan)
            fitted_tar_mu_unc.append(0)
            fitted_tar_sigma.append(np.nan)
            fitted_tar_sigma_unc.append(0)
            continue

        tar_jet_resp_hist, bins = np.histogram(tar_jet_pt_response_bin_i, bins=100)
        bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=tar_jet_resp_hist, p0=[1.0,1.0,0.1])
        popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=tar_jet_resp_hist, p0=[len(tar_jet_pt_response_bin_i),1.0,0.1])
        fit_mu = popt_g[1]
        fit_var = popt_g[2]
        fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
        fit_var_unc = np.sqrt(np.diag(pcov_g))[2]
        print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_var:.4f} ")
        print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_var_unc:.5f}")


        average_tar_response.append(np.mean(tar_jet_pt_response_bin_i))
        std_tar_response.append(np.std(tar_jet_pt_response_bin_i))
        fitted_tar_mu.append(popt_g[1])
        fitted_tar_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
        print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")
        V = popt_g[2]
        sigma_V = np.sqrt(np.diag(pcov_g))[2]

        fitted_tar_sigma.append(np.sqrt(V))
        fitted_tar_sigma_unc.append((sigma_V*np.sqrt(V)) / (2*V))
        print("---------------------------------------------------------")
        plt.figure()
        plt.stairs(tar_jet_resp_hist, bins, fill=True, color='gold',alpha=0.5)
        plt.hist(tar_jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='mediumslateblue')
        x = np.linspace(0,tar_jet_pt_response_bin_i.max(),100)
        plt.plot(x, gaussian(x, *popt_g), linewidth=2.5)
        plt.xlabel('reco/target jet pt')
        plt.ylabel(f'jets in bin {bin_idx}')
        # plt.yscale('log')
        ax = plt.gca()
        text = (f'mu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.4f}  \n'
                f'std = {np.sqrt(V):.4f} +- {(sigma_V*np.sqrt(V)) / (2*V):.5f}\n'
                f'std / pT = {np.sqrt(V) / eta_bin_centers[bin_idx]:.6f}')
        ax.text(0.95, 0.95, f'Fit parameters:\n'+text, transform=ax.transAxes, va='top', ha='right')
        plt.title(f'Jet pT in [{eta_bins[bin_idx]},{eta_bins[bin_idx+1]}]')
        plt.savefig(eta_save_folder + f'tar_truth_pt_response_eta_bin_{bin_idx}.png')
        plt.close()
        

    plt.figure()
    plt.errorbar(eta_bin_centers, fitted_tar_mu, xerr=eta_bin_width/2, yerr=fitted_tar_mu_unc, fmt='o', capsize=5, color='green',label='AKT4EMTopo jets')
    plt.errorbar(eta_bin_centers, fitted_mu, xerr=eta_bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    plt.errorbar(eta_bin_centers, average_response, xerr=eta_bin_width/2, yerr=0.01*np.ones_like(average_response), fmt='o', capsize=5, color='blue',label='CNN np.mean')
    plt.errorbar(eta_bin_centers, average_tar_response, xerr=eta_bin_width/2, yerr=0.01*np.ones_like(average_tar_response), fmt='o', capsize=5, color='pink',label='AKT np.mean')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlim((-2.5,2.5))
    plt.xlabel(r'Truth Jet $\eta$')
    plt.ylabel('Jet Energy Response')
    plt.legend()
    # hep.atlas.label(ax=plt.gca(),label='Work in Progress',data=False,lumi=None,loc=0)
    # plt.text(80,1.07, r"MC21, $\sqrt{s}=14\,$TeV $<\mu >=200$")
    # plt.text(80,1.05, r"Dijet JZ1-4")
    plt.savefig(eta_save_folder+'comp_truth_jet_response_eta2.png')

    plt.figure()
    plt.errorbar(eta_bin_centers, fitted_tar_mu, xerr=eta_bin_width/2, yerr=fitted_tar_mu_unc, fmt='o', capsize=5, color='green',label='AKT4EMTopo jets')
    plt.errorbar(eta_bin_centers, fitted_mu, xerr=eta_bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    # plt.errorbar(eta_bin_centers, average_response, xerr=eta_bin_width/2, yerr=0.01*np.ones_like(average_response), fmt='o', capsize=5, color='pink',label='CNN jets (np.mean)')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlim((-2.5,2.5))
    plt.xlabel(r'Truth Jet $\eta$')
    plt.ylabel('Jet Energy Response')
    plt.legend()
    plt.text(-2.1,0.95, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(-1.35,0.95, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(-2.1,0.93, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(-2.1,0.91, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(eta_save_folder+'comp_truth_jet_response_eta.png')

    plt.figure()
    plt.errorbar(eta_bin_centers, fitted_tar_sigma, xerr=eta_bin_width/2, yerr=fitted_tar_sigma_unc, fmt='o', capsize=5, color='green',label='AKT4EMTopo jets')
    plt.errorbar(eta_bin_centers, fitted_sigma, xerr=eta_bin_width/2, yerr=fitted_sigma_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    # plt.errorbar(eta_bin_centers, std_response, xerr=eta_bin_width/2, yerr=0.01*np.ones_like(std_response), fmt='o', capsize=5, color='pink',label='CNN jets (np.std)')
    plt.xlim((-2.5,2.5))
    plt.xlabel(r'Truth Jet $\eta$')
    plt.ylabel('Jet Energy Resolution')
    plt.legend()
    plt.text(-2.2,0.245, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(-1.4,0.245, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(-2.2,0.235, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(-2.2,0.226, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(eta_save_folder+'comp_truth_jet_resolution_eta.png')































if phi_target:

    total_matched_tar_phi = np.concatenate(load_object(metrics_folder+"/tarboxes_matched_phi.pkl"))
    total_matched_pred_phi = np.concatenate(load_object(metrics_folder+"/pboxes_matched_phi.pkl"))
    total_dRmatched_tar_phi = np.concatenate(load_object(metrics_folder+"/tarboxes_dRmatched_phi.pkl"))
    total_dRmatched_pred_phi = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_phi.pkl"))


    print()
    print("\nNow binning in phi!!!")
    phi_save_folder = save_folder + "/phi_target/"
    if not os.path.exists(phi_save_folder): os.makedirs(phi_save_folder)

    # phi_bins = np.arange(-3.3,3.3,step=0.25)
    phi_bins = [-3.3,-3.1,-2.9,-2.7,-2.5,-2.3,-2.1,-1.9,-1.7,-1.5,-1.3,-1.1,-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3] #
    phi_bin_centers = phi_bins[:-1] + 0.5 * np.diff(phi_bins)
    phi_bin_width = np.diff(phi_bins)

    high_pt_mask = (total_dRmatched_tar_pt>0)
    high_pt_dRmatch_tar_pt = total_dRmatched_tar_pt[high_pt_mask]
    high_pt_dRmatch_tar_phi = total_dRmatched_tar_phi[high_pt_mask]
    high_pt_dRmatch_pre_pt = total_dRmatched_pred_pt[high_pt_mask]
    print("Important that these shapes match: ",high_pt_dRmatch_tar_pt.shape, high_pt_dRmatch_pre_pt.shape)

    average_response, std_response = list(), list()
    fitted_mu, fitted_mu_unc = list(), list()
    fitted_sigma, fitted_sigma_unc = list(), list()
    for bin_idx in range(len(phi_bins)-1):
        bin_mask = (phi_bins[bin_idx]<high_pt_dRmatch_tar_phi) & (high_pt_dRmatch_tar_phi<phi_bins[bin_idx+1])

        target_jet_pt_in_this_bin = high_pt_dRmatch_tar_pt[bin_mask]
        pred_jet_pt_in_this_bin = high_pt_dRmatch_pre_pt[bin_mask]
        jet_pt_response_bin_i = pred_jet_pt_in_this_bin / target_jet_pt_in_this_bin
        print(f"Number of jets in bin {bin_idx}: {len(jet_pt_response_bin_i)}")
        print(f"phi in [{phi_bins[bin_idx]:.2f},{phi_bins[bin_idx+1]:.2f}], np.mean {np.mean(jet_pt_response_bin_i):.4f}")

        if len(jet_pt_response_bin_i)==0:
            average_response.append(0)
            std_response.append(0)
            fitted_mu.append(0)
            fitted_mu_unc.append(0)
            fitted_sigma.append(0)
            fitted_sigma_unc.append(0)
            continue

        jet_resp_hist, bins = np.histogram(jet_pt_response_bin_i, bins=100)
        bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gausmyfunc, xdata=bin_centres, ydata=jet_resp_hist, bounds=[(0.5,0.0, -np.inf),(np.inf, np.inf, np.inf)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[1.0,1.0,0.1])
        popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[len(jet_pt_response_bin_i),1.0,0.1])
        fit_mu = popt_g[1]
        fit_var = popt_g[2]
        fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
        fit_var_unc = np.sqrt(np.diag(pcov_g))[2]
        print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_var:.4f} ")
        print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_var_unc:.5f}")

        average_response.append(np.mean(jet_pt_response_bin_i))
        std_response.append(np.std(jet_pt_response_bin_i))
        fitted_mu.append(popt_g[1])
        fitted_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
        print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")
        V = popt_g[2]
        sigma_V = np.sqrt(np.diag(pcov_g))[2]

        fitted_sigma.append(np.sqrt(V))
        fitted_sigma_unc.append((sigma_V*np.sqrt(V)) / (2*V))
        print("---------------------------------------------------------")
        plt.figure()
        plt.stairs(jet_resp_hist, bins, fill=True, color='orange',alpha=0.5)
        plt.hist(jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='blue')
        x = np.linspace(0,jet_pt_response_bin_i.max(),100)
        plt.plot(x, gaussian(x, *popt_g), linewidth=2.5, label='Custom gausmyfunc')
        plt.xlabel('reco/target jet pt')
        plt.ylabel(f'jets in bin {bin_idx}')
        ax = plt.gca()
        # ax.text(0.95, 0.95, f'Fit parameters:\nmu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.3f}\nstd = {popt_g[2]:.4f} +- {np.sqrt(np.diag(pcov_g))[2]:.5f}', transform=ax.transAxes, va='top', ha='right')
        text = (f'mu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.4f}  \n'
                f'std = {np.sqrt(V):.4f} +- {(sigma_V*np.sqrt(V)) / (2*V):.5f}\n'
                f'std / pT = {np.sqrt(V) / phi_bin_centers[bin_idx]:.6f}')
        ax.text(0.95, 0.95, f'Fit parameters:\n'+text, transform=ax.transAxes, va='top', ha='right')
        plt.title(f'Jet eta in [{phi_bins[bin_idx]:.3f},{phi_bins[bin_idx+1]:.3f}], {len(jet_pt_response_bin_i)} jets')
        plt.savefig(phi_save_folder + f'pt_response_phi_bin_{bin_idx}.png')
        plt.close()
        print()


    plt.figure()
    plt.errorbar(phi_bin_centers, fitted_mu, xerr=phi_bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5, color='tomato',label='CNN jets')
    # plt.errorbar(phi_bin_centers, average_response, xerr=phi_bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5,alpha=0.5, color='blue',label='np.mean')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlabel(r'AntiKt4EMTopo Jet $\phi$ (jet constituent scale)')
    plt.ylabel('Jet Energy Response')
    plt.xlim(-3.5,3.5)
    plt.ylim(0.975,1.2)
    plt.legend()
    plt.text(-3.15,1.185, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(-2.11,1.185, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(-3.15,1.176, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(-3.15,1.168, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(phi_save_folder+'jet_response_phi_simple.png')

    plt.figure()
    plt.errorbar(phi_bin_centers, fitted_sigma, xerr=phi_bin_width/2, yerr=fitted_sigma_unc, fmt='o', capsize=5, color='tomato',label='CNN Jets')
    # plt.errorbar(phi_bin_centers, std_response, xerr=phi_bin_width/2, yerr=fitted_sigma_unc,alpha=0.5, capsize=5, color='blue')
    plt.xlabel(r'AntiKt4EMTopo Jet $\phi$ (jet constituent scale)')
    plt.ylabel('Jet Energy Resolution')
    # plt.legend(loc='lower left', bbox_to_anchor=(0.0,0.64))
    plt.xlim(-3.5,3.5)
    plt.ylim(0.0,0.3)
    plt.text(-3.1,0.28, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(-2.,0.28, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(-3.1,0.27, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(-3.1,0.263, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(phi_save_folder+'jet_resolution_phi_simple.png')






if phi_truth:
    print()
    print("phi binning compare to truth")

    # dR truth matched
    total_dRtruthmatch_tru_phi    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthmatched_phi.pkl"))
    total_dRtruthmatch_p_phi    = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_phi.pkl"))
    # TARGET dR matched to TRUTH 
    dRtruthtarmatch_tru_phi    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthtarmatched_phi.pkl"))
    dRtruthtarmatch_tar_phi    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRtruthtarmatched_phi.pkl"))

    phi_save_folder = save_folder + "/phi_truth/"
    if not os.path.exists(phi_save_folder): os.makedirs(phi_save_folder)

    # phi_bins = np.arange(-3.3,3.3,step=0.25)
    phi_bins = [-3.3,-3.1,-2.9,-2.7,-2.5,-2.3,-2.1,-1.9,-1.7,-1.5,-1.3,-1.1,-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3]
    phi_bin_centers = phi_bins[:-1] + 0.5 * np.diff(phi_bins)
    phi_bin_width = np.diff(phi_bins)

    average_response, std_response = list(), list()
    fitted_mu, fitted_mu_unc = list(), list()
    fitted_sigma, fitted_sigma_unc = list(), list()
    for bin_idx in range(len(phi_bins)-1):
        bin_mask = (phi_bins[bin_idx]<total_dRtruthmatch_tru_phi) & (total_dRtruthmatch_tru_phi<phi_bins[bin_idx+1])

        target_jet_pt_in_this_bin = total_dRtruthmatch_tru_pt[bin_mask]
        pred_jet_pt_in_this_bin = total_dRtruthmatch_p_pt[bin_mask]
        jet_pt_response_bin_i = pred_jet_pt_in_this_bin / target_jet_pt_in_this_bin
        print(f"Number of jets in bin {bin_idx}: {len(jet_pt_response_bin_i)}")
        print(f"phi in [{phi_bins[bin_idx]:.3f},{phi_bins[bin_idx+1]:.3f}], np.mean {np.mean(jet_pt_response_bin_i):.4f}")

        if len(jet_pt_response_bin_i)==0:
            average_response.append(0)
            std_response.append(0)
            fitted_mu.append(0)
            fitted_mu_unc.append(0)
            fitted_sigma.append(0)
            fitted_sigma_unc.append(0)
            continue

        jet_resp_hist, bins = np.histogram(jet_pt_response_bin_i, bins=100)
        bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        try:
            # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[1.0,1.0,0.1])
            popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[len(jet_pt_response_bin_i),1.0,0.1])
            # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, bounds=[(-np.inf,0.0,-np.inf),(np.inf, np.inf, np.inf)], maxfev=1200)
        except RuntimeError:
            popt_g = np.array([0,0,0])
            pcov_g = np.array([[0,0,0],[0,0,0],[0,0,0]])
        fit_mu = popt_g[1]
        fit_var = popt_g[2]
        fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
        fit_var_unc = np.sqrt(np.diag(pcov_g))[2]
        print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_var:.4f} ")
        print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_var_unc:.5f}")

        average_response.append(np.mean(jet_pt_response_bin_i))
        std_response.append(np.std(jet_pt_response_bin_i))
        fitted_mu.append(popt_g[1])
        fitted_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
        print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")
        V = popt_g[2]
        sigma_V = np.sqrt(np.diag(pcov_g))[2]

        fitted_sigma.append(np.sqrt(V))
        fitted_sigma_unc.append((sigma_V*np.sqrt(V)) / (2*V))
        print("---------------------------------------------------------")
        plt.figure()
        plt.stairs(jet_resp_hist, bins, fill=True, color='orange',alpha=0.5)
        plt.hist(jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='blue')
        x = np.linspace(0,jet_pt_response_bin_i.max(),100)
        plt.plot(x, gaussian(x, *popt_g), linewidth=2.5, label='Custom gausmyfunc')
        plt.xlabel('reco/target jet pt')
        plt.ylabel(f'jets in bin {bin_idx}')
        ax = plt.gca()
        text = (f'mu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.4f}  \n'
                f'std = {np.sqrt(V):.4f} +- {(sigma_V*np.sqrt(V)) / (2*V):.5f}\n'
                f'std / pT = {np.sqrt(V) / phi_bin_centers[bin_idx]:.6f}')
        ax.text(0.95, 0.95, f'Fit parameters:\n'+text, transform=ax.transAxes, va='top', ha='right')
        plt.title(f'Jet eta in [{phi_bins[bin_idx]:.3f},{phi_bins[bin_idx+1]:.3f}], {len(jet_pt_response_bin_i)} jets')
        plt.savefig(phi_save_folder + f'truth_pt_response_phi_bin_{bin_idx}.png')
        plt.close()
        print()

    print()
    print("\nNow comparing targets to truth jets in eta")

    average_tar_response, std_tar_response = list(), list()
    fitted_tar_mu, fitted_tar_mu_unc = list(), list()
    fitted_tar_sigma, fitted_tar_sigma_unc = list(), list()
    for bin_idx in range(len(phi_bins)-1):
        # find jets in each bin
        bin_mask = (phi_bins[bin_idx]<dRtruthtarmatch_tru_phi) & (dRtruthtarmatch_tru_phi<phi_bins[bin_idx+1])

        tru_jet_pt_in_this_bin = dRtruthtarmatch_tru_pt[bin_mask]
        tar_jet_pt_in_this_bin = dRtruthtarmatch_tar_pt[bin_mask]
        tar_jet_pt_response_bin_i = tar_jet_pt_in_this_bin / tru_jet_pt_in_this_bin
        print(f"Number of jets in bin {bin_idx}: {len(tar_jet_pt_response_bin_i)}")
        print(f"phi in [{phi_bins[bin_idx]:.3f},{phi_bins[bin_idx+1]:.3f}], np.mean {np.mean(tar_jet_pt_response_bin_i):.4f}")

        if len(tar_jet_pt_response_bin_i)==0:
            average_tar_response.append(0)
            std_tar_response.append(0)
            fitted_tar_mu.append(0)
            fitted_tar_mu_unc.append(0)
            fitted_tar_sigma.append(0)
            fitted_tar_sigma_unc.append(0)
            continue

        tar_jet_resp_hist, bins = np.histogram(tar_jet_pt_response_bin_i, bins=100)
        bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        # popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=tar_jet_resp_hist, p0=[1.0,1.0,0.1])
        popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=tar_jet_resp_hist, p0=[len(tar_jet_pt_response_bin_i),1.0,0.1])
        fit_mu = popt_g[1]
        fit_var = popt_g[2]
        fit_mu_unc = np.sqrt(np.diag(pcov_g))[1]
        fit_var_unc = np.sqrt(np.diag(pcov_g))[2]
        print(f"Fit parameters: A = {popt_g[0]:.4f}, mu = {fit_mu:.4f}, var = {fit_var:.4f} ")
        print(f"Fit error:        +- {np.sqrt(np.diag(pcov_g))[0]:.3f},    +- {fit_mu_unc:.5f},   +- {fit_var_unc:.5f}")

        average_tar_response.append(np.mean(tar_jet_pt_response_bin_i))
        std_tar_response.append(np.std(tar_jet_pt_response_bin_i))
        fitted_tar_mu.append(popt_g[1])
        fitted_tar_mu_unc.append(np.sqrt(np.diag(pcov_g))[1])
        print("Because the fit produces the variance we need the standard deviation. Propogate uncertainties!")
        V = popt_g[2]
        sigma_V = np.sqrt(np.diag(pcov_g))[2]

        fitted_tar_sigma.append(np.sqrt(V))
        fitted_tar_sigma_unc.append((sigma_V*np.sqrt(V)) / (2*V))
        print("---------------------------------------------------------")
        plt.figure()
        plt.stairs(tar_jet_resp_hist, bins, fill=True, color='gold',alpha=0.5)
        plt.hist(tar_jet_pt_response_bin_i,bins=50,alpha=0.6,histtype='step',color='mediumslateblue')
        x = np.linspace(0,tar_jet_pt_response_bin_i.max(),100)
        plt.plot(x, gaussian(x, *popt_g), linewidth=2.5)
        plt.xlabel('reco/target jet pt')
        plt.ylabel(f'jets in bin {bin_idx}')
        # plt.yscale('log')
        ax = plt.gca()
        text = (f'mu = {popt_g[1]:.4f} +- {np.sqrt(np.diag(pcov_g))[1]:.4f}  \n'
                f'std = {np.sqrt(V):.4f} +- {(sigma_V*np.sqrt(V)) / (2*V):.5f}\n'
                f'std / pT = {np.sqrt(V) / phi_bin_centers[bin_idx]:.6f}')
        ax.text(0.95, 0.95, f'Fit parameters:\n'+text, transform=ax.transAxes, va='top', ha='right')
        plt.title(f'Jet pT in [{phi_bins[bin_idx]},{phi_bins[bin_idx+1]}]')
        plt.savefig(phi_save_folder + f'tar_truth_pt_response_phi_bin_{bin_idx}.png')
        plt.close()
        




    plt.figure()
    plt.errorbar(phi_bin_centers, fitted_tar_mu, xerr=phi_bin_width/2, yerr=fitted_tar_mu_unc, fmt='o', capsize=5, color='green',label='AKT4EMTopo jets')
    plt.errorbar(phi_bin_centers, fitted_mu, xerr=phi_bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlim((-2.5,2.5))
    plt.ylim((0.7,1.0))
    plt.xlabel(r'Truth Jet $\eta$')
    plt.ylabel('Jet Energy Response')
    plt.legend()
    plt.text(-2.15,0.96, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(-1.4,0.96, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(-2.15,0.945, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(-2.15,0.932, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(phi_save_folder+'comp_truth_jet_response_phi.png')
   
    plt.figure()
    plt.errorbar(phi_bin_centers, fitted_tar_mu, xerr=phi_bin_width/2, yerr=fitted_tar_mu_unc, fmt='o', capsize=5, color='green',label='AKT4EMTopo jets')
    plt.errorbar(phi_bin_centers, fitted_mu, xerr=phi_bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    plt.errorbar(phi_bin_centers, average_response, xerr=phi_bin_width/2, yerr=0.01*np.ones_like(average_response), fmt='o', capsize=5, color='blue',label='CNN np.mean')
    plt.errorbar(phi_bin_centers, average_tar_response, xerr=phi_bin_width/2, yerr=0.01*np.ones_like(average_tar_response), fmt='o', capsize=5, color='pink',label='AKT np.mean')
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
    plt.xlim((-2.5,2.5))
    plt.xlabel(r'Truth Jet $\eta$')
    plt.ylabel('Jet Energy Response')
    plt.legend()
    plt.savefig(phi_save_folder+'comp_truth_jet_response_phi2.png')

    plt.figure()
    plt.errorbar(phi_bin_centers, fitted_tar_sigma, xerr=phi_bin_width/2, yerr=fitted_tar_sigma_unc, fmt='o', capsize=5, color='green',label='AKT4EMTopo jets')
    plt.errorbar(phi_bin_centers, fitted_sigma, xerr=phi_bin_width/2, yerr=fitted_sigma_unc, fmt='o', capsize=5, color='red',label='CNN jets')
    # plt.errorbar(eta_bin_centers, std_response, xerr=eta_bin_width/2, yerr=0.01*np.ones_like(std_response), fmt='o', capsize=5, color='pink',label='CNN jets (np.std)')
    plt.xlim((-2.5,2.5))
    plt.ylim((0.0,0.3))
    plt.xlabel(r'Truth Jet $\eta$')
    plt.ylabel('Jet Energy Resolution')
    plt.legend()
    plt.text(-2.15,0.27, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
    plt.text(-1.4,0.27, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
    plt.text(-2.15,0.256, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=12)
    plt.text(-2.15,0.244, r"MC21 Dijet JZ1-4, $p_T > 20\,$GeV",fontfamily='sans-serif',fontsize=12)
    plt.savefig(phi_save_folder+'comp_truth_jet_resolution_phi.png')

