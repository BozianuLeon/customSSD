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


model_name = "jetSSD_smallconvnext_central_32e"
# open metrics folder and prepare figure directory
metrics_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/cache/{model_name}/ttbar/20250124-12/box_metrics"
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/ttbar/20250124-12/jet_res/"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)


# load in 
print("==================================================================================================")
print(f"Loading matched jets from\n{metrics_folder}")
print("==================================================================================================\n")
total_matched_tru_pt = np.concatenate(load_object(metrics_folder+"/tboxes_matched_pt.pkl"))
total_matched_pred_pt = np.concatenate(load_object(metrics_folder+"/pboxes_matched_pt.pkl"))
total_matched_tru_eta = np.concatenate(load_object(metrics_folder+"/tboxes_matched_eta.pkl"))
total_matched_pred_eta = np.concatenate(load_object(metrics_folder+"/pboxes_matched_eta.pkl"))




# homemade Gaussian to fit
def gaussian(x, a, mean, variance):
    return a * np.exp(-((x - mean)**2 / (2 * variance)))







bin_edges = [30, 40, 55, 80, 110, 150, 200, 300, 400, 550, 800] #[20,35,50,75,100,125,175,225,300,400,500,600]
bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
bin_width = np.diff(bin_edges)


average_response, std_response = list(), list()
fitted_mu, fitted_mu_unc = list(), list()
fitted_sigma, fitted_sigma_unc = list(), list()
for bin_idx in range(len(bin_edges)-1):
    # find jets in each bin
    bin_mask = (bin_edges[bin_idx]<total_matched_tru_pt) & (total_matched_tru_pt<bin_edges[bin_idx+1])

    target_jet_pt_in_this_bin = total_matched_tru_pt[bin_mask]
    pred_jet_pt_in_this_bin = total_matched_pred_pt[bin_mask]
    jet_pt_response_bin_i = pred_jet_pt_in_this_bin / target_jet_pt_in_this_bin
    print(f"Number of jets in bin {bin_idx}: {len(jet_pt_response_bin_i)}")
    print(f"pT in [{bin_edges[bin_idx],bin_edges[bin_idx+1]}], np.mean {np.mean(jet_pt_response_bin_i):.4f}")
    
    jet_resp_hist, bins = np.histogram(jet_pt_response_bin_i, bins=50)
    bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[1.0,1.0,0.1])
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
    plt.savefig(save_folder + f'pt_response_bin_{bin_idx}.png')
    plt.close()
    
    print()



plt.figure()
plt.errorbar(bin_centers, average_response, xerr=bin_width/2, fmt='o', capsize=5, color='blue',label='Simple mean')
plt.errorbar(bin_centers, fitted_mu, xerr=bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5, color='orange',label='Fit param')
plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
plt.xlabel('AntiKt4EMTopo Jet pT (jet constituent scale)')
plt.ylabel('Jet Energy Response')
plt.legend()
plt.savefig(save_folder+'jet_response_simple.png')

plt.figure()
# plt.errorbar(bin_centers, std_response, xerr=bin_width/2, fmt='o', capsize=5, color='blue',label='Simple np.std')
plt.errorbar(bin_centers, fitted_sigma, xerr=bin_width/2, yerr=fitted_sigma_unc, fmt='o', capsize=5, color='orange',label='Fit param')
plt.errorbar(bin_centers, abs(np.array(fitted_sigma)), xerr=bin_width/2, yerr=fitted_sigma_unc,alpha=0.5, capsize=5, color='pink')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('AntiKt4EMTopo Jet pT (jet constituent scale)')
plt.ylabel('Jet Energy Resolution')
plt.legend()
plt.savefig(save_folder+'jet_resolution_simple.png')

plt.figure()
plt.errorbar(bin_centers, np.array(fitted_sigma)/np.array(bin_centers), xerr=bin_width/2, yerr=fitted_sigma_unc, fmt='o', capsize=5, color='orange',label='Fit param')
ax = plt.gca()
ax.ticklabel_format(style='plain')
plt.xlabel('AntiKt4EMTopo Jet pT (jet constituent scale)')
plt.ylabel('Relative Jet Energy Resolution, sigma(pT) / pT')
plt.legend()
plt.savefig(save_folder+'jet_relative_resolution_simple.png')













print("\nNow binning in eta")






eta_bins = [-2.5,-1.8,-1.3,-1.0,-0.7,-0.2,0.0,0.2,0.7,1.0,1.3,1.8,2.5] #
eta_bin_centers = eta_bins[:-1] + 0.5 * np.diff(eta_bins)
eta_bin_width = np.diff(eta_bins)

average_response, std_response = list(), list()
fitted_mu, fitted_mu_unc = list(), list()
fitted_sigma, fitted_sigma_unc = list(), list()

for bin_idx in range(len(eta_bins)-1):
    bin_mask = (eta_bins[bin_idx]<total_matched_tru_eta) & (total_matched_tru_eta<eta_bins[bin_idx+1])

    target_jet_pt_in_this_bin = total_matched_tru_pt[bin_mask]
    pred_jet_pt_in_this_bin = total_matched_pred_pt[bin_mask]
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

    jet_resp_hist, bins = np.histogram(jet_pt_response_bin_i, bins=50)
    bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    # popt_g, pcov_g = scipy.optimize.curve_fit(gausmyfunc, xdata=bin_centres, ydata=jet_resp_hist, bounds=[(0.5,0.0, -np.inf),(np.inf, np.inf, np.inf)])
    popt_g, pcov_g = scipy.optimize.curve_fit(gaussian, xdata=bin_centres, ydata=jet_resp_hist, p0=[1.0,1.0,0.1])
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
    plt.title(f'Jet eta in [{eta_bins[bin_idx]},{eta_bins[bin_idx+1]}]')
    plt.savefig(save_folder + f'pt_response_eta_bin_{bin_idx}.png')
    plt.close()
    
    print()


plt.figure()
plt.errorbar(eta_bin_centers, average_response, xerr=eta_bin_width/2, fmt='o', capsize=5, color='blue',label='Simple mean')
plt.errorbar(eta_bin_centers, fitted_mu, xerr=eta_bin_width/2, yerr=fitted_mu_unc, fmt='o', capsize=5, color='orange',label='Fit param')
plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
plt.xlabel('AntiKt4EMTopo Jet eta (jet constituent scale)')
plt.ylabel('Jet Energy Response')
plt.legend()
plt.savefig(save_folder+'jet_response_eta.png')

plt.figure()
# plt.errorbar(eta_bin_centers, std_response, xerr=eta_bin_width/2, fmt='o', capsize=5, color='blue',label='Simple np.std')
plt.errorbar(eta_bin_centers, fitted_sigma, xerr=eta_bin_width/2, yerr=fitted_sigma_unc, fmt='o', capsize=5, color='orange',label='Fit param')
plt.errorbar(eta_bin_centers, abs(np.array(fitted_sigma)), xerr=eta_bin_width/2, yerr=fitted_sigma_unc,alpha=0.5, capsize=5, color='pink')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('AntiKt4EMTopo Jet eta (jet constituent scale)')
plt.ylabel('Jet Energy Resolution')
plt.legend()
plt.savefig(save_folder+'jet_resolution_eta.png')

# plt.figure()
# plt.errorbar(eta_bin_centers, np.array(fitted_sigma)/np.array(bin_centers), xerr=eta_bin_width/2, yerr=fitted_sigma_unc, fmt='o', capsize=5, color='orange',label='Fit param')
# ax = plt.gca()
# ax.ticklabel_format(style='plain')
# plt.xlabel('AntiKt4EMTopo Jet eta (jet constituent scale)')
# plt.ylabel('Relative Jet Energy Resolution, sigma(pT) / pT')
# plt.legend()
# plt.savefig(save_folder+'jet_relative_resolution_eta.png')


