import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep

import pickle
def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)

hep.style.use(hep.style.ATLAS)


times_path_200  = "/home/users/b/bozianu/work/paperSSD/customSSD/cache/jetSSD_custom_convnext_central_32e/JZcomb0_test/20250707-14/time_per_event.pkl"
times_path_200b = "/home/users/b/bozianu/work/paperSSD/customSSD/cache/jetSSD_custom_convnext_central_32e/JZcomb0_test/20250707-14/time_only_inference.pkl"
times_200  = np.array(load_object(times_path_200))[1:]
times_200b = np.array(load_object(times_path_200b))[1:]
times_path_60  = "/home/users/b/bozianu/work/paperSSD/customSSD/cache/jetSSD_custom_convnext_central_32e/mu60/20250707-14/time_per_event.pkl"
times_path_60b = "/home/users/b/bozianu/work/paperSSD/customSSD/cache/jetSSD_custom_convnext_central_32e/mu60/20250707-14/time_only_inference.pkl"
times_60  = np.array(load_object(times_path_60))[1:]
times_60b = np.array(load_object(times_path_60b))[1:]
times_path_32  = "/home/users/b/bozianu/work/paperSSD/customSSD/cache/jetSSD_custom_convnext_central_32e/mu32/20250707-14/time_per_event.pkl"
times_path_32b = "/home/users/b/bozianu/work/paperSSD/customSSD/cache/jetSSD_custom_convnext_central_32e/mu32/20250707-14/time_only_inference.pkl"
times_32  = np.array(load_object(times_path_32))[1:]
times_32b = np.array(load_object(times_path_32b))[1:]
times_path_0  = "/home/users/b/bozianu/work/paperSSD/customSSD/cache/jetSSD_custom_convnext_central_32e/mu0/20250707-14/time_per_event.pkl"
times_path_0b = "/home/users/b/bozianu/work/paperSSD/customSSD/cache/jetSSD_custom_convnext_central_32e/mu0/20250707-14/time_only_inference.pkl"
times_0  = np.array(load_object(times_path_0))[1:]
times_0b = np.array(load_object(times_path_0b))[1:]
print(len(times_200),len(times_60),len(times_32),len(times_0))
print(len(times_200b),len(times_60b),len(times_32b),len(times_0b))
print(np.mean(times_200),np.mean(times_60),np.mean(times_32),np.mean(times_0))
print(np.mean(times_200b),np.mean(times_60b),np.mean(times_32b),np.mean(times_0b))
print(times_200[:10],times_60[:10],times_32[:10],times_0[:10])



xs = [0, 32, 60, 200]
ys = [np.mean(times_0),np.mean(times_32),np.mean(times_60),np.mean(times_200)]
yerr = [np.std(times_0),np.std(times_32),np.std(times_60),np.std(times_200)]
print('--',ys)
print('--',yerr)
# ys = [0.005867, 0.005891, 0.005806, 0.005976]
# yerr = [0.0000226, 0.0000764, 0.0000521, 0.0000578]

# Create a linear interpolation function
coefficients = np.polyfit(xs, ys, 1)  # 1st degree polynomial (linear)
poly_func = np.poly1d(coefficients)

# Generate x values for the interpolation line
x_interp = np.linspace(min(xs), max(xs), 100)
y_interp = poly_func(x_interp)

plt.figure(figsize=(10, 6))  # Set figure size
plt.errorbar(xs, ys, yerr=yerr, fmt='o', 
             label='MC Inference',
             color='slateblue', 
             ecolor='slateblue', 
             elinewidth=2, 
             capsize=5,  
             markersize=8,  
             markerfacecolor='darkblue', 
             markeredgecolor='black') 

plt.plot(x_interp, y_interp, label='Linear interpolation', color='red', linewidth=2, linestyle='--')  # Dashed line for interpolation
plt.xlabel(r'Pile-up $< \mu >$')
plt.ylabel('Inferece time [s]')
# plt.title('CaloJetSSD Execution Speeds')
plt.ylim(0,0.01)
plt.xlim(-5,210)
plt.legend(loc='lower left', 
            bbox_to_anchor=(0.66, 0.63), 
            fontsize=11)
total_params = 49610
plt.text(0.68, 0.91, f'CaloJetSSD', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontweight='bold',
        fontsize=16)
plt.text(0.68, 0.85, f'Learnable parameters: {total_params:.0f}', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.text(0.68, 0.80, f'NVIDIA GeForce RTX 3080', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.text(0.68, 0.75, r'Incl. data CPU$\rightarrow$GPU$\rightarrow$CPU', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)

plt.tight_layout()
plt.savefig(f"figs/time_interp.png",dpi=400)




plt.figure(figsize=(10, 6))  

# ROUGH!!!! Topocluster numbers
tc_x_axis = [28, 30, 31, 31, 31, 34, 36, 40, 42, 42, 46, 48, 48, 52, 54, 56, 57]
tc_y_axis = [41, 21, 41, 40, 39, 44, 43, 49, 49, 48, 52, 54, 55, 61, 65, 64, 65]

coeffs = np.polyfit(tc_x_axis, tc_y_axis, deg=2)
poly = np.poly1d(coeffs)
x_smooth = np.linspace(min(tc_x_axis), max(tc_x_axis), 300)
y_smooth = poly(x_smooth)
plt.plot(x_smooth, y_smooth, color='black', ls='--', label=f'LSQ Fit')
plt.scatter(tc_x_axis, tc_y_axis, s=25, c='black',label='CPU Topoclustering')


plt.errorbar(xs, np.array(ys)*1000, yerr=np.array(yerr)*1000, fmt='o', 
             label='CaloJetSSD Jet finding',
             color='slateblue', 
             ecolor='slateblue', 
             elinewidth=2, 
             capsize=5,  
             markersize=8,  
             markerfacecolor='dodgerblue',) 

plt.plot(x_interp, y_interp*1000, label='Linear interpolation', color='red', linewidth=2, linestyle='--')  # Dashed line for interpolation
plt.xlabel(r'Pile-up $< \mu >$')
plt.ylabel('Inferece time [ms]')
plt.ylim(0,80)
plt.xlim(-10,210)
plt.legend(loc='lower left', bbox_to_anchor=(0.66, 0.63), fontsize=11)
plt.tight_layout()
plt.savefig(f"figs/rough_time_comp.png",dpi=400)





plt.figure(figsize=(10, 6))  # Set figure size
plt.errorbar(xs, np.array(ys)*1000, yerr=np.array(yerr)*1000, fmt='o', 
             label='MC Inference',
             color='slateblue', 
             ecolor='slateblue', 
             elinewidth=2, 
             capsize=5,  
             markersize=8,  
             markerfacecolor='darkblue', 
             markeredgecolor='black') 

plt.plot(x_interp, y_interp*1000, label='Linear interpolation', 
         color='red', linewidth=2, linestyle='--')  # Dashed line for interpolation
plt.xlabel(r'Pile-up $< \mu >$')
plt.ylabel('Inferece time [ms]')
plt.ylim(0,6)
plt.xlim(-5,210)
plt.legend(loc='lower left', 
            bbox_to_anchor=(0.66, 0.02), 
            fontsize=11)
total_params = 49610
plt.text(0.68, 0.40, f'CaloJetSSD', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontweight='bold',
        fontsize=16)
plt.text(0.68, 0.34, f'Learnable parameters: {total_params:.0f}', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.text(0.68, 0.29, f'AMD EPYC 7742 64-Core CPU', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.text(0.68, 0.24, f'NVIDIA GeForce RTX 3080', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.text(0.68, 0.19, r'Incl. data CPU$\rightarrow$GPU$\rightarrow$CPU', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.gca().yaxis.get_major_formatter().set_scientific(False)

plt.text(-1,5.6, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
plt.text(22,5.6, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
plt.text(-1,5.3, f"MC dijet " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)

plt.tight_layout()
plt.savefig(f"figs/time_interp2.png",dpi=400)



n_jets_path_200 = "/home/users/b/bozianu/work/paperSSD/customSSD/cache/jetSSD_custom_convnext_central_32e/JZcomb0_test/20250627-15/n_jets_per_event.pkl"
n_pred_path_200 = "/home/users/b/bozianu/work/paperSSD/customSSD/cache/jetSSD_custom_convnext_central_32e/JZcomb0_test/20250627-15/n_pred_per_event.pkl"

times = load_object(times_path_200)
time2 = load_object(times_path_200b)
n_jets = load_object(n_jets_path_200)
n_pred = load_object(n_pred_path_200)

times = np.array(times[1:])
time2 = np.array(time2[1:])
n_jets = np.array(n_jets[1:])
n_pred = np.array(n_pred[1:])
print(times[:10])
print(n_jets[:10])
print(n_pred[:10])

binS = np.arange(max(n_jets),step=2)
bin_indices = np.digitize(n_jets, binS)  
bin_centers = []
time_means = []
time_stds = []
for i in range(1, len(binS)):
    # Mask values in current bin
    mask = bin_indices == i

    values_in_bin = times[mask]
    if values_in_bin.size > 0:
        bin_center = (binS[i] + binS[i-1]) / 2
        bin_centers.append(bin_center)
        time_means.append(np.mean(values_in_bin))
        time_stds.append(np.std(values_in_bin))
    else:
        bin_centers.append((binS[i] + binS[i-1]) / 2)
        time_means.append(np.nan)
        time_stds.append(np.nan)


print(bin_centers)
plt.figure(figsize=(10, 6))  # Set figure size
plt.errorbar(bin_centers, np.array(time_means)*1000, yerr=np.array(time_stds)*1000, fmt='o', 
             label='MC21 Inference',
             color='navy', 
             ecolor='navy', 
             elinewidth=2, 
             capsize=5,  
             markersize=8,  
             markerfacecolor='aqua', 
             markeredgecolor='black') 
plt.xlabel(r'AntiKt4EMTopo Jet multiplicity')
plt.ylabel('Inferece time [ms]')
plt.ylim(0,7.5)
plt.xlim(0,max(n_jets))
plt.legend(loc='lower left', 
            bbox_to_anchor=(0.66, 0.05), 
            fontsize=11)
total_params = 49610
plt.text(0.68, 0.40, f'CaloJetSSD', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontweight='bold',
        fontsize=16)
plt.text(0.68, 0.34, f'Learnable parameters: {total_params:.0f}', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.text(0.68, 0.29, f'AMD EPYC 7742 64-Core CPU', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.text(0.68, 0.24, f'NVIDIA GeForce RTX 3080', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.text(0.68, 0.19, r'Incl. data CPU$\rightarrow$GPU$\rightarrow$CPU', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.gca().yaxis.get_major_formatter().set_scientific(False)

plt.text(1.5,6.9, "ATLAS",fontfamily='sans-serif',fontsize=20,fontstyle='italic',fontweight='bold')
plt.text(7,6.9, "Simulation Internal",fontfamily='sans-serif',fontsize=14,fontstyle='italic')
plt.text(1.5,6.6, r"$\sqrt{s}=14\,$TeV $\mu=200$",fontsize=11)
plt.text(1.5,6.3, f"MC dijet " +  r"$p_T > 20\,$GeV, $|\eta|<2.1$",fontfamily='sans-serif',fontsize=12)

plt.tight_layout()
plt.savefig(f"figs/time_interp_per_jet.png",dpi=400)






# checking a number of 2024 runs, to get the timings of FS topoclustering and uncalibrated jet reco
times_fs_tc = [90.72, 82.37, 87.87, 91.68, 90.55, 91.02, 91.42, 89.52, 87.63, 85.98, 90.61, 94.79, 99.46, 99.46, 97.30, 97.53, 99.57, 101.2, 98.09, 97.47, 94.26, 99.90, 98.77, 84.98, 98.30, 97.28]

times_aktem = [11.20, 11.17, 11.02, 11.25, 11.17, 11.23, 11.23, 11.09, 10.93, 10.79, 11.17, 11.59, 11.90, 11.75, 11.75, 11.76, 11.92, 12.03, 11.79, 11.74, 11.56, 11.93, 11.84, 10.89, 11.87, 11.77]

rates_aktem = [3.0651e+04, 3.0589e+04, 3.0065e+04, 3.1322e+04, 3.1322e+04, 3.1322e+04, 3.0951e+04, 3.0693e+04, 3.1297e+04, 3.1021e+04, 3.2011e+04, 3.0587e+04, 3.3822e+04, 3.5320e+04, 3.5174e+04, 3.5056e+04, 3.5208e+04, 3.4929e+04, 3.5743e+04, 3.4926e+04, 3.5141e+04, 3.5794e+04, 3.5535e+04, 3.4984e+04, 3.2050e+04, 3.3242e+04, 3.4742e+04, 3.5153e+04]


plt.figure()
plt.hist(times_fs_tc, bins=10, edgecolor='black', color='aquamarine')
plt.text(0.95, 0.95, f'Mean: {np.mean(times_fs_tc):.6f}\nStd Dev: {np.std(times_fs_tc):.7f}', 
        horizontalalignment='right', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.xlabel('Time [ms]')
plt.title('Average FS topocluster execution time per event (avg over LB)')
plt.savefig(f"figs/time_per_event_fs_topo_avg.png",dpi=400)





plt.figure()
plt.hist(times_aktem, bins=10, edgecolor='black', color='navajowhite')
plt.text(0.95, 0.95, f'Mean: {np.mean(times_aktem):.6f}\nStd Dev: {np.std(times_aktem):.7f}', 
        horizontalalignment='right', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.xlabel('Time [ms]')
plt.title('Average jetrecalg_HLT_AntiKt4EMTopoJets_nojcalib execution time per event (avg over LB)')
plt.savefig(f"figs/time_per_event_AKT4EMTopo_avg.png",dpi=400)

print(f"Mean FS TC: {np.mean(times_fs_tc):.6f}+-{np.std(times_fs_tc):.7f}")
print(f"Mean AKTEM: {np.mean(times_aktem):.6f}+-{np.std(times_aktem):.7f}")
times_sum = np.array(times_fs_tc) + np.array(times_aktem)
print(f"Mean AKTEM: {np.mean(times_sum):.6f}+-{np.std(times_sum):.7f}")
print(np.mean(times_fs_tc)+np.mean(times_aktem),'+-', np.sqrt(np.std(times_fs_tc)**2 + np.std(times_aktem)**2))


plt.figure()
plt.hist(rates_aktem, bins=10, edgecolor='black', color='gray')
plt.text(0.95, 0.95, f'Mean: {np.mean(rates_aktem):.6f}\nStd Dev: {np.std(rates_aktem):.7f}', 
        horizontalalignment='right', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.xlabel('Rate [Hz]')
plt.savefig(f"figs/rates_per_event_AKT4EMTopo_avg.png",dpi=400)


