import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ATLAS)

xs = [0, 32, 60, 200]
ys = [0.005867, 0.005891, 0.005806, 0.005976]
yerr = [0.0000226, 0.0000764, 0.0000521, 0.0000578]

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

plt.plot(x_interp, y_interp, label='Linear interpolation', 
         color='red', linewidth=2, linestyle='--')  # Dashed line for interpolation


plt.xlabel(r'Pile-up $< \mu >$')
plt.ylabel('Inferece speed [s]')
# plt.title('CaloJetSSD Execution Speeds')
plt.ylim(0,0.01)
plt.xlim(-10,210)
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
plt.text(0.68, 0.80, f'NVidia GeForce RTX 2080 Ti', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.text(0.68, 0.75, r'Incl. data CPU$\rightarrow$GPU$\rightarrow$CPU', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
# hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
# ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
plt.tight_layout()
plt.savefig(f"figs/time_interp.png",dpi=400)





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
plt.ylabel('Inferece speed [ms]')
plt.ylim(0,9)
plt.xlim(-10,210)
plt.legend(loc='lower left', 
            bbox_to_anchor=(0.66, 0.07), 
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
plt.text(0.68, 0.29, f'NVidia GeForce RTX 2080 Ti', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.text(0.68, 0.24, r'Incl. data CPU$\rightarrow$GPU$\rightarrow$CPU', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        fontsize=12)
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.gca().yaxis.get_major_formatter().set_scientific(False)
# hep.atlas.label(ax=ax0,label='Work in Progress',data=False,lumi=None,loc=1)
# ax0.set(yscale='log',xlabel='Jet $p_{\mathrm{T}}$ constituentScale [GeV]')
plt.tight_layout()
plt.savefig(f"figs/time_interp2.png",dpi=400)