import numpy as np 
import os
import scipy
import mplhep as hep

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.pyplot as plt
import matplotlib
plt.style.use(hep.style.ATLAS)

def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)

def transform_angle(angle):
    #maps angle to [-pi,pi]
    angle %= 2 * np.pi  # Map angle to [0, 2π]
    if angle >= np.pi:
        angle -= 2 * np.pi  # Map angle to [-π, π]
    return angle


def leading_jet_pt(list_of_jet_energies_in_event):
    return max(list_of_jet_energies_in_event)

def nth_leading_jet_pt(list_of_jet_energies_in_event,n):
    return sorted(list_of_jet_energies_in_event,reverse=True)[n-1]

    
model_name = "comp3_SSD_model_15_real"
save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_plots/" + model_name + "/"
if not os.path.exists(save_loc):
   os.makedirs(save_loc)

file_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp3_SSD_model_15_real/20230526-05/"

file_names = [['total_energy_esdjets','total_energy_fjets','total_energy_tboxjets','total_energy_pboxjets'],
              ['total_pt_esdjets','total_pt_fjets','total_pt_tboxjets','total_pt_pboxjets'],
              ['total_eta_esdjets','total_eta_fjets','total_eta_tboxjets','total_eta_pboxjets'],
              ['total_phi_esdjets','total_phi_fjets','total_phi_tboxjets','total_phi_pboxjets']]


#removing low pt FJets
pt_fjets = load_object(file_to_look_in + '/' + 'total_pt_fjets' + '.pkl')
highpt_indices = np.argwhere(np.concatenate(pt_fjets)>5000)





esdj_pt = load_object(file_to_look_in + '/total_pt_esdjets.pkl')
fjj_pt = load_object(file_to_look_in + '/total_pt_fjets.pkl')
esdj_lead_pt = [leading_jet_pt(x) for x in esdj_pt]
fjj_lead_pt = [leading_jet_pt(y) for y in fjj_pt]
trigger_decision = np.argwhere(np.array(esdj_lead_pt)>400_000).T[0]
print(trigger_decision)


f, axs = plt.subplots(2,1, figsize=(7, 10),sharex=True)
n, bins, _ = axs[0].hist(esdj_lead_pt, bins=50,histtype='step',label='ESD')
axs[0].hist(fjj_lead_pt, bins=bins, histtype='step',label='FJet')
bin_centers = (bins[:-1] + bins[1:]) / 2
axs[1].hist(np.array(esdj_lead_pt)[trigger_decision],bins=bins,histtype='step')
axs[1].hist(np.array(fjj_lead_pt)[trigger_decision],bins=bins,histtype='step')
axs[1].set(xlabel='Leading jet pT',ylabel='Frequency')
f.savefig('eff1.png')




def get_efficiency(success,totals):
    result = np.zeros_like(success, dtype=float)
    non_zero_indices = totals != 0
    
    # Perform element-wise division, handling zeros in the denominator
    result[non_zero_indices] = success[non_zero_indices] / totals[non_zero_indices]
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



np.random.seed(0)
size = 1000
offline_lead_jet_pt = np.random.exponential(scale=55., size=size)
online_lead_jet_pt = offline_lead_jet_pt + np.random.normal(loc=0.0,scale=10.0, size=size)
online_lead_jet_pt = np.clip(online_lead_jet_pt,0,None)

#choose trigger cut
lead_jet_cut = 50
trigger_decision = np.argwhere(np.array(offline_lead_jet_pt)>lead_jet_cut).T[0]

#make bins to use throughout
bin_width = 5
num_bins=20
bins = [i*bin_width for i in range(num_bins+1)]


f,ax = plt.subplots(3,1,figsize=(7.5,12))
n,bins,_ = ax[0].hist(offline_lead_jet_pt,bins=bins,histtype='step',label='Offline')
n2,bins2,_ = ax[0].hist(online_lead_jet_pt,bins=bins,histtype='step',label='Online')
ax[0].set(xlabel="Offline leading jet pT",ylabel='Freq.')
ax[0].legend()

n3,_,_ = ax[1].hist(offline_lead_jet_pt[trigger_decision],bins=bins,histtype='step',label='Offline')
n4,_,_ = ax[1].hist(online_lead_jet_pt[trigger_decision],bins=bins,histtype='step',label='Online')
ax[1].axvline(x=lead_jet_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
ax[1].set(xlabel="Offline leading jet pT",ylabel='Freq.')
ax[1].legend()

#calculate efficiencies and errors
step_eff = get_efficiency(n3,n)
step_err,ci = get_errorbars(n3,n)

efficiency = get_efficiency(n4,n2)
errors,ci = get_errorbars(n4,n2)

bin_centers = (bins[:-1] + bins[1:]) / 2
ax[2].errorbar(bin_centers,step_eff,xerr=bin_width/2,yerr=step_err,marker='.',ls='none',label='Offline')
ax[2].errorbar(bin_centers,efficiency,xerr=bin_width/2,yerr=errors,marker='.',ls='none',label='Online')
ax[2].axvline(x=lead_jet_cut,ymin=0,ymax=1,ls='--',color='red',alpha=0.3,label='Cut')
ax[2].set(xlabel="Offline leading jet pT",ylabel='Efficiency',ylim=(-0.2,1.2))
ax[2].legend(loc='lower right')

plt.tight_layout()
f.savefig('eff2atlas.png')
