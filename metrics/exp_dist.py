import numpy as np
import scipy
import matplotlib.pyplot as plt

# Parameters
size = 10000 
lambda_exp = 0.1 
mu_gaussian = 1  
sigma_gaussian = 0.2 

# Exponentially falling distribution
exp_distribution = np.random.exponential(scale=1/lambda_exp, size=size)

# Gaussian distribution and modify the exponential distribution
gaussian_noise = np.random.normal(loc=mu_gaussian, scale=sigma_gaussian, size=size)
modified_distribution = exp_distribution * abs(gaussian_noise)

# Fit using gamma dist.
x = np.linspace(0,modified_distribution.max(),100)
def myfunc(x, a, b, c):
    return a * np.exp(-b * x) + c
def myfunc2(x, a, b, c, d):
    return a*np.exp(-c*(x-b))+d
def gausmyfunc(x,a,b,c):
    return a*np.exp(-b*x**2)+c

gamma_dist = scipy.stats.gamma
exp_dist   = scipy.stats.expon
pareto_dist   = scipy.stats.genpareto
norm_dist   = scipy.stats.norm
gam_param = gamma_dist.fit(modified_distribution)
exp_param = exp_dist.fit(modified_distribution)
pareto_param = pareto_dist.fit(modified_distribution)
norm_param = norm_dist.fit(modified_distribution)

pdf_fitted_gam = gamma_dist.pdf(x,*gam_param)
pdf_fitted_exp = exp_dist.pdf(x,*exp_param)
pdf_fitted_pareto = pareto_dist.pdf(x,*pareto_param)
pdf_fitted_norm = norm_dist.pdf(x,*norm_param)


plt.figure(figsize=(10, 6))
bins = np.linspace(0,80,num=25)
bin_centres = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
mod_dist_hist, bins_ = np.histogram(modified_distribution, bins=bins)
popt, pcov = scipy.optimize.curve_fit(myfunc, xdata=bin_centres, ydata=mod_dist_hist, bounds=[(0.5, 0.0, 1),(np.inf, 10.0, np.inf)])
popt_2, pcov_2 = scipy.optimize.curve_fit(myfunc2, xdata=bin_centres, ydata=mod_dist_hist, bounds=[(0.5,0.0, 0.0, -np.inf),(np.inf, np.inf, np.inf, np.inf)])
popt_g, pcov_g = scipy.optimize.curve_fit(gausmyfunc, xdata=bin_centres, ydata=mod_dist_hist, bounds=[(0.5,0.0, -np.inf),(np.inf, np.inf, np.inf)])
print(popt)
print(popt_2)
print()
print(pcov)
print()
print(np.linalg.cond(pcov))
print("parameter variance (uncorrelated?)",np.diag(pcov))
print("fit error ",np.sqrt(np.diag(pcov)))

n, bins, _ = plt.hist(modified_distribution, bins=bins, histtype='step', alpha=0.6, label='Modified Distribution', color='red')
n, bins, _ = plt.hist(exp_distribution, bins=bins, histtype='step', alpha=0.6, label='Exponential Distribution', color='blue')
plt.plot(x, myfunc(x, *popt), linewidth=2.5, label='Custom myfunc')
plt.plot(x, myfunc2(x, *popt_2), linewidth=2.5, label='Custom myfunc2')
plt.plot(x, gausmyfunc(x, *popt_g), linewidth=2.5, label='Custom gausmyfunc')
plt.xlabel('pT')
plt.ylabel('# Jets')
plt.legend(loc='upper right')
plt.savefig('pt_distributions.png')

plt.figure(figsize=(10, 6))
bins = np.linspace(0,80,num=25)
n, bins, _ = plt.hist(modified_distribution, bins=bins,density=True, histtype='step', alpha=0.6, label='Modified Distribution', color='red')
n, bins, _ = plt.hist(exp_distribution, bins=bins,density=True, histtype='step', alpha=0.6, label='Exponential Distribution', color='blue')
plt.plot(x, pdf_fitted_gam,alpha=0.6,label='Simple gamma dist. fit')
plt.plot(x, pdf_fitted_exp,alpha=0.6,label='Simple expon dist. fit')
plt.plot(x, pdf_fitted_pareto,alpha=0.6,label='Simple genpareto dist. fit')
plt.plot(x, pdf_fitted_norm,alpha=0.6,label='Simple norm dist. fit')

plt.xlabel('pT')
plt.ylabel('# Jets')
plt.legend(loc='upper right')
plt.savefig('pt_distributions_fit.png')

quit()






# 4. Digitize, compare bin by bin
target_pt = exp_distribution
reco_pt = modified_distribution

target_inds = np.digitize(target_pt, bins)
reco_inds = np.digitize(reco_pt, bins)
print(len(bins),len(n))
print(min(target_inds),max(target_inds))
print(min(reco_inds),max(reco_inds))


for i in range(len(bins)):
    target_jets_bin_i = target_pt[np.where(target_inds==i)]
    reco_jets_bin_i = reco_pt[np.where(target_inds==i)]
    jet_pt_resolution_i = reco_jets_bin_i / target_jets_bin_i

    if len(jet_pt_resolution_i):
        mu, std = scipy.stats.norm.fit(jet_pt_resolution_i)
        x = np.linspace(min(jet_pt_resolution_i), max(jet_pt_resolution_i), 100)  
        p = scipy.stats.norm.pdf(x, mu, std) 
    else:
        x,p,mu,std=0,0,0,0

    plt.figure()
    plt.hist(jet_pt_resolution_i,density=True,bins=20,alpha=0.6,histtype='step',color='orange')
    plt.plot(x, p, 'k', linewidth=2, label=f'Fitted Gaussian\nmu = {mu:.2f}, std = {std:.2f}')
    plt.xlabel('reco/target jet pt')
    plt.ylabel(f'jets in bin {i}')
    plt.legend()
    plt.savefig(f'pt_reso_bin_{i}.png')
    plt.close()




