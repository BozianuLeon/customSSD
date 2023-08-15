print('Starting. WARNING: THIS SCRIPT IS OLD AND NEEDS UPDATING')
import numpy as np
import torch
import sys
import time

import h5py
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')
from utils.metrics import delta_n, n_unmatched_truth, n_unmatched_preds, centre_diffs, hw_diffs, area_covered



# model_name = "comp3_SSD_model_15_real"
# save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/" + model_name + "/" + time.strftime("%Y%m%d-%H") + "/"
# save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp3_SSD_model_15_real/20230526-05/"
save_loc = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/comp3_SSD_model_15_real/20230526-05/"
path_to_structured_array = save_loc + "struc_array.npy"

with open(path_to_structured_array, 'rb') as f:
    a = np.load(f)

print(a.shape,a.dtype)
look_here = 2
h5f = a[look_here]['h5file']
event_no = a[look_here]['event_no']
print(h5f,event_no)

preds = a[look_here]['p_boxes']
tees = a[look_here]['t_boxes']
extent = a[look_here]['extent']
b = preds[np.where(preds[:,0] > 0)]

file = "/home/users/b/bozianu/work/data/real/cells/user.cantel.33075755._00000{}.calocellD3PD_mc16_JZW4.r14423.h5".format(h5f)
with h5py.File(file,"r") as f:
    h5group = f["caloCells"]

    #get column names
    print(h5group['1d'].dtype.descr)
    print()
    print(h5group['2d'].dtype.descr)
    print()
    
    #convert to numpy arrays in chun sizes
    event = h5group["1d"][event_no]
    cells = h5group["2d"][event_no]
    print(event)



total_delta_n, total_n_unmatch_truth, total_n_unmatch_pred = list(), list(), list()
total_centre_diff, total_h_diff, total_w_diff, total_area_cov = list(), list(), list(), list()
total_centre_diff_filter, total_h_diff_filter, total_w_diff_filter, total_area_cov_filter = list(), list(), list(), list()
total_centre_diff_filter_rad, total_h_diff_filter_rad, total_w_diff_filter_rad, total_area_cov_filter_rad = list(), list(), list(), list()
for i in range(len(a)):
    extent_i = a[i]['extent']
    preds = a[i]['p_boxes']
    trues = a[i]['t_boxes']
    pees = preds[np.where(preds[:,0] > 0)]
    tees = trues[np.where(trues[:,0] > 0)]

    pees = torch.tensor(pees)
    tees = torch.tensor(tees)

    #make boxes cover extent
    tees[:,(0,2)] = (tees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
    tees[:,(1,3)] = (tees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]

    pees[:,(0,2)] = (pees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
    pees[:,(1,3)] = (pees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]

    total_delta_n.append(delta_n(tees,pees))
    total_n_unmatch_truth.append(n_unmatched_truth(tees,pees))
    total_n_unmatch_pred.append(n_unmatched_preds(tees,pees))
    total_centre_diff.append(centre_diffs(tees,pees))
    total_h_diff.append(hw_diffs(tees,pees)[0])
    total_w_diff.append(hw_diffs(tees,pees)[1])
    total_area_cov.append(area_covered(tees,pees))

    total_centre_diff_filter.append(centre_diffs(tees,pees,filter=True))
    total_h_diff_filter.append(hw_diffs(tees,pees,filter=True)[0])
    total_w_diff_filter.append(hw_diffs(tees,pees,filter=True)[1])
    total_area_cov_filter.append(area_covered(tees,pees,filter=True))

    total_centre_diff_filter_rad.append(centre_diffs(tees,pees,filter="rad"))
    total_h_diff_filter_rad.append(hw_diffs(tees,pees,filter="rad")[0])
    total_w_diff_filter_rad.append(hw_diffs(tees,pees,filter="rad")[1])
    total_area_cov_filter_rad.append(area_covered(tees,pees,filter="rad"))



def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)


print('Saving the comparison metrics in lists...')
save_object(total_delta_n,save_loc+'total_delta_n.pkl')
save_object(total_n_unmatch_truth, save_loc+'total_n_unmatch_truth.pkl')
save_object(total_n_unmatch_pred, save_loc+'total_n_unmatch_pred.pkl')
save_object(total_centre_diff,save_loc+'total_centre_diff.pkl')
save_object(total_h_diff,save_loc+'total_h_diff.pkl')
save_object(total_w_diff,save_loc+'total_w_diff.pkl')
save_object(total_area_cov,save_loc+'total_area_cov.pkl')





save_object(total_centre_diff_filter,save_loc+'total_centre_diff_filter.pkl')
save_object(total_h_diff_filter,save_loc+'total_h_diff_filter.pkl')
save_object(total_w_diff_filter,save_loc+'total_w_diff_filter.pkl')
save_object(total_area_cov_filter,save_loc+'total_area_cov_filter.pkl')

save_object(total_centre_diff_filter,save_loc+'total_centre_diff_filter_rad.pkl')
save_object(total_h_diff_filter,save_loc+'total_h_diff_filter_rad.pkl')
save_object(total_w_diff_filter,save_loc+'total_w_diff_filter_rad.pkl')
save_object(total_area_cov_filter,save_loc+'total_area_cov_filter_rad.pkl')
















quit()
bins_x = np.linspace(min(cells['cell_eta']), max(cells['cell_eta']), int((max(cells['cell_eta']) - min(cells['cell_eta'])) / 0.1 + 1))
bins_y = np.linspace(min(cells['cell_phi']), max(cells['cell_phi']), int((max(cells['cell_phi']) - min(cells['cell_phi'])) / ((2*np.pi)/64) + 1))
H, xedges, yedges = np.histogram2d(cells['cell_eta'],cells['cell_phi'],bins=(bins_x,bins_y),
                                        weights=abs(cells['cell_E']/cells['cell_Sigma']))

    
H = H.T # Histogram does not follow Cartesian convention, therefore transpose H for visualization

repeat_frac = 0.2
repeat_rows = int(H.shape[0]*repeat_frac)
one_box_height = (yedges[-1]-yedges[0])/H.shape[0]
H = np.pad(H, ((repeat_rows,repeat_rows),(0,0)),'wrap')
print(H.shape)
truncation_threshold = 125
H = np.where(H < truncation_threshold, H, truncation_threshold)

tees[:,(0,2)] = (tees[:,(0,2)]*(extent[1]-extent[0]))+extent[0]
tees[:,(1,3)] = (tees[:,(1,3)]*(extent[3]-extent[2]))+extent[2]

preds[:,(0,2)] = (preds[:,(0,2)]*(extent[1]-extent[0]))+extent[0]
preds[:,(1,3)] = (preds[:,(1,3)]*(extent[3]-extent[2]))+extent[2]

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
im = plt.imshow(H,extent=extent,origin='lower',cmap='binary_r')
fig.colorbar(im,ax=ax,orientation='vertical')
for bbx in tees:
    x,y=float(bbx[0]),float(bbx[1])
    w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
    bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='limegreen',fc='none')
    ax.add_patch(bb)

for pbx in preds:
    x,y=float(pbx[0]),float(pbx[1])
    w,h=float(pbx[2])-float(pbx[0]),float(pbx[3])-float(pbx[1])  
    bbo = matplotlib.patches.Rectangle((x,y),w,h,lw=1,ec='red',fc='none')
    ax.add_patch(bbo)

ax.set(xlabel='eta',ylabel='phi',title='imshow weights == ESig, {} GT Boxes'.format(len(tees)))
fig.savefig('testing.png')     






