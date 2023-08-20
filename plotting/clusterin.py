print('Starting physics metrics')
import numpy as np
from scipy import stats
import matplotlib
from matplotlib import pyplot as plt
import torch
import torchvision
import os
import sys

import h5py
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')
from utils.metrics import delta_n, n_unmatched_truth, n_unmatched_preds, centre_diffs, hw_diffs, area_covered
from utils.utils import wrap_check_NMS, wrap_check_truth, remove_nan, get_cells_from_boxes




event_no = 0
clusters_file = "/home/users/b/bozianu/work/data/pileup-JZ4/clusters/user.cantel.34126190._000001.topoclusterD3PD_mc16_JZ4W.r10788.h5"
with h5py.File(clusters_file,"r") as f:
    cl_data = f["caloCells"] 
    event_data = cl_data["1d"][event_no]
    cluster_data = cl_data["2d"][event_no]
    cluster_cell_data = cl_data["3d"][event_no]

file = "/home/users/b/bozianu/work/data/pileup-JZ4/cells/user.cantel.34126190._000001.calocellD3PD_mc16_JZ4W.r10788.h5"
with h5py.File(file,"r") as f:
    h5group = f["caloCells"]
    cells = h5group["2d"][event_no]

print(cells.dtype)
print(len(cells['cell_IdCells']))
# print(cells['cell_IdCells'])
all_ids = cells['cell_IdCells']
print(len(cells['cell_E']))
# print(cells['cell_E'])


# print(cluster_data['cl_eta'].tolist())
print((cluster_data['cl_E_em']+cluster_data['cl_E_had']).tolist())


quit()
# print(event_data.dtype)
# print('2d\n')
# print(cluster_data.dtype)
# print('3d\n')
# print(cluster_cell_data.dtype)

cl_no = 4
print()
print(cluster_data['cl_E_em'][cl_no])
print(cluster_data['cl_E_had'][cl_no])
print()
print(cluster_data['cl_cell_n'][cl_no])
print(cluster_data['cl_E_em'][cl_no]+cluster_data['cl_E_had'][cl_no])
print()

# print(cluster_cell_data[cl_no]['cl_cell_IdCells'])
selected_ids = cluster_cell_data[cl_no]['cl_cell_IdCells']
mask = np.isin(all_ids, selected_ids)
energies = cells['cell_E']
eta = cells['cell_eta']
phi = cells['cell_phi']
selected_energies = energies[mask]
selected_eta = eta[mask]
selected_phi = phi[mask]
print(len(selected_energies))
print(sum(selected_energies))
print()
print(cluster_data['cl_eta'][cl_no])
print(cluster_data['cl_phi'][cl_no])
print(np.mean(selected_eta))
print(np.mean(selected_phi))
