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
save_folder = f"/home/users/b/bozianu/work/paperSSD/customSSD/plotting/figs/{model_name}/{proc}/{date}/jet_mult/"
if not os.path.exists(save_folder): os.makedirs(save_folder)
image_format = "png"


print("=======================================================================================================")
print(f"Loading all jets from\n{metrics_folder}")
print("=======================================================================================================\n")

event_tar_pt      = load_object(metrics_folder+"/tarboxes_pt.pkl")
event_tru_pt      = load_object(metrics_folder+"/truboxes_pt.pkl")
event_p_pt        = load_object(metrics_folder+"/pboxes_pt.pkl")
total_jet_weight  = np.concatenate(load_object(metrics_folder+"/jet_evt_weight.pkl"))
total_evt_weight  = load_object(metrics_folder+"/evt_weight.pkl")

total_tar_pt      = np.concatenate(event_tar_pt)
total_tru_pt      = np.concatenate(event_tru_pt)
total_p_pt        = np.concatenate(event_p_pt)
total_p_scr       = np.concatenate(load_object(metrics_folder+"/pboxes_scores.pkl"))
# dR matched
dRmatch_tar_pt    = np.concatenate(load_object(metrics_folder+"/tarboxes_dRmatched_pt.pkl"))
dRmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_pt.pkl"))
dRmatch_p_scr   = np.concatenate(load_object(metrics_folder+"/pboxes_dRmatched_scr.pkl"))
dRunmatch_tar_pt  = np.concatenate(load_object(metrics_folder+"/tarboxes_dRunmatched_pt.pkl"))
dRunmatch_p_pt  = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_pt.pkl"))
dRunmatch_p_scr = np.concatenate(load_object(metrics_folder+"/pboxes_dRunmatched_scr.pkl"))
# dR matched (truth)
dRtruthmatch_tru_pt    = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthmatched_pt.pkl"))
dRtruthmatch_p_pt    = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_pt.pkl"))
dRtruthmatch_p_scr   = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthmatched_scr.pkl"))
dRtruthunmatch_tru_pt  = np.concatenate(load_object(metrics_folder+"/truboxes_dRtruthunmatched_pt.pkl"))
dRtruthunmatch_p_pt  = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthunmatched_pt.pkl"))
dRtruthunmatch_p_scr = np.concatenate(load_object(metrics_folder+"/pboxes_dRtruthunmatched_scr.pkl"))


print(f"Unfortunately, the MC samples are all using different JZ slices")