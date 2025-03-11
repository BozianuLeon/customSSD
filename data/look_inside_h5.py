# source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-dbg/setup.sh 
import h5py 
import numpy as np

# file_path = 'bigtopoClD3PD_mc21.h5'
# file_path = 'bigtopoClD3PD_mc21_ttbar.h5'

# with h5py.File(file_path, 'r') as h5_file:
#     # Recursive function to explore the HDF5 file
#     def explore_h5_group(group, path="/"):
#         for key in group.keys():
#             item = group[key]
#             item_path = f"{path}{key}"
#             if isinstance(item, h5py.Group):
#                 print(f"Group: {item_path}\n")
#                 explore_h5_group(item, item_path + "/")
#             elif isinstance(item, h5py.Dataset):
#                 print(f"Dataset: {item_path}")
#                 print(f" - Shape: {item.shape}")
#                 print(f" - Dtype: {item.dtype}\n")
    
#     # Start exploring from the root
#     explore_h5_group(h5_file)


#     # Now look at the first event
#     h5group = h5_file["caloCells"]
#     event_data = h5group["1d"][0]
#     cl_data = h5group["2d"][0]
#     cl_cell_data = h5group["3d"][0]

#     print("Number of clusters ",event_data["cl_n"])
#     print("Number of clusters (cl_pt) ",cl_data["cl_pt"].shape)
#     print("Number of clusters (cl_cell_E) ",cl_cell_data["cl_cell_E"].shape)

#     print()
#     print("Cluster 1 energy:  ", cl_data["cl_E_em"][0]+cl_data["cl_E_had"][0],\
#     "\tCluster 1 energy (cells):  ", np.nansum(cl_cell_data["cl_cell_E"][0]), sum(cl_cell_data["cl_cell_E"][0]))
#     print("Cluster 1 pt:  ", cl_data["cl_pt"][0],\
#     "\tCluster 1 pt (cells):  ", np.nansum(cl_cell_data["cl_cell_pt"][0]), sum(cl_cell_data["cl_cell_pt"][0]))
#     print("Recall that some topoclusters share cells due to topocluster splitting and weighting a cell's energy between different clusters.")
#     print()
#     print()
#     print()
#     print()

    # for i in range(cl_data["cl_pt"].shape[0]):
    #     print(f"Cluster {i} E", cl_data["cl_E_em"][i]+cl_data["cl_E_had"][i], "cl cell energy", np.nansum(cl_cell_data["cl_cell_E"][i]))
    #     print(f"Cluster {i} pt", cl_data["cl_pt"][i], "cl cell pt", np.nansum(cl_cell_data["cl_cell_pt"][i]), "\n")




# -r 'cl_n|cl_E_em|cl_E_had|cl_RecoStatus|cl_cell_n|cl_cellmaxfrac|cl_centerlambda|cl_centermag|cl_eng_bad_cells|/^cl_eta$/|cl_firstEdens|cl_isolation|cl_lateral|cl_longitudinal|cl_n_bad_cells|/^cl_phi$/|cl_pt|cl_secondR|cl_secondlambda|cl_time|cl_cell_BadCells|cl_cell_DetCells|cl_cell_E|cl_cell_GainCells|cl_cell_IdCells|cl_cell_QCells|cl_cell_TimeCells|cl_cell_eta|cl_cell_phi|cl_cell_pt|cl_cell_xCells|cl_cell_yCells|cl_cell_zCells|eventNumber|mc_event_weight|timestamp|timestamp_ns|RunNumber|bcid|timestamp|timestamp_ns|bcid|lbn|actualIntPerXing|averageIntPerXing'
# -r 'cl_*|EventNumber|mc_event_weight|timestamp|timestamp_ns|RunNumber|bcid|timestamp|timestamp_ns|bcid|lbn|actualIntPerXing|averageIntPerXing'




# jet_file_path = 'bigjetClD3PD_mc21_ttbar.h5'
cells_file_path = "/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/cells/ttbar/user.lbozianu/user.lbozianu.42998650._000100.calocellD3PD_mc21_14TeV_ttbar.r15583.h5"
jet_file_path = "/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/jets/ttbar/user.lbozianu/user.lbozianu.42998650._000100.jetD3PD_mc21_14TeV_ttbar.r15583.h5"

with h5py.File(jet_file_path, 'r') as jet_h5_file:
    # Recursive function to explore the HDF5 file
    def explore_h5_group(group, path="/"):
        for key in group.keys():
            item = group[key]
            item_path = f"{path}{key}"
            if isinstance(item, h5py.Group):
                print(f"Group: {item_path}\n")
                explore_h5_group(item, item_path + "/")
            elif isinstance(item, h5py.Dataset):
                print(f"Dataset: {item_path}")
                print(f" - Shape: {item.shape}")
                print(f" - Dtype: {item.dtype}\n")
    
    # Start exploring from the root
    explore_h5_group(jet_h5_file)


    # Now look at the first event
    h5group = jet_h5_file["caloCells"]
    event_data = h5group["1d"][0]
    jet_data = h5group["2d"][0]

    print("Number of AntiKt4EMTopoJets ",event_data["AntiKt4EMTopoJets_n"], "Number of AntiKt4TruthJets ",event_data["AntiKt4TruthJets_n"])
    print("Number of AntiKt4EMTopoJets",jet_data["AntiKt4EMTopoJets_pt"].shape, "JetConstitScaleMomentum jets",jet_data["AntiKt4EMTopoJets_JetConstitScaleMomentum_pt"].shape, "JetEMScaleMomentum", jet_data["AntiKt4EMTopoJets_JetEMScaleMomentum_pt"].shape, "TruthJets", jet_data["AntiKt4TruthJets_pt"].shape)
    print("Number of not nan values: ", np.count_nonzero(~np.isnan(jet_data["AntiKt4EMTopoJets_pt"])), np.count_nonzero(~np.isnan(jet_data["AntiKt4EMTopoJets_JetConstitScaleMomentum_pt"])), np.count_nonzero(~np.isnan(jet_data["AntiKt4EMTopoJets_JetEMScaleMomentum_pt"])), np.count_nonzero(~np.isnan(jet_data["AntiKt4TruthJets_pt"])))


# |AntiKt4EMTopoJets_AverageLArQF|AntiKt4EMTopoJets_BCH_CORR_CELL|AntiKt4EMTopoJets_BCH_CORR_DOTX|AntiKt4EMTopoJets_BCH_CORR_JET|AntiKt4EMTopoJets_BCH_CORR_JET_FORCELL|AntiKt4EMTopoJets_E|AntiKt4EMTopoJets_HECQuality|AntiKt4EMTopoJets_JetConstitScaleMomentum_eta|AntiKt4EMTopoJets_JetConstitScaleMomentum_m|AntiKt4EMTopoJets_JetConstitScaleMomentum_phi|AntiKt4EMTopoJets_JetConstitScaleMomentum_pt|AntiKt4EMTopoJets_JetEMScaleMomentum_eta|AntiKt4EMTopoJets_JetEMScaleMomentum_m|AntiKt4EMTopoJets_JetEMScaleMomentum_phi|AntiKt4EMTopoJets_JetEMScaleMomentum_pt|AntiKt4EMTopoJets_LArQuality|AntiKt4EMTopoJets_NegativeE|AntiKt4EMTopoJets_NumTowers|AntiKt4EMTopoJets_OriginIndex|AntiKt4EMTopoJets_Timing|AntiKt4EMTopoJets_eta|AntiKt4EMTopoJets_hecf|AntiKt4EMTopoJets_isBadLoose|AntiKt4EMTopoJets_isBadMedium|AntiKt4EMTopoJets_isBadTight|AntiKt4EMTopoJets_isUgly|AntiKt4EMTopoJets_m|AntiKt4EMTopoJets_n90|AntiKt4EMTopoJets_ootFracCells10|AntiKt4EMTopoJets_ootFracCells5|AntiKt4EMTopoJets_ootFracClusters10|AntiKt4EMTopoJets_ootFracClusters5|AntiKt4EMTopoJets_phi|AntiKt4EMTopoJets_pt|AntiKt4TruthJets_E|AntiKt4TruthJets_eta|AntiKt4TruthJets_m|AntiKt4TruthJets_n90|AntiKt4TruthJets_phi|AntiKt4TruthJets_pt|
# [_000022,_000023,_000026,_000027,_000028,_000030,_000031,_000032,_000033,_000034,_000035,_000036,_000037,_000038,_000039,_000040,_000041,_000042,_000043,_000044,_000045,_000046,_000047,_000048,_000049,_000050,_000051,_000052,_000053,_000054,_000055,_000056,_000057,_000058,_000059,_000060,_000061,_000062,_000063,_000064,_000065,_000066,_000067,_000068,_000069,_000070,_000071,_000072,_000073,_000074,_000075,_000076,_000077,_000078,_000079,_000080,_000081,_000082,_000083,_000084,_000085,_000086,_000087,_000088,_000089,_000090,_000091,_000092,_000093,_000094,_000095,_000096,_000097,_000098,_000099,_000100]
