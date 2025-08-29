import torch
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt


import h5py
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='path to the cells .h5 directory',)
args = parser.parse_args()




if __name__=="__main__":

    annotation_dict = {}
    annotation_dict_jet = {}
    global_counter = 0

    JZ_file_dict = {
        # # training files
        # "JZ2" : ["000001","000002","000003","000004","000005","000006","000007","000008","000009","000010","000011","000012","000013","000014","000015","000016","000017","000018","000019","000020","000021","000022","000023","000024","000025","000026",],
        # "JZ3" : ["000011","000012","000015","000016","000017","000018","000019","000021","000022","000023","000025","000026","000027","000028","000029","000030","000031","000032","000033","000034","000035","000036","000037","000038","000039","000040"],
        # "JZ4"   : ["000014","000015","000016","000017","000018","000019","000020","000021","000022","000023","000049","000050","000051","000052","000053","000054","000055","000056","000057","000058","000059","000060","000061","000062","000063","000064","000065","000066","000067","000068","000069","000070","000071","000072","000073","000074","000075","000076","000077","000078","000079","000080","000081","000083","000084"]
        # # test files
        # "JZ2" : ["000027","000028","000029","000030","000041","000042","000043","000044","000045","000046","000047","000048","000049","000050"],
        "JZ2" : ["000027","000028","000029"],
        # "JZ3" : ["000041","000042","000043","000044","000045","000046","000047","000048","000049","000050","000051","000052","000053"],
        "JZ3" : ["000041","000042","000043"],
        # "JZ4" : ["000085","000086","000087","000090","000092","000093","000095","000097","000101","000103","000104","000105","000107","000108","000109","000110","000111","000112","000113","000114","000115","000116","000117"],
        "JZ4" : ["000085","000086","000087"],
    }

    JZ_grid_dict = {
        "JZ2" : "43186502",
        "JZ3" : "43186499",
        "JZ4" : "43589851",
    }

    eta_bins = np.arange(-2.5,2.525,step=0.025)
    eta_bin_centers = eta_bins[:-1] + 0.5 * np.diff(eta_bins)
    eta_bin_width = np.diff(eta_bins)
    bin_sums = np.zeros(len(eta_bin_centers))
    bin_counts = np.zeros(len(eta_bin_centers))

    for proc in ["JZ2", "JZ3", "JZ4"]:
    # for proc in ["JZ2"]:
        file_nos = JZ_file_dict[proc]
        for file_no in file_nos:
            print('Loading file {}/{}'.format(file_no,len(file_nos)))     
            cells_file = args.path + "{}/user.lbozianu/user.lbozianu.{}._{}.calocellD3PD_mc21_14TeV_{}.r14365.h5".format(proc, JZ_grid_dict[proc], file_no, proc) 

            with h5py.File(cells_file,"r") as f1:
                cl_data1 = f1["caloCells"] 
                n_events_in_file = len(cl_data1["2d"])

            chunk_size = 100
            chunk_counter = 0
            for i in range(int(n_events_in_file/chunk_size)):
                print('\tLoading chunk {}/{}'.format(chunk_counter,int(n_events_in_file/chunk_size)))
                with h5py.File(cells_file,"r") as f:
                    h5group = f["caloCells"]       
                    #convert to numpy arrays in chunk sizes
                    events = h5group["1d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]
                    cells = h5group["2d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]

                #now we'll look at each event individually
                for event_no in range(len(events)):
                    unique_file_chunk_event_no = "0"+str(file_no)+"-"+str(chunk_counter)+"-"+str(event_no)
                    # i want to know how many negative cells there are in the event
                    cell_E = cells['cell_E'][event_no] 
                    negatifmask = cell_E < 0
                    cells_negative = cells[event_no][negatifmask]
                    
                    # above 2 sigma
                    cell_esig =  cells_negative['cell_E'] / cells_negative['cell_Sigma']
                    twosigmask = abs(cell_esig) >= 2.0
                    cellsnegative2sig = cells_negative[twosigmask]
                    
                    # within eta < 2.5
                    cell_etas   = cellsnegative2sig['cell_eta']
                    centralmask = abs(cell_etas) < 2.5
                    cellsnegativecentral = cellsnegative2sig[centralmask]

                    # place negative cells in histogram
                    cell_etas   = cellsnegativecentral['cell_eta']
                    cell_eta_hist, bins = np.histogram(cell_etas, bins=eta_bins)
                    bin_sums += cell_eta_hist
                    # print(cell_eta_hist.shape,eta_bin_centers.shape,bin_sums.shape)
                    # plt.figure()
                    # plt.hist(cell_etas,bins=eta_bins,histtype='step')
                    # plt.stairs(cell_eta_hist, bins, fill=True, color='orange',alpha=0.5)
                    # plt.stairs(bin_sums-1, bins, fill=False, color='green',ls='--',alpha=0.7)
                    # plt.xlabel('eta')
                    # plt.savefig("negatif_cells_1.png")

                    global_counter += 1
                chunk_counter += 1

    print(f"Run over {global_counter} events")
    average_number_cells_in_each_bin_per_event = bin_sums / global_counter

    plt.figure(figsize=(8, 5))
    plt.plot(eta_bin_centers, average_number_cells_in_each_bin_per_event, marker='+', linestyle=None, color='blue')
    plt.title("Cells with |signif| > 2 (Eta Bin Centers)")
    plt.xlabel("Eta")
    plt.ylabel("Average Number of Negative Cells per Event")
    plt.tight_layout()
    plt.savefig("negatif_cells_eta.png")


