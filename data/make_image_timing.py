import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import sys

import torch
import torchvision

import h5py
import json
import scipy
import time


def examine_one_image(path):
    #code to plot the calorimeter + cluster bboxes 
    #boxes should be in x,y,w,h 
    print('Examining one image, then exiting.')
    loaded_tensor = torch.load(path)
    print('Tensor shape: ',loaded_tensor.shape)
    f,ax = plt.subplots()
    channel = 8
    ii = ax.imshow(loaded_tensor[channel],cmap='binary_r')
    # for bbx in boxes_array:
    #     bb = matplotlib.patches.Rectangle((bbx[0],bbx[1]),bbx[2],bbx[3],lw=1,ec='limegreen',fc='none')
    #     ax.add_patch(bb)
    cbar = f.colorbar(ii,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label(f'cell var. {channel}', rotation=90)
    ax.set(xlabel='eta',ylabel='phi')
    f.savefig(f'exam-{channel}.png')
    plt.close()
    quit()




if __name__=="__main__":

    EM_layers = [65,81,97,113,  #EM barrel
                257,273,289,305, #EM Endcap
                145,161, # IW EM
                2052] #EM FCAL

    HAD_layers = [2,514,1026,1538, #HEC layers
                4100,6148, #FCAL HAD
                65544, 73736,81928, #Tile barrel
                131080,139272,147464, #Tile endcap
                811016,278536,270344] #Tile gap

    annotation_dict = {}
    annotation_dict_jet = {}
    global_counter = 0
    time_per_event = list()
    time_per_event2 = list()
    file_nos = np.arange(14,18).tolist()
    for file_no in file_nos:
        file_start_time = time.perf_counter()
        print('Loading file {}/{}'.format(file_no,22))     

        cells_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/cells/JZ4/user.lbozianu/user.lbozianu.43589851._0000{}.calocellD3PD_mc21_14TeV_JZ4.r14365.h5".format(file_no)
        chunk_size = 50

        with h5py.File(cells_file,"r") as f1:
            cl_data1 = f1["caloCells"] 
            n_events_in_file = len(cl_data1["2d"])

        chunk_counter = 0
        for i in range(int(n_events_in_file/chunk_size)):
            print('\tLoading chunk {}/{}'.format(chunk_counter,int(n_events_in_file/chunk_size)))
            with h5py.File(cells_file,"r") as f:
                h5group = f["caloCells"]       
                #convert to numpy arrays in chun sizes
                events = h5group["1d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]
                cells = h5group["2d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]

                # define bins once here (using ALL cells)
                cell_etas = cells['cell_eta'][0]
                cell_phis = cells['cell_phi'][0]
                #np.linspace(start, stop, int((stop - start) / step + 1))
                bins_x = np.linspace(min(cell_etas), max(cell_etas), int((max(cell_etas) - min(cell_etas)) / 0.1 + 1))
                bins_y = np.linspace(min(cell_phis), max(cell_phis), int((max(cell_phis) - min(cell_phis)) / ((2*np.pi)/64) + 1))


            #now we'll look at each event individually
            for event_no in range(len(events)):
                start_time = time.perf_counter()
                # unique_file_chunk_event_no = "0"+str(file_no)+"-"+str(chunk_counter)+"-"+str(event_no)
                
                # new, now using only 2 sigma cells
                cell_esig =  cells['cell_E'][event_no] / cells['cell_Sigma'][event_no]          
                signif_mask = 2.0
                twosigmask = abs(cell_esig) >= signif_mask
                cells2sig = cells[event_no][twosigmask]
                start_time2 = time.perf_counter()

                cell_etas   = cells2sig['cell_eta']
                centralmask = abs(cell_etas) < 2.5
                cellscentral = cells2sig[centralmask]

                cell_etas   = cellscentral['cell_eta']
                cell_phis   = cellscentral['cell_phi'] 
                cell_energy = cellscentral['cell_E']
                cell_pt     = cellscentral['cell_pt']
                cell_sigma  = cellscentral['cell_Sigma']    
                cell_time   = cellscentral['cell_TimeCells']   
                cell_Esig   =  cell_energy / cell_sigma     

                # make 2d histograms
                H_sum_pt, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                                    values=cell_pt,
                                                                    bins=(bins_x,bins_y),
                                                                    statistic='sum')
              
                H_max_pt, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                                    values=cell_pt,
                                                                    bins=(bins_x,bins_y),
                                                                    statistic='max')

                H_sum_signif, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                                    values=cell_Esig,
                                                                    bins=(bins_x,bins_y),
                                                                    statistic='sum')

                H_max_signif, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                                    values=abs(cell_Esig),
                                                                    bins=(bins_x,bins_y),
                                                                    statistic='max')
              
                H_max_noise, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                                    values=cell_sigma,
                                                                    bins=(bins_x,bins_y),
                                                                    statistic='max')
              
                #transpose to correct format/shape
                H_sum_pt = H_sum_pt.T
                H_max_pt = H_max_pt.T
                H_sum_signif = H_sum_signif.T
                H_max_signif = H_max_signif.T
                H_max_noise = H_max_noise.T

                # Padding
                repeat_frac = 0.5
                repeat_rows = int(H_sum_pt.shape[0]*repeat_frac)
                # one_box_height = (yedges[-1]-yedges[0])/H_sum_pt.shape[0]
                H_sum_pt    = np.pad(H_sum_pt, ((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_max_pt    = np.pad(H_max_pt, ((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_sum_signif    = np.pad(H_sum_signif, ((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_max_signif    = np.pad(H_max_signif, ((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_max_noise    = np.pad(H_max_noise, ((repeat_rows,repeat_rows),(0,0)),'wrap')
   
                # NaNs
                H_sum_pt[np.isnan(H_sum_pt)] = 0
                H_max_pt[np.isnan(H_max_pt)] = 0
                H_sum_signif[np.isnan(H_sum_signif)] = 0
                H_max_signif[np.isnan(H_max_signif)] = 0
                H_max_noise[np.isnan(H_max_noise)] = -1
     
                # extent = (xedges[0],xedges[-1],yedges[0]-(repeat_rows*one_box_height),yedges[-1]+(repeat_rows*one_box_height)) 

                # Saving, now we save all H_* as a layer in one tensor
                # when we want to access only EM layers, just take that slice out of the sing .pt
                # print('\t\tSaving image {}, id: {}...'.format(global_counter,unique_file_chunk_event_no))
                # overall_save_path = "/home/users/b/bozianu/work/data/mu200/cell_images_no_trunc/"
                H_layers = np.stack([H_sum_pt,
                                     H_max_pt,
                                     H_sum_signif,
                                     H_max_signif,
                                     H_max_noise],axis=0)
                end_time = time.perf_counter()
                time_per_event.append(end_time-start_time)
                time_per_event2.append(end_time-start_time2)
                
                # H_layers_tensor = torch.tensor(H_layers)
                # torch.save(H_layers_tensor,overall_save_path+"cell-img-{}.pt".format(unique_file_chunk_event_no))
                # examine_one_image(overall_save_path+"cell-img-{}.pt".format(unique_file_chunk_event_no))

                global_counter += 1
            chunk_counter += 1
        file_end_time = time.perf_counter()
        print(f"Time taken for file {file_no}: {(file_end_time-file_start_time)/60:.3f} mins, (or {(file_end_time-file_start_time):.3f}s), average {(file_end_time-file_start_time)/n_events_in_file:.4f} per image over {n_events_in_file} events")


        plt.figure()
        mean = np.mean(time_per_event)
        std_dev = np.std(time_per_event)
        # bins = np.linspace(0, max(time_per_event), 16) 
        bins = np.linspace(np.percentile(time_per_event, 0), np.percentile(time_per_event, 90), 16) 
        plt.hist(time_per_event, bins=bins, edgecolor='black', color='blue')
        plt.xlabel('Time [s]')
        plt.title(f'Pre-processing time per event ({n_events_in_file} events)')
        plt.text(0.45, 0.95, f'Mean: {mean:.6f}\nStd Dev: {std_dev:.7f}', 
                horizontalalignment='right', 
                verticalalignment='top', 
                transform=plt.gca().transAxes,
                fontsize=12)

        plt.savefig(f"time_per_event.png",dpi=400)
        plt.close()

        plt.figure()
        mean = np.mean(time_per_event2)
        std_dev = np.std(time_per_event2)
        # bins = np.linspace(0, max(time_per_event2), 16) 
        bins = np.linspace(np.percentile(time_per_event2, 0), np.percentile(time_per_event2, 90), 16) 
        plt.hist(time_per_event2, bins=bins, edgecolor='black', color='red')
        plt.xlabel('Time2 [s]')
        plt.title(f'Pre-processing time per event ({n_events_in_file} events)')
        plt.text(0.45, 0.95, f'Mean: {mean:.6f}\nStd Dev: {std_dev:.7f}', 
                horizontalalignment='right', 
                verticalalignment='top', 
                transform=plt.gca().transAxes,
                fontsize=12)

        plt.savefig(f"time_per_event2.png",dpi=400)
        plt.close()


