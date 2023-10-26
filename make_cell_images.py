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

sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')
from utils.utils import remove_nan, phi_mod2pi, clip_boxes_to_image, merge_rectangles



def examine_one_image(path,boxes_array):
    #code to plot the calorimeter + cluster bboxes 
    #boxes should be in x,y,w,h 
    print('Examining one image, then exiting.')
    loaded_tensor = torch.load(path)
    f,ax = plt.subplots()
    ii = ax.imshow(loaded_tensor[3],cmap='binary_r')
    for bbx in boxes_array:
        bb = matplotlib.patches.Rectangle((bbx[0],bbx[1]),bbx[2],bbx[3],lw=1,ec='limegreen',fc='none')
        ax.add_patch(bb)
    cbar = f.colorbar(ii,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('cell significance', rotation=90)
    ax.set(xlabel='eta',ylabel='phi')
    f.savefig('examine-1.png')
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
    file_nos = ["01","02","03","04","05","06","07","08","09"] + np.arange(10,52).tolist()
    for file_no in file_nos[::-1]:
        print('Loading file {}/{}'.format(file_no,10))    

        cells_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/cells/user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(file_no)
        clusters_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/clusters/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(file_no)
        jets_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/jets/user.cantel.34126190._0000{}.jetD3PD_mc16_JZ4W.r10788.h5".format(file_no)
        chunk_size = 50

        with h5py.File(clusters_file,"r") as f1:
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

            #now we'll look at each event individually
            for event_no in range(len(events)):
                unique_file_chunk_event_no = "0"+str(file_no)+"-"+str(chunk_counter)+"-"+str(event_no)

                cell_etas = cells['cell_eta'][event_no]
                cell_phis = cells['cell_phi'][event_no] 
                cell_energy = cells['cell_E'][event_no]
                cell_sigma = cells['cell_Sigma'][event_no]    
                cell_time = cells['cell_TimeCells'][event_no]   
                cell_Esig =  cell_energy / cell_sigma        
                
                EM_indices = np.isin(cells['cell_DetCells'][event_no],EM_layers)
                HAD_indices = np.isin(cells['cell_DetCells'][event_no],HAD_layers)
                cell_etas_EM = cells['cell_eta'][event_no][EM_indices]
                cell_etas_HAD = cells['cell_eta'][event_no][HAD_indices]
                cell_phis_EM = cells['cell_phi'][event_no][EM_indices]
                cell_phis_HAD = cells['cell_phi'][event_no][HAD_indices]
                cell_E_EM = cells['cell_E'][event_no][EM_indices]
                cell_E_HAD = cells['cell_E'][event_no][HAD_indices]
                cell_Esig_EM = cell_E_EM / cells['cell_Sigma'][event_no][EM_indices]
                cell_Esig_HAD = cell_E_HAD / cells['cell_Sigma'][event_no][HAD_indices]

                #np.linspace(start, stop, int((stop - start) / step + 1))
                bins_x = np.linspace(min(cell_etas), max(cell_etas), int((max(cell_etas) - min(cell_etas)) / 0.1 + 1))
                bins_y = np.linspace(min(cell_phis), max(cell_phis), int((max(cell_phis) - min(cell_phis)) / ((2*np.pi)/64) + 1))

                H_tot, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                                    values=abs(cell_Esig),
                                                                    bins=(bins_x,bins_y),
                                                                    statistic='sum')

                H_em, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas_EM, cell_phis_EM,
                                                            values=abs(cell_Esig_EM),
                                                            bins=(bins_x,bins_y),
                                                            statistic='sum')

                H_had, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas_HAD, cell_phis_HAD,
                                                            values=abs(cell_Esig_HAD),
                                                            bins=(bins_x,bins_y),
                                                            statistic='sum')

                H_max, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                            values=cell_Esig,
                                                            bins=(bins_x,bins_y),
                                                            statistic='max')

                H_mean, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                            values=abs(cell_Esig),
                                                            bins=(bins_x,bins_y),
                                                            statistic='mean')

                H_sigma, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                            values=cell_sigma,
                                                            bins=(bins_x,bins_y),
                                                            statistic='mean')

                H_energy, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                            values=cell_energy,
                                                            bins=(bins_x,bins_y),
                                                            statistic='sum')

                H_time, _, _, _ = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                            values=cell_time,
                                                            bins=(bins_x,bins_y),
                                                            statistic='mean')

                #transpose to correct format/shape
                H_tot = H_tot.T
                H_em = H_em.T
                H_had = H_had.T
                H_max = H_max.T
                H_mean = H_mean.T
                H_sigma = H_sigma.T
                H_energy = H_energy.T
                H_time = H_time.T
            
                repeat_frac = 0.5
                repeat_rows = int(H_tot.shape[0]*repeat_frac)
                one_box_height = (yedges[-1]-yedges[0])/H_tot.shape[0]

                # Padding
                H_tot  = np.pad(H_tot, ((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_em   = np.pad(H_em,  ((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_had  = np.pad(H_had, ((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_max  = np.pad(H_max, ((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_mean = np.pad(H_mean,((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_sigma = np.pad(H_sigma,((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_energy = np.pad(H_energy,((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_time = np.pad(H_time,((repeat_rows,repeat_rows),(0,0)),'wrap')

                #If total cell signficance in a cell exceeds a threshold truncate 
                truncation_threshold = 125
                H_tot = np.where(H_tot < truncation_threshold, H_tot, truncation_threshold)
                H_em = np.where(H_em < truncation_threshold, H_em, truncation_threshold)
                H_had = np.where(H_had < truncation_threshold, H_had, truncation_threshold)

                #Max cell significance in a pixel
                H_max[np.isnan(H_max)] = 0
                H_max = np.where(H_max < 5, 0, H_max) 
                H_max = np.where(H_max > 15, 15, H_max) 
                
                H_mean[np.isnan(H_mean)] = 0
                H_sigma[np.isnan(H_sigma)] = -1
                H_time[np.isnan(H_time)] = 0

                #treat the barrel and forward regions differently
                raw_energy_thresh = 1000
                central_H_energy = np.where(H_energy[:,10:-10]>raw_energy_thresh,H_energy[:,10:-10],0)
                left_edge_H_energy = np.where(H_energy[:,:10]>4*raw_energy_thresh,H_energy[:,:10]/4,0)
                right_edge_H_energy = np.where(H_energy[:,-10:]>4*raw_energy_thresh,H_energy[:,-10:]/4,0)
                H_energy = np.hstack((left_edge_H_energy,central_H_energy,right_edge_H_energy))


                extent = (xedges[0],xedges[-1],yedges[0]-(repeat_rows*one_box_height),yedges[-1]+(repeat_rows*one_box_height)) 

                # Saving, now we save all H_* as a layer in one tensor
                # when we want to access only EM layers, just take that slice out of the sing .pt
                print('\t\tSaving image {}, id: {}, adding to dictionary...'.format(global_counter,unique_file_chunk_event_no))
                overall_save_path = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/cell_images/"
                H_layers = np.stack([H_tot,H_em,H_had,H_max,H_mean,H_sigma,H_energy,H_time],axis=0)
                H_layers_tensor = torch.tensor(H_layers)
                torch.save(H_layers_tensor,overall_save_path+"cell-image-tensor-{}.pt".format(unique_file_chunk_event_no))

                # examine_one_image(overall_save_path+"cell-image-tensor-{}.pt".format(unique_file_chunk_event_no),GT_cluster_boxes)

                global_counter += 1
            chunk_counter += 1




