import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import sys
import os

import torch
import torchvision

import h5py
import json
import scipy

sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')
from utils.utils import remove_nan, phi_mod2pi, clip_boxes_to_image, merge_rectangles




def get_cluster_bounding_boxes(cluster_data,cells,event_no,extent):
    clusters = cluster_data[event_no][['cl_E_em', 'cl_E_had', 'cl_cell_n', 'cl_eta', 'cl_phi', 'cl_pt', 'cl_time']]
    # clusters = remove_nan(clusters) #remove nan padding
    cluster_cells = cluster_cell_data[event_no]
    mask1 = (clusters['cl_E_em']+clusters['cl_E_had']) > 5000 #5GeV cut
    filtered_clusters = clusters[mask1]
    filtered_cluster_cells = cluster_cells[mask1]
    cells_in_event = cells[event_no] #new line

    #loop over filtered clusters in this event 
    box_list = []
    for cluster_no in range(len(filtered_clusters)):
        cluster_cells_this_cluster = remove_nan(filtered_cluster_cells[cluster_no]) #remove nan padding on cl_cells
        cell_ids = cluster_cells_this_cluster['cl_cell_IdCells']
        find_cells_mask = np.isin(cells_in_event['cell_IdCells'],cell_ids)
        desired_cells = cells_in_event[find_cells_mask]

        cell_etas = desired_cells['cell_eta']
        cell_phis = desired_cells['cell_phi']
        cell_phis_wrap = phi_mod2pi(cell_phis) #

        xmin,ymin = min(cell_etas),min(cell_phis)
        width,height = max(cell_etas)-xmin, max(cell_phis)-ymin

        #if the cells "wrap around", some lie at the top of the image, some at the bottom
        if height > 5.0:
            # Create two boxes assoc. to this cluster, one >0, one <0
            bottom_of_top_box = min(cell_phis[cell_phis>0])
            top_of_top_box = max(cell_phis_wrap)
            height_of_top_box = top_of_top_box - bottom_of_top_box

            top_of_bottom_box = max(cell_phis[cell_phis<=0])
            bottom_of_bottom_box = min(cell_phis_wrap)
            height_of_bottom_box = top_of_bottom_box - bottom_of_bottom_box
            # assert abs(height_of_top_box-height_of_bottom_box)<1e-10, "Heights should be the same! {}, {}".format(height_of_top_box,height_of_bottom_box)
            #add both to list
            box_list.append([xmin,bottom_of_top_box,width,height_of_top_box])
            box_list.append([xmin,bottom_of_top_box,width,height_of_bottom_box])
        else:
            box_list.append([xmin,ymin,width,height])

            # Since we repeat a large chunk of normal calorimeter we need to compensatingly repeat these boxes as well
            bottom_of_new_box = min(cell_phis_wrap) 
            top_of_new_box = max(cell_phis_wrap)
            box_list.append([xmin,bottom_of_new_box,width,height])

    tensor_of_boxes = torch.tensor(box_list)
    clipped_boxes = clip_boxes_to_image(tensor_of_boxes,extent) # custom from detectron + xywh->xyxy->xywh
    merged_boxes = merge_rectangles(clipped_boxes.tolist()) #merge overlapping/contained boxes, taking the union

    return np.array(merged_boxes)






def make_gif(H_tensor,boxes,zlabel):

    fig = plt.figure()
    ax = plt.axes()
    im=plt.imshow(H_tensor)
    cbar = fig.colorbar(im,ax=ax,cmap='binary_r')
    cbar.set_label(zlabel, rotation=90)
    for bbx in boxes:
        bb = matplotlib.patches.Rectangle((bbx[0],bbx[1]),bbx[2],bbx[3],lw=1,ec='limegreen',fc='none')
        ax.add_patch(bb)
    tx = ax.set_title(f'IMSHOW RESPONSE')
    
    truncationthresholds = np.linspace(1,np.max(H_tensor),num=300)
    # initialization function: plot the background of each frame
    def init():
        im.set_data(H_tensor)
        return [im]

    def animate(i):
        # arr = im.get_array()
        arr = H_tensor.copy()
        arr = np.where(arr>truncationthresholds[i],truncationthresholds[i],arr)
        vmax = np.max(arr)
        vmin = np.min(arr)
        print(i,truncationthresholds[i], (vmin,vmax))
        im.set_data(arr)
        im.set_cmap('binary_r')
        im.set_clim(vmin, vmax)
        tx.set_text(f'Truncation threshold: {truncationthresholds[i]:.0f}')
        return [im]

    anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init,
                                frames=300, interval=250, blit=True)
    anim.save(f'/home/users/b/bozianu/work/SSD/SSD/cached_plots/test/data_truncation/{zlabel}.gif')
    print(f'Saved gif {zlabel}')
    plt.close()





def examine_one_image(H_tensor,boxes_array):
    #code to plot the calorimeter + cluster bboxes 
    #boxes should be in x,y,w,h 
    print('Examining one image, then exiting.')
    f,ax = plt.subplots()
    ii = ax.imshow(H_tensor,cmap='binary_r')
    for bbx in boxes_array:
        bb = matplotlib.patches.Rectangle((bbx[0],bbx[1]),bbx[2],bbx[3],lw=1,ec='limegreen',fc='none')
        ax.add_patch(bb)
    cbar = f.colorbar(ii,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('Max cell significance', rotation=90)
    # ax.set(xlabel='eta',ylabel='phi')
    f.savefig(f'/home/users/b/bozianu/work/SSD/SSD/cached_plots/test/data_truncation/H_max.png')
    plt.close()









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
            with h5py.File(clusters_file,"r") as f:
                cl_data = f["caloCells"] 
                # event_data = cl_data["1d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]
                cluster_data = cl_data["2d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]
                cluster_cell_data = cl_data["3d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]
                print('\t\t',cluster_data.dtype)
            # with h5py.File(jets_file,"r") as f:
            #     j_data = f["caloCells"]
            #     # event_data = j_data["1d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]
            #     jet_data = j_data["2d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]

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
                H_tot_copy  = np.pad(H_tot, ((repeat_rows,repeat_rows),(0,0)),'wrap')
                
                H_energy_copy  = np.pad(H_energy, ((repeat_rows,repeat_rows),(0,0)),'wrap')
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
                H_max_copy  = H_max
                H_max = np.where(H_max < 5, 0, H_max) 
                H_max = np.where(H_max > 15, 15, H_max) 
                
                H_mean[np.isnan(H_mean)] = 0
                H_sigma[np.isnan(H_sigma)] = -1
                H_time[np.isnan(H_time)] = 0
                H_mean_copy = H_mean
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


                GT_cluster_boxes = get_cluster_bounding_boxes(cluster_data,cells,event_no,extent)
                #  we'll need to scale the boxes by num bins and x/y range
                GT_cluster_boxes[:,0] = (H_tot.shape[1]) * (GT_cluster_boxes[:,0]-extent[0])/(extent[1] - extent[0])
                GT_cluster_boxes[:,1] = (H_tot.shape[0]) * (GT_cluster_boxes[:,1]-extent[2])/(extent[3] - extent[2])
                GT_cluster_boxes[:,2] = (H_tot.shape[1]) * GT_cluster_boxes[:,2]/(extent[1] - extent[0])
                GT_cluster_boxes[:,3] = (H_tot.shape[0]) * GT_cluster_boxes[:,3]/(extent[3] - extent[2])



                #code to plot the calorimeter + cluster bboxes 
                #boxes should be in x,y,w,h 
                print('Examining one image, then exiting.')
                f,ax = plt.subplots()
                ii = ax.imshow(H_tot_copy,cmap='binary_r')
                for bbx in GT_cluster_boxes:
                    bb = matplotlib.patches.Rectangle((bbx[0],bbx[1]),bbx[2],bbx[3],lw=1,ec='limegreen',fc='none')
                    ax.add_patch(bb)
                cbar = f.colorbar(ii,ax=ax)
                cbar.ax.get_yaxis().labelpad = 10
                cbar.set_label('cell significance', rotation=90)
                ax.set(xlabel='eta',ylabel='phi')
                plt.close()

                examine_one_image(H_max,GT_cluster_boxes)
                examine_one_image(H_mean,GT_cluster_boxes)

                make_gif(H_tot_copy,GT_cluster_boxes,'Sum of cell significance')
                make_gif(H_max_copy,GT_cluster_boxes,'Max cell significance')
                make_gif(H_mean_copy,GT_cluster_boxes,'Mean cell significance')
                make_gif(H_energy_copy,GT_cluster_boxes,'Sum of cell energy')


                quit()


                global_counter += 1
            chunk_counter += 1




