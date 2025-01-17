import torch
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt


import h5py
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='path to the cells .h5 directory',)
parser.add_argument('--output_dir', type=str, nargs='?', const='../cache/images/', help='path to the output directory')
args = parser.parse_args()




def examine_one_image(path):
    #code to plot the calorimeter + cluster bboxes 
    #boxes should be in x,y,w,h 
    print('Examining one image, then exiting.')
    loaded_tensor = torch.load(path)
    print('Tensor shape: ',loaded_tensor.shape)
    f,ax = plt.subplots()
    channel = 1
    ii = ax.imshow(loaded_tensor[channel],cmap='binary_r')
    cbar = f.colorbar(ii,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label(f'cell var. {channel}', rotation=90)
    ax.set(xlabel='eta',ylabel='phi')
    f.savefig(f'examine-{channel}.png')
    plt.close()
    quit()




if __name__=="__main__":

    annotation_dict = {}
    annotation_dict_jet = {}
    global_counter = 0
    file_nos = ["01","02","03","04","05","06","07","08","09"] + np.arange(10,20).tolist() + np.arange(21,26).tolist() # jz4
    # file_nos = [36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56] # ttbar
    for file_no in file_nos:
        print('Loading file {}/{}'.format(file_no,len(file_nos)))     
        cells_file = args.path + "/user.lbozianu.42443923._0000{}.calocellD3PD_mc21_14TeV_JZ4.r14365.h5".format(file_no) # JZ4
        # cells_file = args.path + "user.lbozianu.42443875._0000{}.calocellD3PD_mc21_14TeV_ttbar.r15583.h5".format(file_no) # ttbar


        chunk_size = 100
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
                cell_etas = cell_etas[abs(cell_etas)<2.5]

                #np.linspace(start, stop, int((stop - start) / step + 1))
                bins_x = np.linspace(min(cell_etas), max(cell_etas), int((max(cell_etas) - min(cell_etas)) / 0.1 + 1))
                bins_y = np.linspace(min(cell_phis), max(cell_phis), int((max(cell_phis) - min(cell_phis)) / ((2*np.pi)/64) + 1))


            #now we'll look at each event individually
            for event_no in range(len(events)):
                unique_file_chunk_event_no = "0"+str(file_no)+"-"+str(chunk_counter)+"-"+str(event_no)
                
                # filter cells
                cell_esig =  cells['cell_E'][event_no] / cells['cell_Sigma'][event_no]          
                signif_mask = 2.0
                twosigmask = abs(cell_esig) >= signif_mask
                cells2sig = cells[event_no][twosigmask]
                
                cell_etas   = cells2sig['cell_eta']
                centralmask = abs(cell_etas) < 2.5
                cellscentral = cells2sig[centralmask]

                cell_etas   = cellscentral['cell_eta']
                cell_phis   = cellscentral['cell_phi']
                cell_energy = cellscentral['cell_E']
                cell_pt     = cellscentral['cell_pt']
                cell_sigma  = cellscentral['cell_Sigma']    
                cell_time   = cellscentral['cell_TimeCells']   
                cell_Esig   = cell_energy / cell_sigma      

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
                one_box_height = (yedges[-1]-yedges[0])/H_sum_pt.shape[0]
                
                H_sum_pt        = np.pad(H_sum_pt, ((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_max_pt        = np.pad(H_max_pt, ((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_sum_signif    = np.pad(H_sum_signif, ((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_max_signif    = np.pad(H_max_signif, ((repeat_rows,repeat_rows),(0,0)),'wrap')
                H_max_noise     = np.pad(H_max_noise, ((repeat_rows,repeat_rows),(0,0)),'wrap')
   
                # Replace NaNs
                H_sum_pt[np.isnan(H_sum_pt)] = 0
                H_max_pt[np.isnan(H_max_pt)] = 0
                H_sum_signif[np.isnan(H_sum_signif)] = 0
                H_max_signif[np.isnan(H_max_signif)] = 0
                H_max_noise[np.isnan(H_max_noise)] = -1
     
                extent = (xedges[0],xedges[-1],yedges[0]-(repeat_rows*one_box_height),yedges[-1]+(repeat_rows*one_box_height)) 

                print('\t\tSaving image {}, id: {}...'.format(global_counter,unique_file_chunk_event_no))
                H_layers = np.stack([H_sum_pt,
                                     H_max_pt,
                                     H_sum_signif,
                                     H_max_signif,
                                     H_max_noise],axis=0)
                H_layers_tensor = torch.tensor(H_layers)
                torch.save(H_layers_tensor,args.output_dir+"cell-img-{}.pt".format(unique_file_chunk_event_no))

                # examine_one_image(args.output_dir+"cell-img-{}.pt".format(unique_file_chunk_event_no))

                global_counter += 1
            chunk_counter += 1



