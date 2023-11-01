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


def get_jet_bounding_boxes(jet_data,event_no,extent,min_max_tuple):
    R = 0.4 # anti-kt 
    WIDTH,HEIGHT = 2*R, 2*R 
    MIN_PHI_VALUE = min_max_tuple[0]
    MAX_PHI_VALUE = min_max_tuple[1]


    jets = jet_data[event_no]
    real_jets = remove_nan(jets)
    #loop over all jets in this event 
    if len(real_jets) > 0:
        filtered_jets = real_jets[real_jets['AntiKt4EMTopoJets_pt'] > 20_000] #Select the pt threshold (in MeV)
        box_list = []
        for jet_no in range(len(filtered_jets)):
            jet_eta = filtered_jets['AntiKt4EMTopoJets_eta'][jet_no]
            jet_phi = filtered_jets['AntiKt4EMTopoJets_phi'][jet_no]
            xmin = jet_eta - R
            ymin = jet_phi - R

            # jet boxes that cross the discontinuity/wrap around
            if (jet_phi+R > MAX_PHI_VALUE - (extent[3]-MAX_PHI_VALUE)) or (jet_phi-R < MIN_PHI_VALUE - (extent[2]-MIN_PHI_VALUE)):
                wrapped_jet_phi = jet_phi - np.sign(jet_phi)*2*np.pi
                box_list.append([xmin,wrapped_jet_phi-R,WIDTH,HEIGHT])
                box_list.append([xmin,ymin,WIDTH,HEIGHT])

            else:
                box_list.append([xmin,ymin,WIDTH,HEIGHT])

        tensor_of_boxes = torch.tensor(box_list)
        clipped_boxes = clip_boxes_to_image(tensor_of_boxes,extent) # Custom from detectron + xywh->xyxy->xywh

    else:
        print('NO JETS IN THIS EVENT',len(real_jets))
        clipped_boxes = torch.tensor([[0.0,0.0,0.0,0.0]]) #placeholder value

    return clipped_boxes



def examine_one_image(path,boxes_array,extent):
    #code to plot the calorimeter + cluster bboxes 
    #boxes should be in x,y,w,h 
    print('Examining one image, then exiting.')
    loaded_tensor = torch.load(path)

    f,ax = plt.subplots()
    ii = ax.imshow(loaded_tensor[0],cmap='binary_r')
    # ax.hlines([-np.pi,np.pi],-3,3,color='red',ls='dashed')
    # ax.hlines([-1.9396086193266369,1.940238044375465],-3,3,color='orange',ls='dashed')
    for bbx in boxes_array:
        bb = matplotlib.patches.Rectangle((bbx[0],bbx[1]),bbx[2],bbx[3],lw=1,ec='limegreen',fc='none')
        ax.add_patch(bb)
    cbar = f.colorbar(ii,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('cell significance', rotation=90)
    ax.set(xlabel='eta',ylabel='phi')
    f.savefig('example-make-jet.png')
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
        jets_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/jets/user.cantel.34126190._0000{}.jetD3PD_mc16_JZ4W.r10788.h5".format(file_no)
        chunk_size = 50
        chunk_counter = 0


        with h5py.File(cells_file,"r") as f:
            h5group = f["caloCells"]       
            #convert to numpy arrays in chun sizes
            events = h5group["1d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]
            cells = h5group["2d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]
            n_events_in_file = len(h5group["2d"])

        cell_etas = cells['cell_eta'][0]
        cell_phis = cells['cell_phi'][0] 
        cell_energy = cells['cell_E'][0]
        cell_sigma = cells['cell_Sigma'][0]      
        cell_Esig =  cell_energy / cell_sigma        

        #np.linspace(start, stop, int((stop - start) / step + 1))
        bins_x = np.linspace(min(cell_etas), max(cell_etas), int((max(cell_etas) - min(cell_etas)) / 0.1 + 1))
        bins_y = np.linspace(min(cell_phis), max(cell_phis), int((max(cell_phis) - min(cell_phis)) / ((2*np.pi)/64) + 1))

        H_tot, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                            values=abs(cell_Esig),
                                                            bins=(bins_x,bins_y),
                                                            statistic='sum')

        #transpose to correct format/shape
        H_tot = H_tot.T

        repeat_frac = 0.5
        repeat_rows = int(H_tot.shape[0]*repeat_frac)
        one_box_height = (yedges[-1]-yedges[0])/H_tot.shape[0]
        # Padding
        H_tot  = np.pad(H_tot, ((repeat_rows,repeat_rows),(0,0)),'wrap')



        for i in range(int(n_events_in_file/chunk_size)):
            print('\tLoading chunk {}/{}'.format(chunk_counter,int(n_events_in_file/chunk_size)))

            with h5py.File(jets_file,"r") as f:
                j_data = f["caloCells"]
                # event_data = j_data["1d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]
                jet_data = j_data["2d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]

            #now we'll look at each event individually
            for event_no in range(len(events)):
                unique_file_chunk_event_no = "0"+str(file_no)+"-"+str(chunk_counter)+"-"+str(event_no)

                extent = (xedges[0],xedges[-1],yedges[0]-(repeat_rows*one_box_height),yedges[-1]+(repeat_rows*one_box_height))
                GT_jet_boxes = get_jet_bounding_boxes(jet_data, event_no, extent, (min(cell_phis),max(cell_phis)))

                # Saving, now we save all H_* as a layer in one tensor
                # when we want to access only EM layers, just take that slice out of the sing .pt
                print('\t\tSaving image {}, id: {}, adding to dictionary...'.format(global_counter,unique_file_chunk_event_no))
                overall_save_path = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/cell_images/"
                # H_layers = np.stack([H_tot,H_em,H_had,H_max,H_mean,H_sigma,H_energy,H_time],axis=0)
                # H_layers_tensor = torch.tensor(H_layers)
                # torch.save(H_layers_tensor,overall_save_path+"cell-image-tensor-{}.pt".format(unique_file_chunk_event_no))
                print('GT_jet_boxes\n',GT_jet_boxes)
                #  we'll need to scale the boxes by num bins and x/y range
                GT_jet_boxes[:,0] = (H_tot.shape[1]) * (GT_jet_boxes[:,0]-extent[0])/(extent[1] - extent[0])
                GT_jet_boxes[:,1] = (H_tot.shape[0]) * (GT_jet_boxes[:,1]-extent[2])/(extent[3] - extent[2])
                GT_jet_boxes[:,2] = (H_tot.shape[1]) * GT_jet_boxes[:,2]/(extent[1] - extent[0])
                GT_jet_boxes[:,3] = (H_tot.shape[0]) * GT_jet_boxes[:,3]/(extent[3] - extent[2])

                # examine_one_image(overall_save_path+"cell-image-tensor-{}.pt".format(unique_file_chunk_event_no),GT_jet_boxes,extent)

                annotation_dict_jet[global_counter] = {
                    "image":{
                        "id": global_counter,
                        "file_name": "cell-image-tensor-{}.pt".format(unique_file_chunk_event_no),
                        "img_path": overall_save_path+"cell-image-tensor-{}.pt".format(unique_file_chunk_event_no),
                        "height": len(yedges),
                        "width": len(xedges),
                        "file": file_no,
                        "event": chunk_size*chunk_counter + event_no,
                    },

                    "anns":{
                        "id": global_counter,
                        "n_clusters": len(GT_jet_boxes),
                        "bboxes": GT_jet_boxes.tolist(),
                        "extent": (float(xedges[0]),float(xedges[-1]),float(yedges[0])-(repeat_rows*one_box_height),float(yedges[-1])+(repeat_rows*one_box_height))
                    }
                }

                global_counter += 1
            chunk_counter += 1




    print('Saving jet json annotations file...')
    with open('/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/jet20GeV_annotations.json','w') as json_file:
        json.dump(annotation_dict_jet,json_file)




