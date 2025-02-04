import torch
import torchvision
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt


import h5py
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--jet_path', type=str, required=True, help='path to the jets .h5 directory',)
parser.add_argument('--path', type=str, required=True, help='path to the cells .h5 directory',)
parser.add_argument('--output_dir', type=str, nargs='?', const='../cache/images/', help='path to the output .json file')
parser.add_argument('--output_json', type=str, nargs='?', const='../cache/anns_central_jets_20GeV.json', help='path to the output .json file')
parser.add_argument('--job_id', type=int, required=True, help='Grid job id',)
parser.add_argument('-p','--proc', type=str, required=True, help='Type of process (JZ1,JZ2,JZ3,ttbar)',)
args = parser.parse_args()



def remove_nan(array):
    # find the indices where there are not nan values
    good_indices = np.where(array==array) 
    return array[good_indices]

def clip_boxes_to_image(boxes, extent):
    #https://detectron2.readthedocs.io/en/latest/_modules/torchvision/ops/boxes.html
    boxes = torchvision.ops.box_convert(boxes,'xywh','xyxy')

    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    xmin,xmax,ymin,ymax = extent

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(xmin, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(xmax, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(ymin, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(ymax, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=xmin, max=xmax)
        boxes_y = boxes_y.clamp(min=ymin, max=ymax)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    clipped_boxes = clipped_boxes.reshape(boxes.shape)

    # ensure that the new clipped boxes satisfy height requirements
    # here in xyxy coords
    heights = (clipped_boxes[:,3]-clipped_boxes[:,1])
    final_boxes_xyxy = clipped_boxes[heights>0.1]
    final_boxes = torchvision.ops.box_convert(final_boxes_xyxy, 'xyxy','xywh')
    return final_boxes



def get_jet_bounding_boxes(jet_data,event_no,extent,min_max_tuple):
    R = 0.4 # anti-kt 
    WIDTH,HEIGHT = 2*R, 2*R 
    MIN_PHI_VALUE = min_max_tuple[0]
    MAX_PHI_VALUE = min_max_tuple[1]

    jets = jet_data[event_no]
    real_jets = remove_nan(jets)
    #loop over all jets in this event 
    if len(real_jets) > 0:
        filtered_pt_jets = real_jets[real_jets['AntiKt4EMTopoJets_JetConstitScaleMomentum_pt'] > 20_000] # Select the pt threshold (in MeV)
        filtered_jets = filtered_pt_jets[abs(filtered_pt_jets['AntiKt4EMTopoJets_JetConstitScaleMomentum_eta']) < 2.1] # Select the eta threshold
        
        if len(filtered_jets)>0:
            box_list = []
            pt_list  = []
            for jet_no in range(len(filtered_jets)):
                jet_eta = filtered_jets['AntiKt4EMTopoJets_JetConstitScaleMomentum_eta'][jet_no]
                jet_phi = filtered_jets['AntiKt4EMTopoJets_JetConstitScaleMomentum_phi'][jet_no]
                jet_pt  = filtered_jets['AntiKt4EMTopoJets_JetConstitScaleMomentum_pt'][jet_no] / 1000 
                xmin = jet_eta - R
                ymin = jet_phi - R

                # jet boxes that cross the discontinuity/wrap around
                if (jet_phi+R > MAX_PHI_VALUE - (extent[3]-MAX_PHI_VALUE)) or (jet_phi-R < MIN_PHI_VALUE - (extent[2]-MIN_PHI_VALUE)):
                    wrapped_jet_phi = jet_phi - np.sign(jet_phi)*2*np.pi
                    box_list.append([xmin,wrapped_jet_phi-R,WIDTH,HEIGHT])
                    box_list.append([xmin,ymin,WIDTH,HEIGHT])
                    pt_list.append(jet_pt)
                    pt_list.append(jet_pt)

                else:
                    box_list.append([xmin,ymin,WIDTH,HEIGHT])
                    pt_list.append(jet_pt)

            tensor_of_boxes = torch.tensor(box_list)
            tensor_of_pts = torch.tensor(pt_list)
            clipped_boxes = clip_boxes_to_image(tensor_of_boxes,extent) # Custom from detectron + xywh->xyxy->xywh
        
        else:
            print('NO JETS IN THIS EVENT',len(real_jets))
            clipped_boxes = torch.tensor([[0.0,0.0,0.0,0.0]]) #placeholder value
            tensor_of_pts = torch.tensor([0.0]) 

    else:
        print('NO JETS IN THIS EVENT',len(real_jets))
        clipped_boxes = torch.tensor([[0.0,0.0,0.0,0.0]]) #placeholder value
        tensor_of_pts = torch.tensor([0.0]) 

    return clipped_boxes, tensor_of_pts



def examine_one_image(boxes_array,extent):
    #code to plot the calorimeter + cluster bboxes 
    #boxes should be in x,y,w,h 
    print('Examining one image, then exiting.')

    f,ax = plt.subplots()
    # ax.hlines([-np.pi,np.pi],-3,3,color='red',ls='dashed')
    # ax.hlines([-1.9396086193266369,1.940238044375465],-3,3,color='orange',ls='dashed')
    for bbx in boxes_array:
        bb = matplotlib.patches.Rectangle((bbx[0],bbx[1]),bbx[2],bbx[3],lw=1,ec='limegreen',fc='none')
        ax.add_patch(bb)
    cbar = f.colorbar(ii,ax=ax)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.set_label('cell significance', rotation=90)
    ax.set(xlabel='eta',ylabel='phi')
    f.savefig('examine-jet.png')
    plt.close()
    quit()







if __name__=="__main__":

    annotation_dict = {}
    annotation_dict_jet = {}
    global_counter = 0

    if args.proc == "ttbar":
        file_nos = [22,23,26,27,28,30,31,32,33,34,35,36,37,38,39,40,41,42]
        # file_nos = [22,23,26,27,28,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]
        tag = args.proc + ".r15583"
    elif args.proc == "JZ4":
        file_nos = [13,15,16,19,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
        # file_nos = [13,15,16,19,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85]
        tag = args.proc + ".r14365"
    print(len(file_nos), " files")

    for file_no in file_nos:
        print('Loading file {}/{}'.format(file_no,len(file_nos)))      
        if args.proc in ["JZ1", "JZ2", "JZ3", "JZ4", "JZ5", "ttbar"]:
            cells_file = args.path     + "user.lbozianu.{}._0000{}.calocellD3PD_mc21_14TeV_{}.h5".format(args.job_id,file_no,tag) 
            jets_file  = args.jet_path + "user.lbozianu.{}._0000{}.jetD3PD_mc21_14TeV_{}.h5".format(args.job_id,file_no,tag)
        else:
            print(args.proc," not recognised process, check spelling..")
                          
        chunk_size = 100
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
        #central cells! 
        cell_phis = cell_phis[abs(cell_etas)<2.5]     
        cell_Esig = cell_Esig[abs(cell_etas)<2.5]     
        cell_etas = cell_etas[abs(cell_etas)<2.5]     

        # make one image just to obtain the correct extent etc.
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
        extent = (xedges[0],xedges[-1],yedges[0]-(repeat_rows*one_box_height),yedges[-1]+(repeat_rows*one_box_height))
        # Padding
        H_tot  = np.pad(H_tot, ((repeat_rows,repeat_rows),(0,0)),'wrap')

        #now we'll look at each event individually
        for i in range(int(n_events_in_file/chunk_size)):
            print('\tLoading chunk {}/{}'.format(chunk_counter,int(n_events_in_file/chunk_size)))

            with h5py.File(jets_file,"r") as f:
                j_data = f["caloCells"]
                # event_data = j_data["1d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]
                jet_data = j_data["2d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]

            #now we'll look at each event individually
            for event_no in range(len(events)):
                unique_file_chunk_event_no = "0"+str(file_no)+"-"+str(chunk_counter)+"-"+str(event_no)

                GT_jet_boxes, GT_jet_pts = get_jet_bounding_boxes(jet_data, event_no, extent, (min(cell_phis),max(cell_phis)))

                print('\tProcessing image {}, id: {}, adding to dictionary...'.format(global_counter,unique_file_chunk_event_no))
    
                GT_jet_boxes[:,0] = (H_tot.shape[1]) * (GT_jet_boxes[:,0]-extent[0])/(extent[1] - extent[0])
                GT_jet_boxes[:,1] = (H_tot.shape[0]) * (GT_jet_boxes[:,1]-extent[2])/(extent[3] - extent[2])
                GT_jet_boxes[:,2] = (H_tot.shape[1]) * GT_jet_boxes[:,2]/(extent[1] - extent[0])
                GT_jet_boxes[:,3] = (H_tot.shape[0]) * GT_jet_boxes[:,3]/(extent[3] - extent[2])

                # examine_one_image(GT_jet_boxes,extent)

                annotation_dict_jet[global_counter] = {
                    "image":{
                        "id": global_counter,
                        "file_name": "cell-img-{}.pt".format(unique_file_chunk_event_no),
                        "img_path": args.output_dir+"cell-img-{}.pt".format(unique_file_chunk_event_no),
                        "height": len(yedges),
                        "width": len(xedges),
                        "file": file_no,
                        "event": chunk_size*chunk_counter + event_no,
                    },

                    "anns":{
                        "id": global_counter,
                        "n_jets": len(GT_jet_boxes),
                        "bboxes": GT_jet_boxes.tolist(),
                        "jet_pt": GT_jet_pts.tolist(),
                        "extent": (float(xedges[0]),float(xedges[-1]),float(yedges[0])-(repeat_rows*one_box_height),float(yedges[-1])+(repeat_rows*one_box_height))
                    }
                }

                global_counter += 1
            chunk_counter += 1

    print('Saving jet json annotations json file...')
    with open(args.output_json,'w') as json_file:
        json.dump(annotation_dict_jet,json_file)


