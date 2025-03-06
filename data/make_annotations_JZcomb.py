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
args = parser.parse_args()

# JZ_file_dict = {
#     "JZ2" : ["000001","000002","000003","000004","000005","000006","000007","000008","000009","000010","000011","000012","000013","000014","000015","000016","000017","000018","000019","000020","000021","000022","000023","000024","000025","000026","000027","000028","000029","000030","000041","000042","000043","000044","000045","000046","000047","000048","000049","000050"],
#     "JZ3" : ["000011","000012","000015","000016","000017","000018","000019","000021","000022","000023","000025","000026","000027","000028","000029","000030","000031","000032","000033","000034","000035","000036","000037","000038","000039","000040","000041","000042","000043","000044","000045","000046","000047","000048","000049","000050","000051","000052","000053"],
#     "JZ4" : ["000013","000015","000016","000019","000022","000023","000024","000025","000026","000027","000028","000029","000030","000031","000032","000033","000034","000035","000036","000037","000038","000039","000040","000041","000042","000043","000044","000045","000046","000047","000048","000049","000050","000051","000052","000053","000054","000055","000056","000057","000058","000059","000060","000061","000062","000063","000064","000065","000066","000067","000068","000069","000070","000071","000072","000073","000074","000075","000076","000077","000078","000079","000080","000081","000082","000083","000084","000085"]
# }

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
            print('NO CENTRAL JETS IN THIS EVENT, will not be saved in JSON file',len(real_jets))
            clipped_boxes = torch.tensor([[-0.4,-0.4,0.4,0.4]]) #placeholder value
            tensor_of_pts = torch.tensor([0.99]) 

    else:
        print('NO CENTRAL JETS IN THIS EVENT, will not be saved in JSON file',len(real_jets))
        clipped_boxes = torch.tensor([[-0.4,-0.4,0.4,0.4]]) #placeholder value
        tensor_of_pts = torch.tensor([0.99]) 

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

    JZ_file_dict = {
        # training files
        # "JZ2" : ["000001","000002","000003","000004","000005","000006","000007","000008","000009","000010","000011","000012","000013","000014","000015","000016","000017","000018","000019","000020","000021","000022","000023","000024","000025","000026",],
        # "JZ3" : ["000011","000012","000015","000016","000017","000018","000019","000021","000022","000023","000025","000026","000027","000028","000029","000030","000031","000032","000033","000034","000035","000036","000037","000038","000039","000040"],
        # "JZ4" : ["000013","000015","000016","000019","000022","000023","000024","000025","000026","000027","000028","000029","000030","000031","000032","000033","000034","000035","000036","000037","000038","000039","000040","000041","000042","000043","000044","000045","000046","000047","000048","000049","000050","000051","000052","000053","000054","000055","000056","000057","000058","000059","000060","000061","000062"],
        # test files
        "JZ2" : ["000027","000028","000029","000030","000041","000042","000043","000044","000045","000046","000047","000048","000049","000050"],
        "JZ3" : ["000041","000042","000043","000044","000045","000046","000047","000048","000049","000050","000051","000052","000053"],
        "JZ4" : ["000063","000064","000065","000066","000067","000068","000069","000070","000071","000072","000073","000074","000075","000076","000077","000078","000079","000080","000081","000082","000083","000084","000085",],
    }

    
    JZ_grid_dict = {
        "JZ2" : "43186502",
        "JZ3" : "43186499",
        "JZ4" : "42998779",
    }


    for proc in ["JZ2", "JZ3", "JZ4"]:
        file_nos = JZ_file_dict[proc]                 

        for file_no in file_nos:
            print('Loading file {}/{}'.format(file_no,len(file_nos)))     
            cells_file = args.path +     "{}/user.lbozianu/user.lbozianu.{}._{}.calocellD3PD_mc21_14TeV_{}.r14365.h5".format(proc, JZ_grid_dict[proc], file_no, proc) 
            jets_file  = args.jet_path + "{}/user.lbozianu/user.lbozianu.{}._{}.jetD3PD_mc21_14TeV_{}.r14365.h5".format(proc, JZ_grid_dict[proc], file_no, proc)


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
                    event_data = j_data["1d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]
                    jet_data = j_data["2d"][chunk_size*chunk_counter : chunk_size*(chunk_counter+1)]

                #now we'll look at each event individually
                for event_no in range(len(event_data)):
                    unique_file_chunk_event_no = "0"+str(file_no)+"-"+str(chunk_counter)+"-"+str(event_no)

                    GT_jet_boxes, GT_jet_pts = get_jet_bounding_boxes(jet_data, event_no, extent, (min(cell_phis),max(cell_phis)))
                    # if torch.equal(GT_jet_pts,torch.tensor([0.0])): continue
                        
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
                            "img_path": args.output_dir+"/{}/cell-img-{}.pt".format(proc, unique_file_chunk_event_no),
                            "height": len(yedges),
                            "width": len(xedges),
                            "file": file_no,
                            "proc": proc,
                            "grid_id": JZ_grid_dict[proc],
                            "event": chunk_size*chunk_counter + event_no,
                        },

                        "anns":{
                            "id": global_counter,
                            "mc_event_weight": float(event_data["ei_mc_event_weight"][event_no]),
                            "n_jets": len(GT_jet_boxes),
                            "bboxes": GT_jet_boxes.tolist(),
                            "jet_pt": GT_jet_pts.tolist(),
                            "extent": (float(xedges[0]),float(xedges[-1]),float(yedges[0])-(repeat_rows*one_box_height),float(yedges[-1])+(repeat_rows*one_box_height))
                        }
                    }

                    global_counter += 1
                chunk_counter += 1

    print('Saving jet json annotations json file...')

    with open(args.output_json+"anns_central_jets_JZcomb0_test.json",'w') as json_file:
        json.dump(annotation_dict_jet,json_file)


