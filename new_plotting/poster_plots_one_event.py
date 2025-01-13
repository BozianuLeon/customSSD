import numpy as np
import scipy
import torch
import torchvision
import sys
import os
import time
import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep
hep.style.use(hep.style.ATLAS)

import h5py
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
# caution: path[0] is reserved for script path 
sys.path.insert(1, '/home/users/b/bozianu/work/SSD/SSD')

from utils.utils import matched_boxes, unmatched_boxes, wrap_check_NMS, wrap_check_truth, remove_nan
from utils.metrics import RetrieveCellIdsFromBox, RetrieveCellIdsFromCluster
from utils.metrics import get_physics_dictionary


MIN_CELLS_PHI,MAX_CELLS_PHI = -3.1334076, 3.134037
MIN_CELLS_ETA,MAX_CELLS_ETA = -4.823496, 4.823496

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)


def CalculateMatchFromBox(box_in_question,boxes_array,desired_cells):
    if desired_cells is None:  
        print('It happened! None -> np.nan')  
        return np.nan
    else:
        box_in_question = torch.tensor(box_in_question).unsqueeze(0)
        boxes_tensor = torch.tensor(boxes_array)
        iou_mat = torchvision.ops.boxes.box_iou(box_in_question, boxes_tensor)
        matched_vals, matches = iou_mat.max(dim=1)
    return int(matched_vals>0.4)


def CalculateMatchesFromBoxes(boxes_array1,boxes_array2):
    #calculate the number of predicted boxes lie on top of a GT box
    boxes_tensor1 = torch.tensor(boxes_array1)
    boxes_tensor2 = torch.tensor(boxes_array2)
    iou_mat = torchvision.ops.boxes.box_iou(boxes_tensor1, boxes_tensor2)
    matched_vals, matches = iou_mat.max(dim=1)

    return np.array(matched_vals>0.4,dtype=np.float32)
    # matched_q = np.zeros_like(matched_vals)
    # matched_q[matched_vals>0.4]=1.0
    # return matched_q
















def circular_mean(phi_values):
    """
    Calculate the circular mean (average) of a list of phi_values.
    Handles the periodicity of phi_values correctly.
    
    :param phi_values: List of phi_values in radians
    :return: Circular mean in radians
    """
    sin_sum = np.sum(np.sin(phi_values))
    cos_sum = np.sum(np.cos(phi_values))
    circular_mean = np.arctan2(sin_sum, cos_sum)
    return circular_mean



def phi_mod2pi(phis):
    repeated_phis = np.copy(phis)
    mask = repeated_phis >= 0

    repeated_phis[mask] -= 2*np.pi
    repeated_phis[~mask] += 2*np.pi
    return repeated_phis

def clip_boxes_to_image(boxes, extent):
    #https://detectron2.readthedocs.io/en/latest/_modules/torchvision/ops/boxes.html
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
    #ensure that the new clipped boxes satisfy height requirements
    #remember here we're in xyxy coords
    heights = (clipped_boxes[:,3]-clipped_boxes[:,1])
    return clipped_boxes[heights>0.01]




def union(boxa,boxb):
    x = min(boxa[0], boxb[0])
    y = min(boxa[1], boxb[1])
    w = max(boxa[0]+boxa[2], boxb[0]+boxb[2]) - x
    h = max(boxa[1]+boxa[3], boxb[1]+boxb[3]) - y
    return (x,y,w,h)

def intersection(boxa,boxb):
    x = max(boxa[0], boxb[0])
    y = max(boxa[1], boxb[1])
    w = min(boxa[0]+boxa[2], boxb[0]+boxb[2]) - x
    h = min(boxa[1]+boxa[3], boxb[1]+boxb[3]) - y
    if w<0 or h<0:
        return ()
    return (x,y,w,h)


def merge_rectangles(boxes, max_size=(1.5, 1.5)):

    def contains(rect_a, rect_b):
        x1a, y1a, w1a, h1a = rect_a
        x1b, y1b, w1b, h1b = rect_b
        x2a, y2a = x1a + w1a, y1a + h1a
        x2b, y2b = x1b + w1b, y1b + h1b
        return x1a <= x1b and y1a <= y1b and x2a >= x2b and y2a >= y2b

    new_boxes = list(boxes)  
    merged_boxes = []  
    failed_merge = []  

    while new_boxes:
        ra = new_boxes[0]  # Select the first rectangle for comparison
        new_boxes = new_boxes[1:]  # Remove the first rectangle from the list

        mask = [intersection(ra, rb) != () for rb in new_boxes]
        merged_mask = not any(mask)  # True if ra has no intersections with any remaining rectangles

        if not merged_mask:
            rb_idx = mask.index(True)  # Find the index of the first intersecting rectangle (rb)
            rb = new_boxes[rb_idx]  # Select the intersecting rectangle

            new_boxes = new_boxes[:rb_idx] + new_boxes[rb_idx + 1:]  # Remove rb from the list of rectangles

            merged_box = union(ra, rb)  # Merge ra and rb
            if max_size is None or (merged_box[2] <= max_size[0] and merged_box[3] <= max_size[1]):
                new_boxes.append(merged_box)  # Append the merged rectangle if it meets the size constraint
            else:
                failed_merge.append(ra)  # Add ra and rb to the failed_merge list if they exceed max_size
                failed_merge.append(rb)
        else:
            merged_boxes.append(ra)  # Add ra to merged_boxes if no intersections were found

    # Combine merged_boxes and failed_merge before checking for rectangles contained by larger rectangles
    all_rectangles = np.concatenate((merged_boxes, failed_merge)) if failed_merge else merged_boxes

    # Check for rectangles entirely contained
    final_boxes = []
    for box in all_rectangles:
        if not any(contains(rb, box) for rb in all_rectangles if not np.array_equal(rb, box)):
            final_boxes.append(box)

    return np.array(final_boxes)



def get_bboxes(cluster_data,cluster_cell_data,extent):
    clusters = cluster_data[['cl_E_em', 'cl_E_had', 'cl_cell_n', 'cl_eta', 'cl_phi', 'cl_pt', 'cl_time']]
    clusters = remove_nan(clusters) #remove nan padding
    cluster_cells = cluster_cell_data

    #loop over all clusters in this event 
    box_list = []
    for cluster_no in range(len(clusters)):
        #check the cluster's energy, only include those that make the cut!
        if clusters[cluster_no]['cl_E_em']+clusters[cluster_no]['cl_E_had'] > 5000:

            cluster_cells_this_cluster = remove_nan(cluster_cells[cluster_no]) #remove nan padding on cl_cells
            cell_etas = cluster_cells_this_cluster['cl_cell_raw_eta']
            cell_phis = cluster_cells_this_cluster['cl_cell_raw_phi']
            xmin,ymin = min(cell_etas),min(cell_phis)
            width,height = max(cell_etas)-xmin, max(cell_phis)-ymin

            #if the cells "wrap around", some lie at the top of the image, some at the bottom
            if height>5.0:
                cell_phis_wrap = phi_mod2pi(cell_phis)

                #Create two boxes assoc. to this cluster, one >0
                # Look at the phis above zero, find their min/max
                min_gtzero_phi = min(cell_phis[cell_phis>0])
                ymin_box_above_zero = min_gtzero_phi
                height_box_above_zero = max(cell_phis_wrap) - min_gtzero_phi 

                #and the repeated box, living <0
                max_ltzero_phi = max(cell_phis[cell_phis<=0])
                ymin_box_below_zero = min(cell_phis_wrap) 
                height_box_below_zero = max_ltzero_phi - min(cell_phis_wrap)
                assert abs(height_box_above_zero-height_box_below_zero)<1e-5, "Heights should be the same! {}, {}".format(height_box_above_zero,height_box_below_zero)
                # print('I repeated this box, it now has two reps. One here {} with height {} and one {} with {}'.format(ymin_box_above_zero,height_box_above_zero,ymin_box_below_zero,height_box_below_zero))
                if height_box_above_zero>0.01 and width>0.01:
                    box_list.append([xmin,ymin_box_above_zero,width,height_box_above_zero])
                    box_list.append([xmin,ymin_box_below_zero,width,height_box_below_zero])
            
            #since we repeat a part of the calorimeter in phi, we need to compensate by adding in normal boxes
            #that are repeated
            else:
                cell_phis_wrap = phi_mod2pi(cell_phis)
                # create the 2 symmetric boxes for this cluster
                ymin_orig = min(cell_phis)
                ymin_wrapped_region = min(cell_phis_wrap) 
                height = max(cell_phis) - min(cell_phis) 

                if height>0.01 and width>0.01:
                    box_list.append([xmin,ymin_orig,width,height])
                    box_list.append([xmin,ymin_wrapped_region,width,height])



    #Now we have a list of boxes that we would like to clip to the image and merge them
    tensor_of_boxes = torch.tensor(np.array(box_list))
    boxes_xyxy_tensor = torchvision.ops.box_convert(tensor_of_boxes,'xywh','xyxy')
    unmerged_clipped_boxes_xyxy = clip_boxes_to_image(boxes_xyxy_tensor,extent)
    unmerged_clipped_boxes = torchvision.ops.box_convert(unmerged_clipped_boxes_xyxy, 'xyxy','xywh')

    final_boxes_list = merge_rectangles(unmerged_clipped_boxes.tolist())

    return final_boxes_list



def make_bounding_boxes(list_topocluster_cells,extent):
    # Function that creates the bounding boxes for all topoclusters in an event
    # input:
    # a list of numpy structured arrays, output of RetrieveCellIdsFromCluster metrics.py
    # box should enclose ALL cells in a topocluster (or maybe 2sigma in the future)

    n_clusters = len(list_topocluster_cells)
    #loop over all clusters in this event 
    box_list = []
    for cluster_no in range(n_clusters):
        cluster_i = list_topocluster_cells[cluster_no]
        if sum(cluster_i['cell_E']) > 5000:
            cell_etas = cluster_i['cell_eta']
            cell_phis = cluster_i['cell_phi']
            cell_phis_wrap = phi_mod2pi(cell_phis)

            xmin,ymin = min(cell_etas),min(cell_phis)
            width,height = max(cell_etas)-xmin, max(cell_phis)-ymin
            if height>=1.5*np.pi:
                #Create two boxes assoc. to this cluster, one >0
                cell_phis_gtzero = cell_phis[cell_phis > 0]
                cell_phis_ltzero = cell_phis[cell_phis <= 0]
                cell_phis_wrap_gtzero = cell_phis_wrap[cell_phis > 0]

                # Look at all the phis above zero, find their min/max
                ymin_box_above_zero = np.min(cell_phis_gtzero)
                height_box_above_zero = np.max(cell_phis_wrap) - ymin_box_above_zero 

                # and the repeated box, living < 0
                ymax_box_below_zero = np.max(cell_phis_ltzero)
                height_box_below_zero = ymax_box_below_zero - np.min(cell_phis_wrap)
                
                assert abs(height_box_above_zero-height_box_below_zero)<1e-5, "Heights should be the same! {}, {}".format(height_box_above_zero,height_box_below_zero)
                # print('I repeated this box, it now has two reps. One here {:.4f} with height {:.4f} and one {:.4f} with {:.4f}'.format(ymin_box_above_zero,height_box_above_zero, np.min(cell_phis_wrap),height_box_below_zero))
                if (height_box_above_zero + width)>0.01:
                    box_list.append([xmin,ymin_box_above_zero,width,height_box_above_zero])
                    box_list.append([xmin,np.min(cell_phis_wrap),width,height_box_below_zero])

            else:
                # create the 2 symmetric boxes for this cluster
                ymin_orig = min(cell_phis)
                ymin_wrapped_region = min(cell_phis_wrap) 
                height = max(cell_phis) - min(cell_phis) 

                if (height + width)>0.01:
                    box_list.append([xmin,ymin_orig,width,height])
                    box_list.append([xmin,ymin_wrapped_region,width,height])
    
    #Now we have a list of boxes that we would like to clip to the image and merge them
    tensor_of_boxes = torch.tensor(box_list,dtype=torch.float32)
    boxes_xyxy_tensor = torchvision.ops.box_convert(tensor_of_boxes,'xywh','xyxy')
    unmerged_clipped_boxes_xyxy = clip_boxes_to_image(boxes_xyxy_tensor,extent)
    unmerged_clipped_boxes = torchvision.ops.box_convert(unmerged_clipped_boxes_xyxy, 'xyxy','xywh')

    final_boxes_array = merge_rectangles(unmerged_clipped_boxes)

    return final_boxes_array







necessary_results = {
    'n_tboxes':             [],
    'num_tboxes':           [],
    'tboxes_energies':      [],
    'tboxes_eta':           [],
    'tboxes_phi':           [],
    'tboxes_eT':            [],
    'tboxes_n_cells':       [],
    'tboxes_noise':         [],
    'tboxes_significance':  [],
    'tboxes_neg_frac':      [],
    'tboxes_max_frac':      [],
    'tboxes_energies_2sig': [],
    'tboxes_eT_2sig':       [], 
    'tboxes_n_cells_2sig':  [],  
    'tboxes_matched': [],
    
    'n_pboxes':             [],
    'num_pboxes':           [],
    'pboxes_energies':      [],
    'pboxes_eta':           [],
    'pboxes_phi':           [],
    'pboxes_eT':            [],
    'pboxes_n_cells':       [],
    'pboxes_noise':         [],
    'pboxes_significance':  [],
    'pboxes_neg_frac':      [],
    'pboxes_max_frac':      [],
    'pboxes_energies_2sig': [],
    'pboxes_eT_2sig':       [], 
    'pboxes_n_cells_2sig':  [],  
    'pboxes_matched': [],

}

def make_poster_plot_data(
    folder_containing_struc_array,
    save_folder,
    image_format='png',
):

    with open(folder_containing_struc_array + "/struc_array.npy", 'rb') as f:
        a = np.load(f)

    for i in range(1):
        i=18

        start = time.perf_counter()
        print(i)
        extent_i = a[i]['extent']
        preds = a[i]['p_boxes']
        trues = a[i]['t_boxes']
        scores = a[i]['p_scores']

        pees = preds[np.where(preds[:,0] > 0.0)]
        tees = trues[np.where(trues[:,0] > 0.0)]

        #make boxes cover extent
        tees[:,(0,2)] = (tees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
        tees[:,(1,3)] = (tees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]
        pees[:,(0,2)] = (pees[:,(0,2)]*(extent_i[1]-extent_i[0]))+extent_i[0]
        pees[:,(1,3)] = (pees[:,(1,3)]*(extent_i[3]-extent_i[2]))+extent_i[2]

        #wrap check boxes here
        print(tees.shape)
        pees = wrap_check_NMS(pees,scores,MIN_CELLS_PHI,MAX_CELLS_PHI,threshold=0.2)
        tees = wrap_check_truth(tees,MIN_CELLS_PHI,MAX_CELLS_PHI)
        
        #get the cells
        h5f = a[i]['h5file']
        try:
            h5f = h5f.decode('utf-8')
        except:
            h5f = h5f
        event_no = a[i]['event_no']

        #load cells from h5
        # cells_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/cells/user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        cells_file = "/home/users/b/bozianu/work/data/pileup50k/cells/user.cantel.34126190._0000{}.calocellD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        with h5py.File(cells_file,"r") as f:
            h5group = f["caloCells"]
            cells = h5group["2d"][event_no]

        # clusters_file = "/srv/beegfs/scratch/shares/atlas_caloM/mu_32_50k/clusters/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        clusters_file = "/home/users/b/bozianu/work/data/pileup50k/clusters/user.cantel.34126190._0000{}.topoclusterD3PD_mc16_JZ4W.r10788.h5".format(h5f)
        with h5py.File(clusters_file,"r") as f:
            cl_data = f["caloCells"] 
            event_data = cl_data["1d"][event_no]
            cluster_data = cl_data["2d"][event_no]
            cluster_cell_data = cl_data["3d"][event_no]
            raw_E_mask = (cluster_data['cl_E_em']+cluster_data['cl_E_had']) > 5000 #5GeV cut
            # cluster_data = cluster_data[raw_E_mask]
            # cluster_cell_data = cluster_cell_data[raw_E_mask]


        l_pred_cells = RetrieveCellIdsFromBox(cells,pees)
        l_true_cells = RetrieveCellIdsFromBox(cells,tees)
        l_topo_cells = RetrieveCellIdsFromCluster(cells,cluster_cell_data)

        pb_phys_dict = get_physics_dictionary(l_pred_cells,cells)
        tb_phys_dict = get_physics_dictionary(l_true_cells,cells)
        tc_phys_dict = get_physics_dictionary(l_topo_cells,cells)

        





        # Plotting how we make the images / annotations
        f,a = plt.subplots(1,1,figsize=(7,9))
        #all topoclusters
        for c in range(len(cluster_data)):
            cc = cluster_data[c]
            if cc['cl_E_em'] + cc['cl_E_had'] > 5000:
                a.plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.6,color='blue',ms=7,markeredgecolor='k')
            else:
                a.plot(cc['cl_eta'],cc['cl_phi'], marker='h',alpha=.3,color='plum',ms=6)

        #all topoclusers mydict
        # for cl in range(len(tc_phys_dict['eta'])):
        #     a.plot(tc_phys_dict['eta'][cl],tc_phys_dict['phi'][cl], marker='p',alpha=.25,color='blue',ms=5,markeredgecolor='k')

        #make boxes
        ground_truth_bounding_boxes = get_bboxes(cluster_data,cluster_cell_data,extent_i) #output xywh
        ground_truth_bounding_boxes = torchvision.ops.box_convert(torch.tensor(ground_truth_bounding_boxes),'xywh','xyxy')
        wrapped_truth_boxes = wrap_check_truth(ground_truth_bounding_boxes,MIN_CELLS_PHI,MAX_CELLS_PHI)
        for tbx in wrapped_truth_boxes:
            x,y=float(tbx[0]),float(tbx[1])
            x2,y2=float(tbx[2]),float(tbx[3])
            # a.plot(x,y,marker='*',color='green',alpha=0.5)
            # a.add_patch(matplotlib.patches.Rectangle((x,y),x2-x,y2-y,lw=2,ec='green',fc='none'))

        #new make boxes
        final_boxes_array = make_bounding_boxes(l_topo_cells,extent_i)
        final_boxes_array = torchvision.ops.box_convert(torch.tensor(final_boxes_array),'xywh','xyxy')
        wrapped_boxes_array = wrap_check_truth(final_boxes_array,MIN_CELLS_PHI,MAX_CELLS_PHI)
        for tbx in wrapped_boxes_array:
            x,y=float(tbx[0]),float(tbx[1])
            x2,y2=float(tbx[2]),float(tbx[3])
            # a.add_patch(matplotlib.patches.Rectangle((x,y),x2,y2,lw=2,ec='green',fc='none'))
            # a.add_patch(matplotlib.patches.Rectangle((x,y),x2-x,y2-y,lw=0.9,ec='cyan',fc='none',ls='--'))

        #truth boxes
        for tbx in tees:
            x,y=float(tbx[0]),float(tbx[1])
            w,h=float(tbx[2])-float(tbx[0]),float(tbx[3])-float(tbx[1])  
            bbo = matplotlib.patches.Rectangle((x,y),w,h,lw=2,ec='green',fc='none')
            a.add_patch(bbo)

        #box predictions
        for pbx in pees:
            x,y=float(pbx[0]),float(pbx[1])
            w,h=float(pbx[2])-float(pbx[0]),float(pbx[3])-float(pbx[1])  
            bbo = matplotlib.patches.Rectangle((x,y),w,h,lw=2,ec='red',fc='none')
            # a.add_patch(bbo)

        a.axhline(y=min(cells['cell_phi']), color='r', alpha=0.6, linestyle='--')
        a.axhline(y=max(cells['cell_phi']), color='r', alpha=0.6, linestyle='--')
        a.grid()
        a.set(xlabel='$\eta$',ylabel='$\phi$',xlim=(min(cells['cell_eta']), max(cells['cell_eta'])),ylim=(min(cells['cell_phi'])-3, max(cells['cell_phi'])+3))

        legend_elements = [matplotlib.patches.Patch(facecolor='w', lw=1.5, ec='green',fc='none',label='Target Boxes'),
                           matplotlib.lines.Line2D([],[], marker='h', color='blue', label=f'Topoclusters ≥5 GeV',linestyle='None',markersize=15,markeredgecolor='k'),
                           matplotlib.lines.Line2D([],[], marker='h', color='plum', label=f'Topoclusters <5 GeV',linestyle='None',markersize=11,alpha=0.75)]
        a.legend(handles=legend_elements, loc='lower left',bbox_to_anchor=(0.53, 0.74),prop={'family':'serif','style':'normal','weight':'bold','size':12})
        f.savefig('/home/users/b/bozianu/work/SSD/SSD/cached_plots/SSD1_50k5_mu_15e/poster/boxes-clusters-{}.{}'.format(i,image_format),dpi=400,format=image_format, bbox_inches="tight")




        f,a = plt.subplots(1,1,figsize=(7,9))
        #all topoclusers mydict
        for cl in range(len(tc_phys_dict['eta'])):
            a.plot(tc_phys_dict['eta'][cl],tc_phys_dict['phi'][cl], marker='p',alpha=.25,color='blue',ms=5,markeredgecolor='k')
            cl_cells = l_topo_cells[cl]
            a.plot(np.mean(cl_cells['cell_eta']),circular_mean(cl_cells['cell_phi']), marker='+',alpha=.55,color='green',ms=5)

            if np.sign(min(cl_cells['cell_phi'])) == np.sign(max(cl_cells['cell_phi'])):
                a.add_patch(matplotlib.patches.Rectangle((min(cl_cells['cell_eta']),min(cl_cells['cell_phi'])),max(cl_cells['cell_eta'])-min(cl_cells['cell_eta']),max(cl_cells['cell_phi'])-min(cl_cells['cell_phi']),ls='--',lw=0.7,ec='green',fc='none'))

        a.axhline(y=min(cells['cell_phi']), color='r', alpha=0.6, linestyle='--')
        a.axhline(y=max(cells['cell_phi']), color='r', alpha=0.6, linestyle='--')
        a.grid()
        a.set(xlabel='$\eta$',ylabel='$\phi$',xlim=(min(cells['cell_eta']), max(cells['cell_eta'])),ylim=(min(cells['cell_phi'])-3, max(cells['cell_phi'])+3))
        a.set_title(f'{len(tc_phys_dict["eta"])}, {len(cluster_data)} Clusters, {len(ground_truth_bounding_boxes)},{len(tees)} Boxes')
        f.savefig('/home/users/b/bozianu/work/SSD/SSD/cached_plots/SSD1_50k5_mu_15e/poster/boxes-myclusters-{}.{}'.format(i,image_format),dpi=400,format=image_format, bbox_inches="tight")

        # f, ax = plt.subplots()
        # ax.hist(tc_phys_dict['significance'],bins=200,histtype='step',color='blue',alpha=0.95)
        # ax.set(xlabel='significance',ylabel='Freq')
        # f.tight_layout()
        # f.savefig('/home/users/b/bozianu/work/SSD/SSD/cached_plots/SSD1_50k5_mu_15e/poster/cluster-signif-{}.{}'.format(i,image_format),dpi=400,format=image_format, bbox_inches="tight")



        # Make pixels and display image with target/predictions
        cell_etas = cells['cell_eta']
        cell_phis = cells['cell_phi']
        cell_Esig =  cells['cell_E'] / cells['cell_Sigma']    
        bins_x = np.linspace(min(cell_etas), max(cell_etas), int((max(cell_etas) - min(cell_etas)) / 0.1 + 1))
        bins_y = np.linspace(min(cell_phis), max(cell_phis), int((max(cell_phis) - min(cell_phis)) / ((2*np.pi)/64) + 1))
        H_tot, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(cell_etas, cell_phis,
                                                            values=abs(cell_Esig),
                                                            bins=(bins_x,bins_y),
                                                            statistic='sum')
        H_tot = H_tot.T
        repeat_frac = 0.5
        repeat_rows = int(H_tot.shape[0]*repeat_frac)
        one_box_height = (yedges[-1]-yedges[0])/H_tot.shape[0]
        H_tot  = np.pad(H_tot, ((repeat_rows,repeat_rows),(0,0)),'wrap')
        truncation_threshold = 125
        H_tot = np.where(H_tot < truncation_threshold, H_tot, truncation_threshold)
        extent = (xedges[0],xedges[-1],yedges[0]-(repeat_rows*one_box_height),yedges[-1]+(repeat_rows*one_box_height))

        f,ax = plt.subplots()
        ii = ax.imshow(H_tot,cmap='binary_r',extent=extent,origin='lower')
        for bbx in tees:
            x,y=float(bbx[0]),float(bbx[1])
            w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])  
            bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1.75,ec='limegreen',fc='none')
            ax.add_patch(bb)

        for pred_box,pred_score in zip(pees,scores):
            x,y=float(pred_box[0]),float(pred_box[1])
            w,h=float(pred_box[2])-float(pred_box[0]),float(pred_box[3])-float(pred_box[1])  
            bb = matplotlib.patches.Rectangle((x,y),w,h,lw=1.5,ec='red',fc='none')
            ax.add_patch(bb)
        cbar = f.colorbar(ii,ax=ax)
        cbar.ax.get_yaxis().labelpad = 10
        cbar.set_label('cell significance', rotation=90)
        ax.set(xlabel='$\eta$',ylabel='$\phi$')
        ax.axhline(y=min(cells['cell_phi']), color='red', alpha=0.6, linestyle='--',lw=0.7)
        ax.axhline(y=max(cells['cell_phi']), color='red', alpha=0.6, linestyle='--',lw=0.7)
        # hep.atlas.label(ax=ax,label='Work in Progress',data=False,lumi=None,loc=0)
        f.savefig('/home/users/b/bozianu/work/SSD/SSD/cached_plots/SSD1_50k5_mu_15e/poster/evt-image-{}.{}'.format(i,image_format),dpi=400,format=image_format, bbox_inches="tight")




        quit()











        #extras
        are_pboxes_matched = [CalculateMatchFromBox(pb,tees,des_cells) for pb,des_cells in zip(pees,l_pred_cells)]
        are_tboxes_matched = [CalculateMatchFromBox(tb,pees,des_cells) for tb,des_cells in zip(tees,l_true_cells)]

        # check if tbox tb is "matched" to any box in pees
        necessary_results['tboxes_matched'].append(are_tboxes_matched)
        # check if pbox pb is "matched" to any box in tees
        necessary_results['pboxes_matched'].append(are_pboxes_matched)


        necessary_results['n_tboxes'].append(len(tees))
        necessary_results['num_tboxes'].append(len(tb_phys_dict['energy']))
        necessary_results['tboxes_energies'].append(tb_phys_dict['energy'])
        necessary_results['tboxes_eta'].append(tb_phys_dict['eta'])
        necessary_results['tboxes_phi'].append(tb_phys_dict['phi'])
        necessary_results['tboxes_eT'].append(tb_phys_dict['eT'])
        necessary_results['tboxes_n_cells'].append(tb_phys_dict['n_cells'])
        necessary_results['tboxes_noise'].append(tb_phys_dict['noise'])
        necessary_results['tboxes_significance'].append(tb_phys_dict['significance'])
        necessary_results['tboxes_neg_frac'].append(tb_phys_dict['neg_frac'])
        necessary_results['tboxes_max_frac'].append(tb_phys_dict['max_frac'])

        necessary_results['tboxes_energies_2sig'].append(tb_phys_dict['energy2sig'])
        necessary_results['tboxes_eT_2sig'].append(tb_phys_dict['eT2sig'])
        necessary_results['tboxes_n_cells_2sig'].append(tb_phys_dict['n_cells2sig'])

        necessary_results['n_pboxes'].append(len(pees))
        necessary_results['num_pboxes'].append(len(pb_phys_dict['energy']))
        necessary_results['pboxes_energies'].append(pb_phys_dict['energy'])
        necessary_results['pboxes_eta'].append(pb_phys_dict['eta'])
        necessary_results['pboxes_phi'].append(pb_phys_dict['phi'])
        necessary_results['pboxes_eT'].append(pb_phys_dict['eT'])
        necessary_results['pboxes_n_cells'].append(pb_phys_dict['n_cells'])
        necessary_results['pboxes_noise'].append(pb_phys_dict['noise'])
        necessary_results['pboxes_significance'].append(pb_phys_dict['significance'])
        necessary_results['pboxes_neg_frac'].append(pb_phys_dict['neg_frac'])
        necessary_results['pboxes_max_frac'].append(pb_phys_dict['max_frac']) 

        necessary_results['pboxes_energies_2sig'].append(pb_phys_dict['energy2sig'])
        necessary_results['pboxes_eT_2sig'].append(pb_phys_dict['eT2sig'])
        necessary_results['pboxes_n_cells_2sig'].append(pb_phys_dict['n_cells2sig'])

    save_loc = save_folder + f"/new_phys_metrics/smallposter/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    print('Saving the box metrics in lists...')
    for key, value in necessary_results.items():
        filename = f"{key}.pkl"
        # save_object(value, save_loc+filename)

    return None

if __name__=="__main__":
    folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD1_50k5_mu_15e/20231102-13/"
    save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD1_50k5_mu_15e/"

    print('Making poster plot info')
    make_poster_plot_data(folder_to_look_in,save_at)
    print('Completed poster plot info\n')







