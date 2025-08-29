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
    for chn in range(5):
        f,ax = plt.subplots()
        ii = ax.imshow(loaded_tensor[chn],cmap='binary_r')
        cbar = f.colorbar(ii,ax=ax)
        cbar.ax.get_yaxis().labelpad = 10
        cbar.set_label(f'cell var. {chn}', rotation=90)
        ax.set(xlabel='eta',ylabel='phi')
        f.savefig(f'examine-{chn}.png')
        plt.close()
    quit()

def make_lego_plot(H,xedges,yedges):

    # H = np.nan_to_num(H)
    H[H<0] = 0 # just for visualisation
    print(H.shape,xedges.shape,yedges.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x_widths = np.diff(xedges)
    y_widths = np.diff(yedges)

    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    dx = np.repeat(x_widths, len(y_widths))
    dy = np.tile(y_widths, len(x_widths))
    dz = H.ravel()

    cmap = matplotlib.cm.rainbow  # 'plasma', 'inferno', 'cividis', etc.
    norm = matplotlib.colors.Normalize(vmin=np.min(dz), vmax=np.max(dz))
    colors = cmap(norm(dz))
    colors = cmap((dz - np.min(dz)) / (np.ptp(dz)))

    print(xpos.shape, ypos.shape, zpos.shape, dx.shape, dy.shape, dz.shape)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='max')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$\phi$')
    ax.set_zlabel('Pixel PT')
    ax.set_xlim(-4.0, 4.0)  
    ax.set_ylim(-3.2, 3.2)  
    ax.xaxis._axinfo['grid']['linewidth'] = 0
    ax.yaxis._axinfo['grid']['linewidth'] = 0
    ax.zaxis._axinfo['grid']['linewidth'] = 0
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.set_zticks([])
    ax.set_zlabel('') 
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0.0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.0
    ax.zaxis.line.set_color((1, 1, 1, 0))  # fully transparent
    ax.zaxis.line.set_linewidth(0)
    ax.view_init(elev=25, azim=45) 

    plt.savefig('evt-lego-plot.png',dpi=400)
    quit()
    return 


def make_contour_plot(H,xedges,yedges):
    H[H<0] = 0 # just for visualisation
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    x, y = np.meshgrid(xcenters, ycenters, indexing='ij')  # shape matches H
    z = H  # shape (nx, ny)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    surf = ax.plot_surface(
        x, y, z,
        facecolors=matplotlib.cm.rainbow((z - np.min(z)) / (np.ptp(z))),
        linewidth=0,
        antialiased=False,
        shade=True
    )

    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$\phi$')
    ax.set_zlabel('Pixel PT')
    ax.set_xlim(-4.0, 4.0)  
    ax.set_ylim(-3.2, 3.2)  
    ax.xaxis._axinfo['grid']['linewidth'] = 0
    ax.yaxis._axinfo['grid']['linewidth'] = 0
    ax.zaxis._axinfo['grid']['linewidth'] = 0
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.set_zticks([])
    ax.set_zlabel('') 
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0.0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.0
    ax.zaxis.line.set_color((1, 1, 1, 0))  # fully transparent
    ax.zaxis.line.set_linewidth(0)

    ax.view_init(elev=25, azim=45) 
    plt.savefig('evt-contour-plot.png', dpi=400)
    quit()
    return



if __name__=="__main__":

    annotation_dict = {}
    annotation_dict_jet = {}
    global_counter = 0

    JZ_file_dict = {
        # # training files
        # "JZ2" : [,"000024","000025","000026",],
        # "JZ3" : ["000011","000012","000015","000016","000017","000018","000019","000021","000022","000023","000025","000026","000027","000028","000029","000030","000031","000032","000033","000034","000035","000036","000037","000038","000039","000040"],
        # "JZ4"   : ["000014","000015","000016","000017","000018","000019","000020","000021","000022","000023","000049","000050","000051","000052","000053","000054","000055","000056","000057","000058","000059","000060","000061","000062","000063","000064","000065","000066","000067","000068","000069","000070","000071","000072","000073","000074","000075","000076","000077","000078","000079","000080","000081","000083","000084"]
        # # test files
        "JZ0" : ["000001","000002","000003","000004","000005","000006","000007","000008","000009","000010","000011","000012","000013","000014","000015","000016","000017","000018","000019","000020","000021","000022","000023"],
    }

    JZ_grid_dict = {
        "JZ0" : "46057987",
        "JZ2" : "43186502",
        "JZ3" : "43186499",
        "JZ4" : "43589851",
    }

    for proc in ["JZ0"]:
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
                
                    # make_lego_plot(H_sum_pt, xedges, yedges) # plot if want
                    # make_contour_plot(H_sum_pt, xedges, yedges) 
                    
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
                    torch.save(H_layers_tensor,args.output_dir+"/{}/cell-img-{}.pt".format(proc,unique_file_chunk_event_no))

                    # examine_one_image(args.output_dir+"/{}/cell-img-{}.pt".format(proc,unique_file_chunk_event_no))
                    

                    global_counter += 1
                chunk_counter += 1



