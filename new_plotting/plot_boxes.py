import numpy as np 
import os

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.pyplot as plt
import matplotlib

import mplhep as hep
hep.style.use(hep.style.ATLAS)




def load_object(fname):
    with open(fname,'rb') as file:
        return pickle.load(file)

def is_nested_list(metric):
    #check if we have list or list of list of lists
    if isinstance(metric, list):
        return isinstance(metric[0], list) if metric else False

def get_ratio(numer,denom):
    result = np.zeros_like(numer, dtype=float)
    non_zero_indices = denom != 0
    # Perform element-wise division, handling zeros in the denominator
    result[non_zero_indices] = numer[non_zero_indices] / denom[non_zero_indices]
    return result

list_metric_names = ['delta_n',
                     'n_matched_preds',
                     'n_matched_truth',
                     'n_preds',
                     'n_truth',
                     'n_unmatched_preds',
                     'n_unmatched_truth',
                     'percentage_total_area_covered_preds',
                     'percentage_total_area_covered_truth',
                     'percentage_truth_area_covered'
                     ]

list_x_labels = ['Number Truth Boxes - Number Predicted',
                 'Number of matched prediction boxes',
                 'Number of matched truth boxes (TP)',
                 'Number of predicted boxes',
                 'Number of true boxes',
                 'Number of Truth Boxes Missed (FN)',
                 'Number of unmatched/fake prediction boxes (FP)',
                 'Percentage of calorimeter covered by predicted boxes (%)',   
                 'Percentage of calorimeter covered by truth boxes (%)',   
                 'Percentage of matched truth boxes covered by predictions (%)'   
                 ]


def make_box_plots(
    boxes_folder,
    physics_folder,
    save_folder,
    log  = True,
    image_format = "png",
):

    save_loc = save_folder + f"/boxes/new/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)


    #####################################################################################################################################
    #Plot 1, the number of clusters, truth and pred boxes
    n_truth = load_object(boxes_folder + '/n_truth.pkl')
    n_preds = load_object(boxes_folder + '/n_preds.pkl')
    n_cl = load_object(physics_folder + '/n_clusters.pkl')

    f,ax = plt.subplots(2,1,figsize=(9, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    freq_cl, bins, _   = ax[0].hist(n_cl,bins=50,density=(not log),histtype='step',color='tab:blue',lw=1.5,label='TopoCl >5GeV {:.2f}$\pm${:.1f}'.format(np.mean(n_cl),np.std(n_cl)))
    freq_tru, bins, _ = ax[0].hist(n_truth,bins=bins,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes {:.2f}$\pm${:.1f}'.format(np.mean(n_truth),np.std(n_truth)))
    freq_pred, _, _   = ax[0].hist(n_preds,bins=bins,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes {:.2f}$\pm${:.1f}'.format(np.mean(n_preds),np.std(n_preds)))
    
    ax[0].grid()
    ax[0].set_title('Number of clusters/boxes per event', fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.6, 0.5),fontsize="x-small")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_cl),marker='_',color='red',s=50)
    ax[1].scatter(bin_centers, get_ratio(freq_tru,freq_cl),marker='_',color='green',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
    ax[1].set(xlabel="# Clusters",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log',ylabel='Freq.')
        f.savefig(save_loc + f'/n_clusters_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + f'/n_clusters.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    #####################################################################################################################################
    #Plot 2, the number of matched clusters, truth and pred boxes
    n_match_truth = load_object(boxes_folder + '/n_matched_truth.pkl')
    n_match_preds = load_object(boxes_folder + '/n_matched_preds.pkl')

    f,ax = plt.subplots(2,1,figsize=(9, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    freq_cl, bins, _   = ax[0].hist(n_cl,bins=50,density=(not log),histtype='step',color='tab:blue',lw=1.5,label='TopoCl >5GeV {:.2f}$\pm${:.1f}'.format(np.mean(n_cl),np.std(n_cl)))
    freq_tru, bins, _ = ax[0].hist(n_match_truth,bins=bins,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes {:.2f}$\pm${:.1f}'.format(np.mean(n_match_truth),np.std(n_match_truth)))
    freq_pred, _, _   = ax[0].hist(n_match_preds,bins=bins,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes {:.2f}$\pm${:.1f}'.format(np.mean(n_match_preds),np.std(n_match_preds)))
    
    ax[0].grid()
    ax[0].set_title('Number of Matched boxes per event', fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.6, 0.5),fontsize="x-small")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_cl),marker='_',color='red',s=50)
    ax[1].scatter(bin_centers, get_ratio(freq_tru,freq_cl),marker='_',color='green',s=50)
    ax[1].axhline(1,ls='--',color='tab:blue',alpha=0.5)
    ax[1].set(xlabel="# Matched Boxes/Clusters",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log',ylabel='Freq.')
        f.savefig(save_loc + f'/n_match_clusters_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + f'/n_match_clusters.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    #####################################################################################################################################
    #Plot 3, the number of unmatched clusters, truth and pred boxes
    n_unmatch_truth = load_object(boxes_folder + '/n_unmatched_truth.pkl')
    n_unmatch_preds = load_object(boxes_folder + '/n_unmatched_preds.pkl')

    f,ax = plt.subplots(2,1,figsize=(9, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    freq_tru, bins, _ = ax[0].hist(n_unmatch_truth,bins=50,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes {:.2f}$\pm${:.1f}'.format(np.mean(n_unmatch_truth),np.std(n_unmatch_truth)))
    freq_pred, _, _   = ax[0].hist(n_unmatch_preds,bins=bins,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes {:.2f}$\pm${:.1f}'.format(np.mean(n_unmatch_preds),np.std(n_unmatch_preds)))
    
    ax[0].grid()
    ax[0].set_title('Number of Unmatched boxes per event', fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.65, 0.5),fontsize="x-small")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="# Unmatched Boxes",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',data=False,lumi=None,loc=1)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log',ylabel='Freq.')
        f.savefig(save_loc + f'/n_unmatch_clusters_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + f'/n_unmatch_clusters.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()


    #####################################################################################################################################
    #Plot 4, the percentage area of calorimeter covered
    perc_area_truth = load_object(boxes_folder + '/percentage_total_area_covered_truth.pkl')
    perc_area_preds = load_object(boxes_folder + '/percentage_total_area_covered_preds.pkl')

    f,ax = plt.subplots(2,1,figsize=(7.5, 7.5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    freq_tru, bins, _ = ax[0].hist(perc_area_truth,bins=50,density=(not log),histtype='step',color='green',lw=1.5,label='Truth Boxes ({:.2f}$\pm${:.1f})'.format(np.mean(perc_area_truth),np.std(perc_area_truth)))
    freq_pred, _, _   = ax[0].hist(perc_area_preds,bins=bins,density=(not log),histtype='step',color='red',lw=1.5,label='Predicted Boxes ({:.2f}$\pm${:.1f})'.format(np.mean(perc_area_preds),np.std(perc_area_preds)))
    
    ax[0].grid()
    ax[0].set_title('% of total calorimeter covered', fontsize=16, fontfamily="TeX Gyre Heros")
    ax[0].legend(loc='lower left',bbox_to_anchor=(0.4, 0.4),fontsize="medium")

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50,lw=1.5)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Percentage calorimeter area covered",ylabel='Ratio')
    ax[1].grid()
    hep.atlas.label(ax=ax[0],label='Work in Progress',loc=1,data=False,lumi=None)
    f.subplots_adjust(hspace=0.075)
    if log:
        ax[0].set(yscale='log',ylabel='Freq.')
        f.savefig(save_loc + f'/per_area_covered_log.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    else:
        ax[0].set(ylabel='Freq. Density')
        f.savefig(save_loc + f'/per_area_covered.{image_format}',dpi=400,format=image_format,bbox_inches="tight")
    plt.close()



    #####################################################################################################################################
    #####################################################################################################################################
    #####################################################################################################################################
    #Binned Plots
    bind_save_loc = save_loc + "binned"
    if not os.path.exists(bind_save_loc):
        os.makedirs(bind_save_loc)    


    n_truth_arr = np.array(n_truth)

    n_match_truth_arr = np.array(n_match_truth)
    n_match_preds_arr = np.array(n_match_preds)
    binnings = [0,30,50,80,100,200]
    numbers_of_bins = [15,25,35,40,50]
    for i in range(len(binnings)-1):
        clusters_mask = (n_truth_arr > binnings[i]) & (n_truth_arr < binnings[i+1])

        n_m_truth_many = n_match_truth_arr[clusters_mask]
        n_m_preds_many = n_match_preds_arr[clusters_mask]

        f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        freq_tru, bins, _ = ax[0].hist(n_m_truth_many,bins=int(len(n_m_truth_many)/20),range=(0,max(n_m_truth_many)),histtype='step',lw=1.5,color='green',label='Truth Boxes ({:.2f}$\pm${:.1f})'.format(np.mean(n_m_truth_many),np.std(n_m_truth_many)))
        freq_pred, _, _ = ax[0].hist(n_m_preds_many,bins=bins,histtype='step',color='red',lw=1.5,label='Predictions {:.2f}$\pm${:.1f}'.format(np.mean(n_m_preds_many),np.std(n_m_preds_many)))
        ax[0].grid()
        ax[0].set(ylabel='Freq.',title='{} Events, [{},{}] true objects '.format(len(n_m_truth_many),binnings[i],binnings[i+1]))
        ax[0].legend()

        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
        ax[1].axhline(1,ls='--',color='green',alpha=0.5)
        ax[1].set(xlabel="Number of Matched Boxes (per event)",ylabel='Ratio')
        ax[1].grid()
        f.subplots_adjust(hspace=0)
        f.savefig(bind_save_loc + '/matched_n_boxes_bin{}.png'.format(i))
        plt.close()


    

    n_unmatch_truth_arr = np.array(n_unmatch_truth)
    n_unmatch_preds_arr = np.array(n_unmatch_preds)
    binnings = [0,30,50,80,100,200]
    for i in range(len(binnings)-1):
        clusters_mask = (n_truth_arr > binnings[i]) & (n_truth_arr < binnings[i+1])

        n_unm_truth_many = n_unmatch_truth_arr[clusters_mask]
        n_unm_preds_many = n_unmatch_preds_arr[clusters_mask]

        f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        freq_tru, bins, _ = ax[0].hist(n_unm_truth_many,bins=int(len(n_unm_truth_many)/20),range=(0,max(n_unm_truth_many)),histtype='step',lw=1.5,color='green',label='Truth Boxes ({:.2f}$\pm${:.1f})'.format(np.mean(n_unm_truth_many),np.std(n_unm_truth_many)))
        freq_pred, _, _ = ax[0].hist(n_unm_preds_many,bins=bins,histtype='step',lw=1.5,color='red',label='Predicted Boxes ({:.2f}$\pm${:.1f})'.format(np.mean(n_unm_preds_many),np.std(n_unm_preds_many)))
        ax[0].grid()
        ax[0].set(ylabel='Freq.',title='{} Events, [{},{}] true objects '.format(len(n_unm_truth_many),binnings[i],binnings[i+1]))
        ax[0].legend()

        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
        ax[1].axhline(1,ls='--',color='green',alpha=0.5)
        ax[1].set(xlabel="Number of Unmatched Boxes (per event)",ylabel='Ratio')
        ax[1].grid()
        f.subplots_adjust(hspace=0)
        f.savefig(bind_save_loc + '/unmatched_n_boxes_bin{}.png'.format(i))
        plt.close()









def make_box_metric_plots(
    folder_containing_lists,
    save_folder,
):
    save_loc = save_folder + "/boxes/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    
    for j in range(len(list_metric_names)):
        event_level_metric = load_object(folder_containing_lists + "/{}.pkl".format(list_metric_names[j]))

        itle = f"{len(event_level_metric)} events"
        label = "{:.2f}$\pm${:.1f}".format(np.mean(event_level_metric),np.std(event_level_metric))

        fig, ax = plt.subplots()
        ax.hist(event_level_metric,density=True,histtype='step',bins=50,label=label) 
        ax.set(xlabel=list_x_labels[j],ylabel='Freq. Density',title=itle)
        ax.grid(True)
        ax.legend()
        fig.savefig(save_loc+"{}.png".format(list_metric_names[j]))
        plt.close() 



    comp_save_loc = save_loc + "comp"
    if not os.path.exists(comp_save_loc):
        os.makedirs(comp_save_loc)
    #####################################################################################################################################
    #Plot 1, the number of boxes predicted and true
    n_truth = load_object(folder_containing_lists + '/n_truth.pkl')
    n_preds = load_object(folder_containing_lists + '/n_preds.pkl')

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    freq_tru, bins, _ = ax[0].hist(n_truth,bins=50,range=(0,max(n_truth)),histtype='step',color='green',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(n_truth),np.std(n_truth)))
    freq_pred, _, _ = ax[0].hist(n_preds,bins=bins,histtype='step',color='red',label='Predictions {:.2f}$\pm${:.1f}'.format(np.mean(n_preds),np.std(n_preds)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='{} Events'.format(len(n_truth)),yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Total Number of Boxes (per event)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(comp_save_loc + '/total_n_boxes2.png')
    plt.close()

    #####################################################################################################################################
    #Plot 1, the number of boxes predicted and true
    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    freq_tru, bins, _ = ax[0].hist(n_truth,bins=50,range=(0,max(n_truth)),density=True,histtype='step',color='green',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(n_truth),np.std(n_truth)))
    freq_pred, _, _ = ax[0].hist(n_preds,bins=bins,density=True,histtype='step',color='red',label='Predictions {:.2f}$\pm${:.1f}'.format(np.mean(n_preds),np.std(n_preds)))
    ax[0].grid()
    ax[0].set(ylabel='Freq. Density',title='{} Events'.format(len(n_truth)))
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Total Number of Boxes (per event)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(comp_save_loc + '/total_n_boxes.png')
    plt.close()

    #####################################################################################################################################
    #Plot 2, the number of unmatched predicted and true boxes
    n_unmatch_truth = load_object(folder_containing_lists + '/n_unmatched_truth.pkl')
    n_unmatch_preds = load_object(folder_containing_lists + '/n_unmatched_preds.pkl')

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    freq_tru, bins, _ = ax[0].hist(n_unmatch_truth,bins=50,range=(0,max(n_unmatch_truth)),histtype='step',color='green',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(n_unmatch_truth),np.std(n_unmatch_truth)))
    freq_pred, _, _ = ax[0].hist(n_unmatch_preds,bins=bins,histtype='step',color='red',label='Predictions {:.2f}$\pm${:.1f}'.format(np.mean(n_unmatch_preds),np.std(n_unmatch_preds)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='{} Events'.format(len(n_unmatch_truth)),yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Number of Unmatched Boxes (per event)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(comp_save_loc + '/unmatched_n_boxes2.png')
    plt.close()

    #####################################################################################################################################
    #Plot 2b, the number of unmatched predicted and true boxes
    # f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    f,ax = plt.subplots(1,1,figsize=(8, 6))
    freq_tru, bins, _ = ax.hist(n_unmatch_truth,bins=50,range=(0,max(n_unmatch_truth)),density=True,histtype='step',color='green',label='Missed Truth {:.2f}$\pm${:.1f}'.format(np.mean(n_unmatch_truth),np.std(n_unmatch_truth)))
    freq_pred, _, _ = ax.hist(n_unmatch_preds,bins=bins,density=True,histtype='step',color='red',label='Fake Predictions {:.2f}$\pm${:.1f}'.format(np.mean(n_unmatch_preds),np.std(n_unmatch_preds)))
    ax.grid()
    ax.set(ylabel='Freq. Density',title=r'$\bf{Unmatched}$ $\bf{Boxes}$',xlabel="Number of Unmatched Boxes (per event)")
    ax.legend()

    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
    # ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    # ax[1].set(xlabel="Number of Unmatched Boxes (per event)",ylabel='Ratio')
    # ax[1].grid()
    # f.subplots_adjust(hspace=0)
    f.savefig(comp_save_loc + '/unmatched_n_boxes.png')
    plt.close()

    #####################################################################################################################################
    #Plot 3, the number of matched predicted and true boxes
    n_match_truth = load_object(folder_containing_lists + '/n_matched_truth.pkl')
    n_match_preds = load_object(folder_containing_lists + '/n_matched_preds.pkl')

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    freq_tru, bins, _ = ax[0].hist(n_match_truth,bins=50,range=(0,max(n_match_truth)),histtype='step',color='green',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(n_match_truth),np.std(n_match_truth)))
    freq_pred, _, _ = ax[0].hist(n_match_preds,bins=bins,histtype='step',color='red',label='Predictions {:.2f}$\pm${:.1f}'.format(np.mean(n_match_preds),np.std(n_match_preds)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='{} Events'.format(len(n_match_truth)),yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Number of Matched Boxes (per event)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(comp_save_loc + '/matched_n_boxes2.png')
    plt.close()

    #####################################################################################################################################
    #Plot 3b, the number of matched predicted and true boxes
    # f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    f,ax = plt.subplots(1,1,figsize=(8, 6))
    freq_tru, bins, _ = ax.hist(n_match_truth,bins=50,range=(0,max(n_match_truth)),density=True,histtype='step',color='green',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(n_match_truth),np.std(n_match_truth)))
    freq_pred, _, _ = ax.hist(n_match_preds,bins=bins,density=True,histtype='step',color='red',label='Predictions {:.2f}$\pm${:.1f}'.format(np.mean(n_match_preds),np.std(n_match_preds)))
    ax.grid()
    ax.set(ylabel='Freq. Density',title=r'$\bf{Matched}$ $\bf{Boxes}$',xlabel="Number of Matched Boxes (per event)")
    ax.legend()

    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
    # ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    # ax[1].set(xlabel="Number of Matched Boxes (per event)",ylabel='Ratio')
    # ax[1].grid()
    # f.subplots_adjust(hspace=0)
    f.savefig(comp_save_loc + '/matched_n_boxes.png')
    plt.close()


    #####################################################################################################################################
    #Plot 4, percentage of calo covered by truth and predictions
    perc_total_area_truth = load_object(folder_containing_lists + '/percentage_total_area_covered_truth.pkl')
    perc_total_area_preds = load_object(folder_containing_lists + '/percentage_total_area_covered_preds.pkl')

    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    freq_tru, bins, _ = ax[0].hist(perc_total_area_truth,bins=50,range=(0,1),histtype='step',color='green',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(perc_total_area_truth),np.std(perc_total_area_truth)))
    freq_pred, _, _ = ax[0].hist(perc_total_area_preds,bins=bins,histtype='step',color='red',label='Predictions {:.2f}$\pm${:.1f}'.format(np.mean(perc_total_area_preds),np.std(perc_total_area_preds)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='{} Events'.format(len(perc_total_area_truth)),yscale='log')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Percentage of calorimeter covered by boxes (%)",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(comp_save_loc + '/percentage_total_covered2.png')
    plt.close()

    #####################################################################################################################################
    #Plot 4b, percentage of calo covered by truth and predictions
    f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
    freq_tru, bins, _ = ax[0].hist(perc_total_area_truth,bins=50,range=(0,1),density=True,histtype='step',color='green',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(perc_total_area_truth),np.std(perc_total_area_truth)))
    freq_pred, _, _ = ax[0].hist(perc_total_area_preds,bins=bins,density=True,histtype='step',color='red',label='Predictions {:.2f}$\pm${:.1f}'.format(np.mean(perc_total_area_preds),np.std(perc_total_area_preds)))
    ax[0].grid()
    ax[0].set(ylabel='Freq.',title='Percentage of total calorimeter covered by boxes')
    ax[0].legend()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
    ax[1].axhline(1,ls='--',color='green',alpha=0.5)
    ax[1].set(xlabel="Fraction of calorimeter",ylabel='Ratio')
    ax[1].grid()
    f.subplots_adjust(hspace=0)
    f.savefig(comp_save_loc + '/percentage_total_covered.png')
    plt.close()


    #####################################################################################################################################
    #plot 5 number of unmatched boxes when n_truth boxes <50
    #####################################################################################################################################
    #Binned Plots
    bind_save_loc = save_loc + "binned"
    if not os.path.exists(bind_save_loc):
        os.makedirs(bind_save_loc)    
    
    n_truth_arr = np.array(n_truth)
    n_unmatch_truth_arr = np.array(n_unmatch_truth)
    n_unmatch_preds_arr = np.array(n_unmatch_preds)
    binnings = [0,30,50,80,100,200]
    numbers_of_bins = [15,25,35,40,50]
    for i in range(len(binnings)-1):
        # clusters_mask = n_truth_arr > binnings[i]
        clusters_mask = (n_truth_arr > binnings[i]) & (n_truth_arr < binnings[i+1])

        n_unm_truth_many = n_unmatch_truth_arr[clusters_mask]
        n_unm_preds_many = n_unmatch_preds_arr[clusters_mask]

        f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        freq_tru, bins, _ = ax[0].hist(n_unm_truth_many,bins=numbers_of_bins[i],range=(0,max(n_unm_truth_many)),histtype='step',color='green',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(n_unm_truth_many),np.std(n_unm_truth_many)))
        freq_pred, _, _ = ax[0].hist(n_unm_preds_many,bins=bins,histtype='step',color='red',label='Predictions {:.2f}$\pm${:.1f}'.format(np.mean(n_unm_preds_many),np.std(n_unm_preds_many)))
        ax[0].grid()
        ax[0].set(ylabel='Freq.',title='{} Events, [{},{}] true objects '.format(len(n_unm_truth_many),binnings[i],binnings[i+1]))
        ax[0].legend()

        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
        ax[1].axhline(1,ls='--',color='green',alpha=0.5)
        ax[1].set(xlabel="Number of Unmatched Boxes (per event)",ylabel='Ratio')
        ax[1].grid()
        f.subplots_adjust(hspace=0)
        f.savefig(bind_save_loc + '/unmatched_n_boxes_bin{}.png'.format(i))
        plt.close()
    
    #####################################################################################################################################
    #Binned Plots

    n_match_truth_arr = np.array(n_match_truth)
    n_match_preds_arr = np.array(n_match_preds)
    binnings = [0,30,50,80,100,200]
    numbers_of_bins = [15,25,35,40,50]
    for i in range(len(binnings)-1):
        clusters_mask = (n_truth_arr > binnings[i]) & (n_truth_arr < binnings[i+1])

        n_m_truth_many = n_match_truth_arr[clusters_mask]
        n_m_preds_many = n_match_preds_arr[clusters_mask]

        f,ax = plt.subplots(2,1,figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        freq_tru, bins, _ = ax[0].hist(n_m_truth_many,bins=numbers_of_bins[i],range=(0,max(n_m_truth_many)),histtype='step',color='green',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(n_m_truth_many),np.std(n_m_truth_many)))
        freq_pred, _, _ = ax[0].hist(n_m_preds_many,bins=bins,histtype='step',color='red',label='Predictions {:.2f}$\pm${:.1f}'.format(np.mean(n_m_preds_many),np.std(n_m_preds_many)))
        ax[0].grid()
        ax[0].set(ylabel='Freq.',title='{} Events, [{},{}] true objects '.format(len(n_m_truth_many),binnings[i],binnings[i+1]))
        ax[0].legend()

        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax[1].scatter(bin_centers, get_ratio(freq_pred,freq_tru),marker='_',color='red',s=50)
        ax[1].axhline(1,ls='--',color='green',alpha=0.5)
        ax[1].set(xlabel="Number of Matched Boxes (per event)",ylabel='Ratio')
        ax[1].grid()
        f.subplots_adjust(hspace=0)
        f.savefig(bind_save_loc + '/matched_n_boxes_bin{}.png'.format(i))
        plt.close()

    #####################################################################################################################################
    #Binned Plots

    perc_truth_area = load_object(folder_containing_lists + '/percentage_truth_area_covered.pkl')
    perc_truth_area_arr = np.array(perc_truth_area)
    binnings = [0,30,50,80,100,200]

    fig,ax = plt.subplots()
    for i in range(len(binnings)-1):
        clusters_mask = (n_truth_arr > binnings[i]) & (n_truth_arr < binnings[i+1])

        mask_perc_truth_area_arr = perc_truth_area_arr[clusters_mask]
        ax.hist(mask_perc_truth_area_arr,bins=40,range=(0,1),density=True,histtype='step',label='[{},{}] true objects, {:.3f}'.format(binnings[i],binnings[i+1],np.mean(mask_perc_truth_area_arr)))
    
    ax.hist(perc_truth_area_arr,bins=40,range=(0,1),density=True,histtype='step',lw=0.75,color='black',label='Overall {} events, {:.3f}'.format(len(perc_truth_area_arr),np.mean(perc_truth_area_arr)))
    ax.grid()
    ax.set(xlabel='Fraction of matched truth boxes covered by predictions',ylabel='Freq. Density')
    ax.legend(fontsize='small', frameon=False)
    fig.savefig(bind_save_loc + '/perc_truth_cover_bin{}.png'.format(i))
    plt.close()

    



    print('Finished making metric plots.')
    return 




#make a plot of the number of truth boxes, number predicted boxes
#jaccard index


folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD1_50k5_mu_15e/box_metrics/"
save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_plots/SSD1_50k5_mu_15e/"
box_folder = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD1_50k5_mu_15e/box_metrics/"
physics_folder = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD1_50k5_mu_15e/new_phys_metrics/total/"

if __name__=="__main__":
    print('Making plots about boxes')
    make_box_plots(box_folder,physics_folder,save_at,log=True)
    make_box_plots(box_folder,physics_folder,save_at,log=False)
    # make_box_plots(box_folder,physics_folder,save_at,log=True,image_format='pdf')
    # make_box_plots(box_folder,physics_folder,save_at,log=False,image_format='pdf')
    # make_box_metric_plots(folder_to_look_in,save_at)
    print('Completed plots about boxes\n')
















    # #####################################################################################################################################
    # #Plot 4, number of matched truth boxes
    # f,ax = plt.subplots(1,1,figsize=(8, 6))
    # ax.hist(n_match_truth,bins=50,density=True,histtype='step',color='green',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(n_match_truth),np.std(n_match_truth)))
    # ax.grid()
    # ax.set(xlabel='Number of matched truth boxes (TP)',ylabel='Freq. Density',title='{} Events'.format(len(n_match_truth)))
    # ax.legend()
    # f.savefig(comp_save_loc + '/matched_truths.png')
    # plt.close()

    # #####################################################################################################################################
    # #Plot 5, number of matched pred boxes
    # f,ax = plt.subplots(1,1,figsize=(8, 6))
    # ax.hist(n_match_preds,bins=50,density=True,histtype='step',color='red',label='Truth {:.2f}$\pm${:.1f}'.format(np.mean(n_match_preds),np.std(n_match_preds)))
    # ax.grid()
    # ax.set(xlabel='Number of matched prediction boxes',ylabel='Freq. Density',title='{} Events'.format(len(n_match_preds)))
    # ax.legend()
    # f.savefig(comp_save_loc + '/matched_preds.png')
    # plt.close()