from make_box_metrics import calculate_box_metrics
from make_phys_metrics import calculate_phys_metrics
from make_jet_metrics import calculate_jet_metrics



folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD_50k5_mu_20e/20231005-12/"
save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD_50k5_mu_20e/"


if __name__=="__main__":
    # print('Making box metrics')
    # calculate_box_metrics(folder_to_look_in,save_at)
    # print('Completed box metrics\n')

    # print('Making phys metrics')
    # calculate_phys_metrics(folder_to_look_in,save_at)
    # print('Completed [phys] metrics\n')

    print('Making jet metrics')
    calculate_jet_metrics(folder_to_look_in,save_at)
    print('Completed jet metrics\n')



