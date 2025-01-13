from make_box_metrics import calculate_box_metrics


folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD_raw_50k_mu_13e/20231130-09/"
save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD_raw_50k5_mu_13e"


if __name__=="__main__":
    print('Making box metrics')
    calculate_box_metrics(folder_to_look_in,save_at)
    print('Completed box metrics\n')





