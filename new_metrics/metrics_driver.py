from make_box_metrics import calculate_box_metrics




folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD_model_25_large_mu/20230908-12/"
save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD_model_25_large_mu/"


if __name__=="__main__":
    print('Making box metrics')
    calculate_box_metrics(folder_to_look_in,save_at,save_at)
    print('Completed plots about 1 event\n')



