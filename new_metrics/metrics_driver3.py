from make_jet_metrics import calculate_jet_metrics

folder_to_look_in = "/home/users/b/bozianu/work/SSD/SSD/cached_inference/SSD1_50k5_mu_15e/20231102-13/"
save_at = "/home/users/b/bozianu/work/SSD/SSD/cached_metrics/SSD1_50k5_mu_15e"

if __name__=="__main__":
    print('Making jet metrics')
    calculate_jet_metrics(folder_to_look_in,save_at)
    print('Completed jet metrics\n')



