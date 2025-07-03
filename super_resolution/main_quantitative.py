import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Build_list_data_prepare.Build_list as Build_list
import CMR_HFpEF_Analysis.Image_utils as util

import os
import numpy as np
import nibabel as nb
import pandas as pd

cg = Defaults.Parameters()
def measure(pred_img, gt_img):
    lv_dice_optimized, ds, pred_optimize = util.Dice_optim(pred_img, gt_img, [1,2], d = 4)

    # Dice
    # calculate dice 
    lv_dice = ff.np_categorical_dice(pred_img, gt_img, 1)
    myo_dice = ff.np_categorical_dice(pred_img, gt_img, 2)
    # dice for optmized prediction
    lv_dice_optimized = ff.np_categorical_dice(pred_optimize, gt_img, 1)
    myo_dice_optimized = ff.np_categorical_dice(pred_optimize, gt_img, 2)

    # HD
    gt_lv_pixels = np.asarray(ff.count_pixel(gt_img,1)[1]); gt_myo_pixels = np.asarray(ff.count_pixel(gt_img,2)[1])
    pred_lv_pixels = np.asarray(ff.count_pixel(pred_img,1)[1]); pred_myo_pixels = np.asarray(ff.count_pixel(pred_img,2)[1])
    pred_lv_pixels_optimized = np.asarray(ff.count_pixel(pred_optimize,1)[1]); pred_myo_pixels_optimized = np.asarray(ff.count_pixel(pred_optimize,2)[1])
    # calculate HD 
    lv_hd = ff.HD(pred_lv_pixels, gt_lv_pixels, 1.0, min = True)
    myo_hd = ff.HD( pred_myo_pixels, gt_myo_pixels,1.0, min = True)
    lv_hd_one_direct = ff.HD(pred_lv_pixels, gt_lv_pixels, 1.0, min = False)
    myo_hd_one_direct = ff.HD( pred_myo_pixels, gt_myo_pixels,1.0, min = False)    
    # HD for optimized prediction
    lv_hd_optimized = ff.HD(pred_lv_pixels_optimized, gt_lv_pixels, 1.0, min = True)
    myo_hd_optimized = ff.HD( pred_myo_pixels_optimized, gt_myo_pixels,1.0, min = True)
    lv_hd_one_direct_optimized = ff.HD(pred_lv_pixels_optimized, gt_lv_pixels, 1.0, min = False)
    myo_hd_one_direct_optimized = ff.HD( pred_myo_pixels_optimized, gt_myo_pixels,1.0, min = False)

    # print all results including Dice, dice_optimized, HD, HD_optimized, HD_one_direct, HD_one_direct_optimized
    print(lv_dice, myo_dice, lv_dice_optimized, myo_dice_optimized, lv_hd, myo_hd, lv_hd_optimized, myo_hd_optimized, lv_hd_one_direct, myo_hd_one_direct, lv_hd_one_direct_optimized, myo_hd_one_direct_optimized)
    return lv_dice, myo_dice, lv_dice_optimized, myo_dice_optimized, lv_hd, myo_hd, lv_hd_optimized, myo_hd_optimized, lv_hd_one_direct, myo_hd_one_direct, lv_hd_one_direct_optimized, myo_hd_one_direct_optimized

trial_name = 'EDSR_LVmyo_motion_new'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_simulated_data_15normals.xlsx')
save_folder = os.path.join(cg.predict_dir, trial_name, 'images')

# build list
b = Build_list.Build(data_sheet)
patient_id_list,patient_tf_list,motion_name_list,_, _,y_list, _,x_list,_,_, _, _ = b.__build__(batch_list = [5])
n = np.arange(0,patient_id_list.shape[0],1)
x_list = x_list[n]; y_list = y_list[n]

# main script
Results = []
for i in range(0,x_list.shape[0]):
    patient_id = patient_id_list[n[i]] 
    patient_tf  = patient_tf_list[n[i]]
    motion_name = motion_name_list[n[i]]
    print(patient_id, patient_tf, motion_name)

    # load ground truth HR data
    gt = nb.load(os.path.join(cg.data_dir, 'contour_dataset/processed_HR_data', str(patient_id), patient_tf,  'HR_ED_zoomed_crop_flip_clean.nii.gz'))
    gt_img = gt.get_fdata();  gt_img = np.round(gt_img); gt_img = gt_img.astype(int); gt_img = util.relabel(gt_img, 4, 0)
    
    # load all predictions
    pred_files = ff.find_all_target_files(['pred_img_HR_*'],os.path.join(save_folder, str(patient_id), patient_tf, motion_name))
    case_results = []
    for k in range(0,len(pred_files)):
        pred = nb.load(pred_files[k])
        pred_img = pred.get_fdata(); pred_img = np.round(pred_img); pred_img = pred_img.astype(int); pred_img = util.relabel(pred_img, 4, 0)
        lv_dice, myo_dice, lv_dice_optimized, myo_dice_optimized, lv_hd, myo_hd, lv_hd_optimized, myo_hd_optimized, lv_hd_one_direct, myo_hd_one_direct, lv_hd_one_direct_optimized, myo_hd_one_direct_optimized = measure(pred_img, gt_img)
        case_results.append([lv_dice, myo_dice, lv_dice_optimized, myo_dice_optimized, lv_hd, myo_hd, lv_hd_optimized, myo_hd_optimized, lv_hd_one_direct, myo_hd_one_direct, lv_hd_one_direct_optimized, myo_hd_one_direct_optimized])
    case_results = np.asarray(case_results)
    # pick the best result
    best_results = np.zeros((case_results.shape[1]))
    for kk in range(0,4):
        best_results[kk] = np.max(case_results[:,kk])
    for kk in range(4,case_results.shape[1]):
        best_results[kk] = np.min(case_results[:,kk])

    # print all results including Dice, dice_optimized, HD, HD_optimized, HD_one_direct, HD_one_direct_optimized
    print(best_results[0], best_results[1], best_results[2], best_results[3], best_results[4], best_results[5], best_results[6], best_results[7], best_results[8], best_results[9], best_results[10], best_results[11])

    Results.append([patient_id, patient_tf, motion_name] + list(best_results))

    columns = ['patient_id', 'patient_tf', 'motion_name', 'lv_dice', 'myo_dice', 'lv_dice_optimized', 'myo_dice_optimized', 'lv_hd', 'myo_hd', 'lv_hd_optimized', 'myo_hd_optimized', 'lv_hd_one_direct', 'myo_hd_one_direct', 'lv_hd_one_direct_optimized', 'myo_hd_one_direct_optimized']
    df = pd.DataFrame(Results, columns = columns)
    df.to_excel(os.path.join(os.path.dirname(save_folder), 'comparison_img.xlsx'), index = False)


