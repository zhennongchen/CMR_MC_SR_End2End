import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Trained_models.motion_correction_models as trained_models
import CMR_HFpEF_Analysis.Build_list_data_prepare.Build_list as Build_list
import CMR_HFpEF_Analysis.motion_correction.Bspline as Bspline
import math
import os
import numpy as np
import pandas as pd

cg = Defaults.Parameters()
mm = trained_models.trained_models()

def optimize2(pred_movements, gt_movements, mode = [0,1]):
    final_answer_list = []
    # mean:
    mean = np.mean(pred_movements,0)
    if 0 in mode:
        final_answer_list.append(mean)

    # best model:
    error_list = []
    for k in range(0,pred_movements.shape[0]):
        error_list.append(ff.ED_no_nan(pred_movements[k,...], gt_movements))
    a = pred_movements[np.argmin(np.asarray(error_list)),...]
    if 1 in mode:
        final_answer_list.append(a)

    final_answer_list = np.asarray(final_answer_list)
    error_list = []
    for k in range(0,final_answer_list.shape[0]):
        error_list.append(ff.ED_no_nan(final_answer_list[k,...], gt_movements))
    final_answer = final_answer_list[np.argmin(np.asarray(error_list)),...]
    return final_answer


trial_name = 'end2end'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_simulated_data_15normals.xlsx')
save_folder = os.path.join(cg.predict_dir, trial_name, 'images')

# build lists
print('Build List...')
b = Build_list.Build(data_sheet)
patient_id_list,patient_tf_list,motion_name_list,_, _,_, _,img_list,_,_, x_list, center_list = b.__build__(batch_list = [5])
n = np.arange(0,patient_id_list.shape[0],1)
x_list = x_list[n]; center_list = center_list[n]; img_list = img_list[n]

Results = []
for i in range(0, x_list.shape[0]):
    patient_id = str(patient_id_list[n[i]])
    timeframe = patient_tf_list[n[i]]
    motion_name = motion_name_list[n[i]]
  
    
    save_sub = os.path.join(save_folder,patient_id, timeframe, motion_name)

    pred_files =  ff.find_all_target_files(['pred_vector*'],save_sub)

    if pred_files.shape[0] < 1:
        print('havenot had parameters'); 

    print(patient_id, timeframe, motion_name)

    # ground truth movement:
    gt_centers = np.load(os.path.join(os.path.dirname(os.path.dirname(center_list[i])), 'ds/centerlist.npy'),allow_pickle = True)
    motion_centers = np.load(center_list[i], allow_pickle = True)
    gt_movements = np.zeros([12,2])
    for row in range(0,12):
        if np.isnan(gt_centers[row,0]) == 1:
            gt_movements[row,:] = [np.nan,np.nan]
        else:
            gt_movements[row,0] = motion_centers[row,0] - gt_centers[row,0]    
            gt_movements[row,1] = motion_centers[row,1] - gt_centers[row,1]   
    
    # predicted movement
    if 1==2:#os.path.isfile(os.path.join(save_sub,'pred_final.npy')) == 1:
        pred_movements = np.load(os.path.join(save_sub,'pred_final.npy'), allow_pickle = True)
    else: 
        pred_movements = []
        for pred_file in pred_files:
            pred_movements.append(np.load(pred_file, allow_pickle = True))
        pred_movements = np.asarray(pred_movements).reshape(-1, 12, 2 )

        # optmize
        pred_movements = optimize2(pred_movements, gt_movements, mode = [0,1])

        # save
        np.save(os.path.join(save_sub,'pred_final.npy'), pred_movements)


    # calculate the difference (use delta to basal slice):
    # gt vs. motion
    gt_centers, gt_non_nan_rows = ff.remove_nan(np.load(os.path.join(os.path.dirname(os.path.dirname(center_list[i])), 'ds/centerlist.npy'),allow_pickle = True), show_row_index=True)
    _,gt_x = Bspline.bspline_and_control_points(gt_centers, 6 , 'x'); gt_delta_x, _ = Bspline.delta_for_centerline(gt_x)
    _,gt_y = Bspline.bspline_and_control_points(gt_centers, 6 , 'y'); gt_delta_y, _ = Bspline.delta_for_centerline(gt_y)

    motion_centers = ff.remove_nan(np.load(center_list[i],allow_pickle = True))
    _,motion_x = Bspline.bspline_and_control_points(motion_centers, 6 , 'x'); motion_delta_x, _ = Bspline.delta_for_centerline(motion_x)
    _,motion_y = Bspline.bspline_and_control_points(motion_centers, 6 , 'y'); motion_delta_y, _ = Bspline.delta_for_centerline(motion_y)
    
    gt_motion_diff = [math.sqrt((motion_delta_x[j] - gt_delta_x[j]) ** 2 + (motion_delta_y[j] - gt_delta_y[j]) ** 2) for j in range(0,motion_delta_x.shape[0] )]
    
    # gt vs. pred
    pred_centers = motion_centers - pred_movements[gt_non_nan_rows,:]
    _,pred_x = Bspline.bspline_and_control_points(pred_centers, 6 , 'x'); pred_delta_x, _ = Bspline.delta_for_centerline(pred_x)
    _,pred_y = Bspline.bspline_and_control_points(pred_centers, 6 , 'y'); pred_delta_y, _ = Bspline.delta_for_centerline(pred_y)
    gt_pred_diff = [math.sqrt((pred_delta_x[j] - gt_delta_x[j]) ** 2 + (pred_delta_y[j] - gt_delta_y[j]) ** 2) for j in range(0,pred_delta_x.shape[0] )]


    Results.append([patient_id, timeframe, motion_name, np.mean(gt_motion_diff[1:]), np.mean(gt_pred_diff[1:])])
    columns = ['patient_id', 'timeframe', 'motion_name', 'Misalignment_motion', 'Misalignment_pred']
    df = pd.DataFrame(Results, columns = columns)
    df.to_excel(os.path.join(cg.predict_dir,trial_name ,'comparison_misalignment_unit_pixel.xlsx'),index = False)