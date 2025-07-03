# assess the misalignment (across all slices not all Control points)

import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.iterative.Build_list as Build_list
import CMR_HFpEF_Analysis.motion_correction.Bspline as Bspline
import math
import os
import numpy as np
import pandas as pd


cg = Defaults.Parameters()

###### define study and data
trial_name = 'Iteration_C'
study_set = 'Combined'
iteration_num = 2
save_folder = os.path.join(cg.predict_dir,study_set, trial_name, 'round_'+str(iteration_num), 'images')
ff.make_folder([os.path.dirname(os.path.dirname(save_folder)),os.path.dirname(save_folder),save_folder])

###### define data sheet
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_motion_flip_clean_7_slice_10_normal.xlsx')

###### build patient list
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [5]
x_list_predict, y_list_predict, patient_id_list, tf_list, motion_name_list, batch_list,_,_ = b.__build__(batch_list = batches)
n = np.arange(0,patient_id_list.shape[0],1)
x_list_predict = x_list_predict[n]

Results = []
for i in range(0,x_list_predict.shape[0]):
    patient_id = str(patient_id_list[n[i]])
    timeframe = tf_list[n[i]]
    motion_name = motion_name_list[n[i]]
    batch = batch_list[n[i]]
    
    save_sub = os.path.join(save_folder,patient_id, timeframe, motion_name)

    print(patient_id, timeframe, motion_name)

    # load truth:
    # low resolution
    if iteration_num == 1:
        truth_file = os.path.join(cg.data_dir,'simulated_data_version2', patient_id, timeframe, 'ds/centerlist.npy')
        gt_centers = np.asarray(ff.remove_nan(np.load(truth_file,allow_pickle = True)))

        gt_CP_x,gt_x = Bspline.bspline_and_control_points(gt_centers, 7, 'x'); gt_delta_x, _ = Bspline.delta_for_centerline(gt_x)
        gt_CP_y,gt_y = Bspline.bspline_and_control_points(gt_centers, 7, 'y'); gt_delta_y, _ = Bspline.delta_for_centerline(gt_y)
    else:
        truth_file = os.path.join(cg.data_dir,'simulated_data_version2', patient_id, timeframe, 'ds/HR/HR_centerlist_LV_slices.npy')
        gt_centers = np.asarray(ff.remove_nan(np.load(truth_file,allow_pickle = True)))

        gt_CP_x,gt_x = Bspline.bspline_and_control_points(gt_centers, 10, 'x'); gt_delta_x, _ = Bspline.delta_for_centerline(gt_x)
        gt_CP_y,gt_y = Bspline.bspline_and_control_points(gt_centers, 10, 'y'); gt_delta_y, _ = Bspline.delta_for_centerline(gt_y)


    # load motion data
    if iteration_num == 1:
        motion_centers = np.asarray(ff.remove_nan(np.load(os.path.join(cg.data_dir, 'simulated_data_version2', patient_id, timeframe, motion_name, 'centerlist.npy'),allow_pickle = True)))
        motion_CP_x,motion_x = Bspline.bspline_and_control_points(motion_centers, 7, 'x'); motion_delta_x, _ = Bspline.delta_for_centerline(motion_x)
        motion_CP_y,motion_y = Bspline.bspline_and_control_points(motion_centers, 7, 'y'); motion_delta_y, _ = Bspline.delta_for_centerline(motion_y)
    else:
        motion_centers = np.load(os.path.join(cg.predict_dir,study_set, trial_name, 'round_'+str(iteration_num-1), 'images',patient_id, timeframe, motion_name,'centers', 'pred_final.npy'),allow_pickle = True)
        motion_delta_x = motion_centers[0,:]
        motion_delta_y = motion_centers[1,:]
        motion_delta_x = Bspline.control_points(np.linspace(0,1,7), motion_delta_x, gt_x.shape[0] )
        motion_delta_y = Bspline.control_points(np.linspace(0,1,7), motion_delta_y, gt_x.shape[0] )


    # do predictions
    pred = np.load(os.path.join(save_sub,'centers', 'pred_final.npy'),allow_pickle = True)
    pred_x = pred[0,:]
    pred_y = pred[1,:]
    if iteration_num == 1:
        pred_delta_x = Bspline.control_points(np.linspace(0,1,7), pred_x, gt_x.shape[0] )
        pred_delta_y = Bspline.control_points(np.linspace(0,1,7), pred_y, gt_x.shape[0] )
    else:
        pred_delta_x = Bspline.control_points(np.linspace(0,1,10), pred_x, gt_x.shape[0] )
        pred_delta_y = Bspline.control_points(np.linspace(0,1,10), pred_y, gt_x.shape[0] )

    ## print(gt_delta_x.shape, motion_delta_x.shape, pred_delta_x.shape)

    # quantitative:
    r = [patient_id, timeframe, motion_name]

    # calculate difference
    gt_motion_diff = [math.sqrt((motion_delta_x[j] - gt_delta_x[j]) ** 2 + (motion_delta_y[j] - gt_delta_y[j]) ** 2) for j in range(0,motion_delta_x.shape[0] )]
    gt_pred_diff = [math.sqrt((pred_delta_x[j] - gt_delta_x[j]) ** 2 + (pred_delta_y[j] - gt_delta_y[j]) ** 2)for j in range(0,pred_delta_x.shape[0] )]
    
   
    # for j in range(1,len(gt_motion_diff)):
    #     r += [gt_motion_diff[j], gt_pred_diff[j], gt_pred_diff[j] - gt_motion_diff[j]]
    r += [np.mean(gt_motion_diff[1:]),np.mean(gt_pred_diff[1:]), np.mean(gt_pred_diff[1:]) - np.mean(gt_motion_diff[1:]) ]
    print(np.mean(gt_motion_diff[1:]) - np.mean(gt_pred_diff[1:]))


    # c2c distance
    gt_c2c_x = np.mean(abs(np.diff(gt_delta_x))); gt_c2c_y = np.mean(abs(np.diff(gt_delta_y)))
    motion_c2c_x = np.mean(abs(np.diff(motion_delta_x))); motion_c2c_y = np.mean(abs(np.diff(motion_delta_y)))
    pred_c2c_x = np.mean(abs(np.diff(pred_delta_x))); pred_c2c_y = np.mean(abs(np.diff(pred_delta_y)))
    r += [gt_c2c_x, gt_c2c_y, motion_c2c_x, motion_c2c_y, pred_c2c_x, pred_c2c_y]

    Results.append(r)



columns = ['Patient_ID', 'timeframe', 'motion_name']

columns += ['error_motion_mean', 'error_pred_mean','improvement', 'gt_c2c_x','gt_c2c_y', 'motion_c2c_x', 'motion_c2c_y', 'pred_c2c_x', 'pred_c2c_y']
df = pd.DataFrame(Results, columns = columns )
df = df.round(decimals = 3)
df.to_excel(os.path.join(os.path.dirname(save_folder),'comparison_centers_test_complete_across_slices.xlsx'), index = False)