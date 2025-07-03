# take the average of predicted motion parameters across models

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
iteration_num = 1
save_folder = os.path.join(cg.predict_dir,study_set, trial_name, 'round_'+str(iteration_num), 'images')
ff.make_folder([os.path.dirname(os.path.dirname(save_folder)),os.path.dirname(save_folder),save_folder])

###### define data sheet
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_motion_flip_clean_7_slice_10_normal.xlsx')

###### build patient list
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [0,1,2,3,4]
x_list_predict, y_list_predict, patient_id_list, tf_list, motion_name_list, batch_list,_,_ = b.__build__(batch_list = batches)
n = np.arange(0,patient_id_list.shape[0],1)
x_list_predict = x_list_predict[n]

Results = []
for i in range(0, x_list_predict.shape[0]):
    patient_id = str(patient_id_list[n[i]])
    timeframe = tf_list[n[i]]
    motion_name = motion_name_list[n[i]]
    batch = batch_list[n[i]]
    
    save_sub = os.path.join(save_folder,patient_id, timeframe, motion_name)

    center_x_files = ff.find_all_target_files(['center_x_*'],os.path.join(save_sub,'centers'))
    center_y_files = ff.find_all_target_files(['center_y_*'],os.path.join(save_sub,'centers'))

    # if center_x_files.shape[0] < 1 or center_y_files.shape[0] < 1:
    #     print('havenot had parameters'); 

    print(patient_id, timeframe, motion_name)

    # load truth:
    truth_file = os.path.join(cg.data_dir,'simulated_data_version2', patient_id, timeframe, 'ds/centerlist.npy')
    gt_centers = np.asarray(ff.remove_nan(np.load(truth_file,allow_pickle = True)))

    gt_CP_x,gt_x = Bspline.bspline_and_control_points(gt_centers, 7, 'x'); gt_delta_x, _ = Bspline.delta_for_centerline(gt_CP_x)
    gt_CP_y,gt_y = Bspline.bspline_and_control_points(gt_centers, 7, 'y'); gt_delta_y, _ = Bspline.delta_for_centerline(gt_CP_y)

    # load motion data
    if iteration_num == 1:
        motion_centers = np.asarray(ff.remove_nan(np.load(os.path.join(cg.data_dir, 'simulated_data_version2', patient_id, timeframe, motion_name, 'centerlist.npy'),allow_pickle = True)))
        motion_CP_x,motion_x = Bspline.bspline_and_control_points(motion_centers, 7, 'x'); motion_delta_x, _ = Bspline.delta_for_centerline(motion_CP_x)
        motion_CP_y,motion_y = Bspline.bspline_and_control_points(motion_centers, 7, 'y'); motion_delta_y, _ = Bspline.delta_for_centerline(motion_CP_y)
    else:
        motion_centers = np.load(os.path.join(cg.predict_dir,study_set, trial_name, 'round_'+str(iteration_num-1), 'images',patient_id, timeframe, motion_name,'centers', 'pred_final.npy'),allow_pickle = True)
        motion_delta_x = motion_centers[0,:]
        motion_delta_y = motion_centers[1,:]
        motion_x = Bspline.control_points(np.linspace(0,1,7), motion_delta_x, gt_x.shape[0] )
        motion_y = Bspline.control_points(np.linspace(0,1,7), motion_delta_y, gt_x.shape[0] )

    # do predictions
    if os.path.isfile(os.path.join(save_sub,'centers', 'pred_final.npy')) == 1:
        pred = np.load(os.path.join(save_sub,'centers', 'pred_final.npy'),allow_pickle = True)
        pred_delta_x = pred[0,:]
        pred_delta_y = pred[1,:]

    else:
        # get center_x
        center_x = []
        for center_x_file in center_x_files:
            center_x.append(np.load(center_x_file, allow_pickle = True)[0,:])
        center_x = np.asarray(center_x).reshape(-1, 7)
        pred_delta_x = ff.optimize(center_x, gt_delta_x, random_mode = False, mode = 2, rank_max = 0, random_rank = False,  boundary = 0.8)

        center_y = []
        for center_y_file in center_y_files:
            center_y.append(np.load(center_y_file, allow_pickle = True)[1,:])
        center_y = np.asarray(center_y).reshape(-1, 7)
        pred_delta_y = ff.optimize(center_y, gt_delta_y, random_mode = False, mode = 2, rank_max = 0, random_rank = False,  boundary = 0.8)

        # save
        np.save(os.path.join(save_sub,'centers', 'pred_final.npy'), np.reshape(np.concatenate([pred_delta_x, pred_delta_y], axis = -1), (2,-1)))

    # quantitative:
    r = [patient_id, timeframe, motion_name]

    # print each data
    for j in range(1,pred_delta_x.shape[0]):
        r += [gt_delta_x[j], motion_delta_x[j] , pred_delta_x[j]]
        r += [gt_delta_y[j], motion_delta_y[j] , pred_delta_y[j]]
        r += [math.sqrt(gt_delta_x[j] ** 2 + gt_delta_y[j] ** 2), math.sqrt(motion_delta_x[j] ** 2 + motion_delta_y[j] ** 2),math.sqrt(pred_delta_x[j] ** 2 + pred_delta_y[j] ** 2)]

    # calculate difference
    gt_motion_diff = [math.sqrt((motion_delta_x[j] - gt_delta_x[j]) ** 2 + (motion_delta_y[j] - gt_delta_y[j]) ** 2) for j in range(0,motion_delta_x.shape[0] )]
    gt_pred_diff = [math.sqrt((pred_delta_x[j] - gt_delta_x[j]) ** 2 + (pred_delta_y[j] - gt_delta_y[j]) ** 2)for j in range(0,pred_delta_x.shape[0] )]
    
   
    for j in range(1,len(gt_motion_diff)):
        r += [gt_motion_diff[j], gt_pred_diff[j], gt_pred_diff[j] - gt_motion_diff[j]]
    r += [np.mean(gt_motion_diff[1:]),np.mean(gt_pred_diff[1:])]


    # c2c distance
    gt_c2c_x = np.mean(abs(np.diff(gt_x))); gt_c2c_y = np.mean(abs(np.diff(gt_y)))
    motion_c2c_x = np.mean(abs(np.diff(motion_x))); motion_c2c_y = np.mean(abs(np.diff(motion_y)))
    pred_x = Bspline.control_points(np.linspace(0,1,7), pred_delta_x, gt_x.shape[0] )
    pred_y = Bspline.control_points(np.linspace(0,1,7), pred_delta_y, gt_x.shape[0] )
    pred_c2c_x = np.mean(abs(np.diff(pred_x))); pred_c2c_y = np.mean(abs(np.diff(pred_y)))
    r += [gt_c2c_x, gt_c2c_y, motion_c2c_x, motion_c2c_y, pred_c2c_x, pred_c2c_y]

    Results.append(r)



columns = ['Patient_ID', 'timeframe', 'motion_name']
for j in range(1,len(gt_motion_diff)):
    columns += ['gt_x'+str(j), 'motion_x'+str(j), 'pred_x'+str(j), 'gt_y'+str(j), 'motion_y'+str(j), 'pred_y'+str(j), 'gt_dis'+str(j), 'motion_dis'+str(j), 'pred_dis'+str(j)]
    
for j in range(1,len(gt_motion_diff)):
    columns += ['error_motion'+str(j), 'error_pred'+str(j), 'improve'+str(j)]

columns += ['error_motion_mean', 'error_pred_mean','gt_c2c_x','gt_c2c_y', 'motion_c2c_x', 'motion_c2c_y', 'pred_c2c_x', 'pred_c2c_y']
df = pd.DataFrame(Results, columns = columns )
df = df.round(decimals = 3)
df.to_excel(os.path.join(os.path.dirname(save_folder),'comparison_centers_validation_complete2.xlsx'), index = False)