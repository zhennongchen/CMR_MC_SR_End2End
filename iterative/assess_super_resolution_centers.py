# take the average of predicted motion parameters across models

import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Trained_models.motion_correction_models as trained_models
import CMR_HFpEF_Analysis.iterative.Build_list as Build_list
import CMR_HFpEF_Analysis.motion_correction.Bspline as Bspline
import math
import os
import numpy as np
import pandas as pd
import nibabel as nb


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
batches = [5]
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

    print(patient_id, timeframe, motion_name)

    # load truth:
    truth_file = os.path.join(cg.data_dir,'simulated_data_version2', patient_id, timeframe, 'ds/centerlist.npy')
    gt_centers = np.asarray(ff.remove_nan(np.load(truth_file,allow_pickle = True)))

    gt_CP_x,gt_x = Bspline.bspline_and_control_points(gt_centers, 7, 'x'); gt_delta_x, _ = Bspline.delta_for_centerline(gt_CP_x)
    gt_CP_y,gt_y = Bspline.bspline_and_control_points(gt_centers, 7, 'y'); gt_delta_y, _ = Bspline.delta_for_centerline(gt_CP_y)

    # load motion data
    motion_centers = np.load(os.path.join(cg.predict_dir,study_set, trial_name, 'round_'+str(iteration_num), 'images',patient_id, timeframe, motion_name,'centers', 'pred_final.npy'),allow_pickle = True)
    motion_delta_x = motion_centers[0,:]
    motion_delta_y = motion_centers[1,:]
    motion_x = Bspline.control_points(np.linspace(0,1,7), motion_delta_x, gt_x.shape[0] )
    motion_y = Bspline.control_points(np.linspace(0,1,7), motion_delta_y, gt_x.shape[0] )

    # do predictions
    pred = nb.load(os.path.join(save_sub, 'pred_ds_flip_clean.nii.gz')).get_fdata()
    center_list_raw = []
    for j in range(0,pred.shape[-1]):
        I = pred[:,:,j]
        # no LV:
        if np.where(I == 1)[0].shape[0] < 20 :
            center_list_raw.append(util.center_of_mass(np.zeros((20,20)),0,large = True))
            continue
        center_list_raw.append(np.round(util.center_of_mass(I,0,large = True),2))
    pred_center_list = ff.remove_nan(center_list_raw)
    pred_CP_x,pred_x = Bspline.bspline_and_control_points(pred_center_list, 7, 'x'); pred_delta_x, _ = Bspline.delta_for_centerline(pred_CP_x)
    pred_CP_y,pred_y = Bspline.bspline_and_control_points(pred_center_list, 7, 'y'); pred_delta_y, _ = Bspline.delta_for_centerline(pred_CP_y)

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
df.to_excel(os.path.join(os.path.dirname(save_folder),'comparison_super_resolution_centers_test_complete.xlsx'), index = False)