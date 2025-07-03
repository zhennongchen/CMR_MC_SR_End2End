import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Trained_models.motion_correction_models as trained_models
import CMR_HFpEF_Analysis.Build_list_data_prepare.Build_list as Build_list
import Bspline
import math
import os
import numpy as np
import pandas as pd


cg = Defaults.Parameters()
mm = trained_models.trained_models()

# build lists
CP_num = 7
trial_name = 'Motion_ResNet_new'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_simulated_data_15normals.xlsx')
save_folder = os.path.join(cg.predict_dir, trial_name, 'images'); ff.make_folder([os.path.dirname(save_folder), save_folder])

b = Build_list.Build(data_sheet)
patient_id_list, patient_tf_list, motion_name_list, batch_list, _,_, _, _, gt_center_list,  _, x_list, motion_center_list = b.__build__(batch_list = [5])
n = np.arange(0,patient_id_list.shape[0],1)
x_list = x_list[n]; gt_center_list = gt_center_list[n]; motion_center_list = motion_center_list[n]


Results = []
for i in range(0, x_list.shape[0]):
    patient_id = str(patient_id_list[n[i]])
    timeframe = patient_tf_list[n[i]]
    motion_name = motion_name_list[n[i]]
    batch = batch_list[n[i]]
    
    save_sub = os.path.join(save_folder,patient_id, timeframe, motion_name, 'centers')

    pred_files = ff.find_all_target_files(['pred_centers*'],save_sub)

    if pred_files.shape[0] < 1:
        print('havenot had parameters'); 

    print(patient_id, timeframe, motion_name)

    # load truth:
    print('truth file: ' , gt_center_list[i])
    gt_centers = np.asarray(ff.remove_nan(np.load(gt_center_list[i],allow_pickle = True)))
   
    gt_CP_x,gt_x = Bspline.bspline_and_control_points(gt_centers, CP_num , 'x'); gt_delta_x, _ = Bspline.delta_for_centerline(gt_CP_x)
    gt_CP_y,gt_y = Bspline.bspline_and_control_points(gt_centers, CP_num , 'y'); gt_delta_y, _ = Bspline.delta_for_centerline(gt_CP_y)

    # do predictions
    if 1==2:#os.path.isfile(os.path.join(save_sub,'centers', 'pred_final.npy')) == 1:
        pred = np.load(os.path.join(save_sub,'centers', 'pred_final.npy'),allow_pickle = True)
        pred_delta_x = pred[0,:]
        pred_delta_y = pred[1,:]

    else:
        # get center_x
        pred_delta_x = []
        for pred_file in pred_files:
            pred_delta_x.append(np.load(pred_file, allow_pickle = True)[0,:])
        pred_delta_x = np.asarray(pred_delta_x).reshape(-1, CP_num )
        pred_delta_x = ff.optimize(pred_delta_x, gt_delta_x, mode = [0,1] )

        pred_delta_y = []
        for pred_file in pred_files:
            pred_delta_y.append(np.load(pred_file, allow_pickle = True)[1,:])
        pred_delta_y = np.asarray(pred_delta_y).reshape(-1, CP_num )
        pred_delta_y = ff.optimize(pred_delta_y, gt_delta_y, mode = [0,1] )
      
        # save
        np.save(os.path.join(save_sub,'pred_final.npy'), np.reshape(np.concatenate([pred_delta_x, pred_delta_y], axis = -1), (2,-1)))


    # per real center point - not per CP point:

    # load motion data
    motion_centers = np.asarray(ff.remove_nan(np.load(motion_center_list[i],allow_pickle = True)))
    motion_CP_x,motion_x = Bspline.bspline_and_control_points(motion_centers, CP_num , 'x'); motion_delta_x, _ = Bspline.delta_for_centerline(motion_x)
    motion_CP_y,motion_y = Bspline.bspline_and_control_points(motion_centers, CP_num , 'y'); motion_delta_y, _ = Bspline.delta_for_centerline(motion_y)

    gt_delta_x, _ = Bspline.delta_for_centerline(gt_x)
    gt_delta_y, _ = Bspline.delta_for_centerline(gt_y)

    pred_delta_x = Bspline.control_points(np.linspace(0,1,CP_num), pred_delta_x, gt_x.shape[0] )
    pred_delta_y = Bspline.control_points(np.linspace(0,1,CP_num), pred_delta_y, gt_x.shape[0] )
    print(pred_delta_x.shape)

    # calculate difference
    gt_motion_diff = [math.sqrt((motion_delta_x[j] - gt_delta_x[j]) ** 2 + (motion_delta_y[j] - gt_delta_y[j]) ** 2) for j in range(0,motion_delta_x.shape[0] )]
    gt_pred_diff = [math.sqrt((pred_delta_x[j] - gt_delta_x[j]) ** 2 + (pred_delta_y[j] - gt_delta_y[j]) ** 2)for j in range(0,pred_delta_x.shape[0] )]
    
    r = [patient_id, timeframe, motion_name]
    # # for j in range(1,len(gt_motion_diff)):
    # #     r += [gt_motion_diff[j], gt_pred_diff[j], gt_pred_diff[j] - gt_motion_diff[j]]
    r += [np.mean(gt_motion_diff[1:]),np.mean(gt_pred_diff[1:])]
    print(np.mean(gt_motion_diff[1:]), np.mean(gt_pred_diff[1:]))

    # smoothness evaluation
    # r += [ff.smooth(gt_delta_x[1:]), ff.smooth(gt_delta_y[1:]), ff.smooth(motion_delta_x[1:]), ff.smooth(motion_delta_y[1:]), ff.smooth(pred_delta_x[1:]), ff.smooth(pred_delta_y[1:])]

    # # c2c distance
    # gt_c2c_x = np.mean(abs(np.diff(gt_x))); gt_c2c_y = np.mean(abs(np.diff(gt_y)))
    # motion_c2c_x = np.mean(abs(np.diff(motion_x))); motion_c2c_y = np.mean(abs(np.diff(motion_y)))
    # pred_x = Bspline.control_points(np.linspace(0,1,7), pred_delta_x, gt_x.shape[0] )
    # pred_y = Bspline.control_points(np.linspace(0,1,7), pred_delta_y, gt_x.shape[0] )
    # pred_c2c_x = np.mean(abs(np.diff(pred_x))); pred_c2c_y = np.mean(abs(np.diff(pred_y)))
    # r += [gt_c2c_x, gt_c2c_y, motion_c2c_x, motion_c2c_y, pred_c2c_x, pred_c2c_y]

    Results.append(r)

    columns = ['Patient_ID', 'timeframe', 'motion_name']
    # for j in range(1,len(gt_motion_diff)):
    #     columns += ['shift_motion', 'shift_pred', 'improve'+str(j)]
    columns += ['Misalignment_motion', 'Misalignment_pred']#, 'gt_std_x', 'gt_std_y', 'motion_std_x', 'motion_std_y', 'pred_std_x', 'pred_std_y','gt_c2c_x','gt_c2c_y', 'motion_c2c_x', 'motion_c2c_y', 'pred_c2c_x', 'pred_c2c_y']
    df = pd.DataFrame(Results, columns = columns )
    df = df.round(decimals=3)
    df.to_excel(os.path.join(cg.predict_dir, trial_name,'comparison_centers_unit_pixel.xlsx'), index = False)