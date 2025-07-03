import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import Build_list
import Generator
import EDSR 
import CMR_HFpEF_Analysis.motion_correction.Bspline as Bspline

import os
import numpy as np
import nibabel as nb
import pandas as pd
import math
cg = Defaults.Parameters()

trial_name = 'EDSR_3class_normal_motion_dataversion2'

patient_list = ff.sort_timeframe(ff.find_all_target_files(['*'],os.path.join(cg.predict_dir, 'Super_resolution', trial_name, 'images')),0)
case_list = []
for p in patient_list:
    cases = ff.find_all_target_files(['*'],p)
    for c in cases:
        case_list.append(c)

Results = []
for case in case_list:
    patient_id = os.path.basename(os.path.dirname(case))
    timeframe = os.path.basename(case)
    motion_folders = ff.find_all_target_files(['ds'],case).tolist()
    motion_folders += ff.sort_timeframe(ff.find_all_target_files(['normal_motion*'],case),0,'_').tolist()

    for motion_folder in motion_folders:
        motion_name = os.path.basename(motion_folder)

        print(patient_id, timeframe, motion_name)

        # load ground truth and its centerline:
        gt = nb.load(os.path.join(cg.data_dir,'simulated_data_version2', patient_id, timeframe, 'ds/data.nii.gz')).get_fdata()
        gt = np.round(gt); gt = gt.astype(int); gt = util.relabel(gt, 4, 0)
        # only take LV center points
        slice_list = []; center_list_raw = []
        for i in range(0,gt.shape[-1]):
            I = gt[:,:,i]
            # no LV:
            if np.where(I == 1)[0].shape[0] < 20 :
                center_list_raw.append(util.center_of_mass(np.zeros((20,20)),0,large = True))
                continue
            slice_list.append(i)
            center_list_raw.append(np.round(util.center_of_mass(I,0,large = True),2))
        gt_center_list = ff.remove_nan(center_list_raw)
        gt_CP_x,_ = Bspline.bspline_and_control_points(gt_center_list, 7, 'x'); gt_delta_x, _ = Bspline.delta_for_centerline(gt_CP_x)
        gt_CP_y,_ = Bspline.bspline_and_control_points(gt_center_list, 7, 'y'); gt_delta_y, _ = Bspline.delta_for_centerline(gt_CP_y)
        # gt_x = [gt_center_list[j][0] for j in range(0,len(gt_center_list))]; gt_y = [gt_center_list[j][1] for j in range(0,len(gt_center_list))]


        # load motion and its centerline
        motion = nb.load(os.path.join(cg.data_dir,'simulated_data_new', patient_id, timeframe, motion_name,'data.nii.gz')).get_fdata()
        motion = np.round(motion); motion = motion.astype(int); motion = util.relabel(motion, 4, 0)
        motion_center_list = []
        for i in slice_list:
            I = motion[:,:,i]
            motion_center_list.append(np.round(util.center_of_mass(I,0,large = True),2))
        motion_CP_x,_ = Bspline.bspline_and_control_points(motion_center_list, 7, 'x'); motion_delta_x, _ = Bspline.delta_for_centerline(motion_CP_x)
        motion_CP_y,_ = Bspline.bspline_and_control_points(motion_center_list, 7, 'y'); motion_delta_y, _ = Bspline.delta_for_centerline(motion_CP_y)
        # motion_x = [motion_center_list[j][0] for j in range(0,len(motion_center_list))]; motion_y = [motion_center_list[j][1] for j in range(0,len(motion_center_list))]

        # load prediction and its centerline
        pred = nb.load(os.path.join(motion_folder, 'batch_0/pred.nii.gz')).get_fdata(); pred= np.round(pred); pred = pred.astype(int)
        pred = util.downsample_in_z(pred,5)
        pred_center_list = []
        for i in slice_list:
            I = pred[:,:,i]
            pred_center_list.append(np.round(util.center_of_mass(I,0,large = True),2))
        pred_CP_x,_ = Bspline.bspline_and_control_points(pred_center_list, 7, 'x'); pred_delta_x, _ = Bspline.delta_for_centerline(pred_CP_x)
        pred_CP_y,_ = Bspline.bspline_and_control_points(pred_center_list, 7, 'y'); pred_delta_y, _ = Bspline.delta_for_centerline(pred_CP_y)
        # pred_x = [pred_center_list[j][0] for j in range(0,len(pred_center_list))]; pred_y = [pred_center_list[j][1] for j in range(0,len(pred_center_list))]
        
        # calculate difference
        gt_motion_diff = [math.sqrt((motion_delta_x[j] - gt_delta_x[j]) ** 2 + (motion_delta_y[j] - gt_delta_y[j]) ** 2) for j in range(0,motion_delta_x.shape[0] )]
        gt_pred_diff = [math.sqrt((pred_delta_x[j] - gt_delta_x[j]) ** 2 + (pred_delta_y[j] - gt_delta_y[j]) ** 2)for j in range(0,pred_delta_x.shape[0] )]

        r = [patient_id, timeframe, motion_name]
        for j in range(1,7):
            r += [gt_motion_diff[j], gt_pred_diff[j], gt_pred_diff[j] - gt_motion_diff[j]]
        r += [np.mean(gt_motion_diff[1:]),np.mean(gt_pred_diff[1:])]

        # smoothness evaluation
        r += [ff.smooth(gt_delta_x[1:]), ff.smooth(gt_delta_y[1:]), ff.smooth(motion_delta_x[1:]), ff.smooth(motion_delta_y[1:]), ff.smooth(pred_delta_x[1:]), ff.smooth(pred_delta_y[1:])]

        # sign evaluation
        r += [ff.same_sign(np.diff(gt_delta_x[1:])), ff.same_sign(np.diff(gt_delta_y[1:])), ff.same_sign(np.diff(motion_delta_x[1:])), ff.same_sign(np.diff(motion_delta_y[1:])), ff.same_sign(np.diff(pred_delta_x[1:])), ff.same_sign(np.diff(pred_delta_y[1:]))]
        Results.append(r)
        

        

columns = ['Patient_ID', 'timeframe', 'motion_name']
for j in range(1,7):
    columns += ['shift_motion', 'shift_pred', 'improve'+str(j)]
columns += ['shift_motion_mean', 'shift_pred_mean', 'gt_std_x', 'gt_std_y', 'motion_std_x', 'motion_std_y', 'pred_std_x', 'pred_std_y', 'gt_sign_x', 'gt_sign_y', 'motion_sign_x', 'motion_sign_y', 'pred_sign_x', 'pred_sign_y']
df = pd.DataFrame(Results, columns = columns)
df.to_excel(os.path.join(cg.predict_dir, 'Super_resolution', trial_name, 'comparison_centers_batch0model_testing.xlsx'), index = False)

   
 


