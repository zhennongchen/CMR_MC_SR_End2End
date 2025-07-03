# check the baseline error caused by B-spline interpolation in Low-resolution data and high-resolution data

import numpy as np
import os
import nibabel as nb
import pandas as pd
import math
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.motion_correction.Bspline as Bspline

cg = Defaults.Parameters()

patient_list = ff.sort_timeframe(ff.find_all_target_files(['*'],os.path.join(cg.data_dir,'simulated_data_version2')),0)
case_list = []
for p in patient_list:
    cases = ff.find_all_target_files(['*'],p)
    for c in cases:
        case_list.append(c)

results = []
for case in case_list:
    patient_id = str(os.path.basename(os.path.dirname(case)))
    tf = os.path.basename(case)
    print('patient: ', patient_id, tf)

    for i in range(0,1):
        if i > 0:
            folder = os.path.join(case, 'normal_motion_' + str(i))
        else: 
            folder = os.path.join(case, 'ds')
        print(folder)

        motion_name = os.path.basename(folder)

        # load image
        img = nb.load(os.path.join(folder, 'HR/HR_data_flip_clean.nii.gz')).get_fdata()
        # img = nb.load(os.path.join(folder, 'data_flip_clean.nii.gz')).get_fdata()
        img = np.round(img).astype(int)
        img = util.relabel(img, 4, 0)

        # find centerline points
        # only take LV center points
        slice_list = []; center_list_raw = []
        for i in range(0,img.shape[-1]):
            I = img[:,:,i]
            # no LV:
            if np.where(I == 1)[0].shape[0] < 50 :
                center_list_raw.append(util.center_of_mass(np.zeros((20,20)),0,large = True))
                continue
            slice_list.append(i)
            center_list_raw.append(np.round(util.center_of_mass(I,0,large = True),2))
        center_list = ff.remove_nan(center_list_raw)

        # fit B-spline
        N = 10

        # center_x
        # original spline S and its control points by centerline points
        xx = np.linspace(0,1,len(center_list))
        center_x_cp, center_x = Bspline.bspline_and_control_points(center_list, N, x_or_y = 'x')
        if len(center_x)< 7:
            continue
        # in spline S, find the points corresponding to the position of original centerline points
        center_x_S = Bspline.bspline(np.linspace(xx[0], xx[-1],N), center_x_cp, xx)

        error_x = np.mean(abs(center_x - center_x_S))

        # center_y
        center_y_cp, center_y = Bspline.bspline_and_control_points(center_list, N, x_or_y = 'y')
        # in spline S, find the points corresponding to the position of original centerline points
        center_y_S = Bspline.bspline(np.linspace(xx[0], xx[-1],N), center_y_cp, xx)
        error_y = np.mean(abs(center_y - center_y_S))

        error = np.mean(np.sqrt((np.asarray(center_x)  - np.asarray(center_x_S))** 2 + (np.asarray(center_y)  - np.asarray(center_y_S))** 2))

        results.append([patient_id, tf,  motion_name, len(center_list),error_x, error_y, error])
        print(error_x, error_y, error)

 
    

df = pd.DataFrame(results, columns= ['Patient_ID', 'timeframe', 'random_name','num_slices', 'error_x', 'error_y', 'error'])
# df.to_excel(os.path.join(cg.data_dir, 'Bspline_7-control-points_baseline_error_simulated_data_version2_LR.xlsx'), index = False)
df.to_excel(os.path.join(cg.data_dir, 'Bspline_10-control-points_baseline_error_simulated_data_version2_HR.xlsx'), index = False)
