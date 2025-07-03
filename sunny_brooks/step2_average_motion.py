# take the average of predicted motion parameters across models

import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Trained_models.motion_correction_models as trained_models
import CMR_HFpEF_Analysis.sunny_brooks.Build_list as Build_list
import CMR_HFpEF_Analysis.motion_correction.Bspline as Bspline
import math
import os
import numpy as np
import pandas as pd


cg = Defaults.Parameters()

###### define study and data
trial_name = 'Iteration_C'
study_set = 'Sunny_Brooks'
iteration_num = 1
save_folder = os.path.join(cg.predict_dir,study_set, trial_name, 'round_'+str(iteration_num), 'images')
ff.make_folder([os.path.dirname(os.path.dirname(save_folder)),os.path.dirname(save_folder),save_folder])

###### define data sheet
data_sheet = os.path.join(cg.data_dir,'Patient_list/Sunny_Brooks.xlsx')

###### build patient list
print('Build List...')
b = Build_list.Build(data_sheet)
input_list, patient_id_list, patient_tf_list = b.__build__()
n = np.arange(0,patient_id_list.shape[0],1)
input_list = input_list[n]

Results = []
for i in range(0,patient_id_list.shape[0]):
    patient_id = patient_id_list[n[i]]
    timeframe = patient_tf_list[n[i]]
   
    
    save_sub = os.path.join(save_folder,patient_id, timeframe,'centers')

    center_x_files = ff.find_all_target_files(['center_x_*'],save_sub)
    center_y_files = ff.find_all_target_files(['center_y_*'],save_sub)

    # if center_x_files.shape[0] < 1 or center_y_files.shape[0] < 1:
    #     print('havenot had parameters'); 

    print(patient_id, timeframe)


    # do predictions
    if os.path.isfile(os.path.join(save_sub, 'pred_final.npy')) == 1:
        continue

    else:
        # get center_x
        center_x = []
        for center_x_file in center_x_files:
            center_x.append(np.load(center_x_file, allow_pickle = True)[0,:])
        center_x = np.asarray(center_x).reshape(-1, 7)
        # print('center_x:', center_x)
        pred_delta_x = ff.optimize(center_x, center_x, random_mode = False, mode = 2, rank_max = 0, random_rank = False,  boundary = 0.8)
        # print('final x: ',pred_delta_x)

        center_y = []
        for center_y_file in center_y_files:
            center_y.append(np.load(center_y_file, allow_pickle = True)[1,:])
        center_y = np.asarray(center_y).reshape(-1, 7)
        pred_delta_y = ff.optimize(center_y, center_y, random_mode = False, mode = 2, rank_max = 0, random_rank = False,  boundary = 0.8)
        # print('center_y:', center_y)
        # print('final y: ',pred_delta_y)

        # print(pred_delta_x, pred_delta_y)
        # save
        np.save(os.path.join(save_sub, 'pred_final.npy'), np.reshape(np.concatenate([pred_delta_x, pred_delta_y], axis = -1), (2,-1)))

