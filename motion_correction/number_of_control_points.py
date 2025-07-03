# assess how many control points are suitable
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nb
import pandas as pd
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.motion_correction.Bspline as Bspline
cg = Defaults.Parameters()

patient_list = ff.sort_timeframe(ff.find_all_target_files(['*'],os.path.join(cg.data_dir,'simulated_data_correct')),0)
case_list = []
for p in patient_list:
    cases = ff.find_all_target_files(['*'],p)
    for c in cases:
        case_list.append(c)


N_list = [4,5,6,7,8]

for N in N_list:
    result = []
    for case in case_list:
        patient_id = os.path.basename(os.path.dirname(case))
        tf = os.path.basename(case)

        center_file = os.path.join(case,'ds/center_list_flip_clean.npy')

        centers= np.round(ff.remove_nan(np.load(center_file, allow_pickle = True)),0)
        
        x = np.round(np.asarray([centers[i][0] for i in range(0,len(centers))]), 0)
        y = np.round(np.asarray([centers[i][1] for i in range(0,len(centers))]), 0)

        # find control points
        x_c = Bspline.bspline(np.linspace(0,1, x.shape[0]), x, np.linspace(0,1,N))
        y_c = Bspline.bspline(np.linspace(0,1, y.shape[0]), y, np.linspace(0,1,N))

        # in the spline constructed by control points, find the center point back
        xs = np.round(Bspline.bspline(np.linspace(0,1,N), x_c, np.linspace(0,1,x.shape[0])),0)
        ys = np.round(Bspline.bspline(np.linspace(0,1,N), y_c, np.linspace(0,1,y.shape[0])),0)

        # diff
        x_diff = np.sum(abs(x - xs)) 
        y_diff = np.sum(abs(y - ys))

        result.append([patient_id, tf, x_diff, y_diff])

    df = pd.DataFrame(result, columns=['Patient_ID', 'timeframe','x_diff', 'y_diff'])
    df.to_excel(os.path.join(cg.data_dir,'control_points_num_'+str(N)+'_results.xlsx'), index = False)


