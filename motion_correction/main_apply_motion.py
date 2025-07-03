# apply predicted motion correction to the simulated data

import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.data_simulation.transformation as transform
import Generator_motion 
import Bspline
import os
import numpy as np
import pandas as pd
import nibabel as nb
from scipy.spatial.distance import directed_hausdorff
cg = Defaults.Parameters()

trial_name = 'Motion_ResNet_3'
data_set = 'Motion'
save_folder = os.path.join(cg.predict_dir,data_set, trial_name, 'images')

patient_list = ff.sort_timeframe(ff.find_all_target_files(['*'],save_folder),0)
case_list = []
for p in patient_list:
    cases = ff.find_all_target_files(['*'],p)
    for c in cases:
        case_list.append(c)

Results = []
for case in case_list:
    patient_id = os.path.basename(os.path.dirname(case))
    timeframe = os.path.basename(case)
    motion_folders = ff.sort_timeframe(ff.find_all_target_files(['normal_motion*'],case),0,'_').tolist()

    for motion_folder in motion_folders:
        motion_name = os.path.basename(motion_folder)

        print(patient_id, timeframe, motion_name)

        # load gt image
        gt = nb.load(os.path.join(cg.data_dir,'simulated_data_version2',patient_id, timeframe, 'ds/data_flip_clean.nii.gz'))
        affine = gt.affine; gt = gt.get_fdata();  gt = np.round(gt); gt = gt.astype(int); gt = util.relabel(gt, 4, 0)
        # load gt centerpoints
        gt_centers = np.asarray(ff.remove_nan(np.load(os.path.join(cg.data_dir,'simulated_data_version2',patient_id, timeframe, 'ds/centerlist.npy'),allow_pickle = True)))
        
        # load motion image
        motion = nb.load(os.path.join(cg.data_dir,'simulated_data_version2',patient_id, timeframe, motion_name,'data_flip_clean.nii.gz')).get_fdata()
        motion = np.round(motion); motion = motion.astype(int); motion = util.relabel(motion, 4, 0)
        # load motion centerpoints
        motion_centers = np.asarray(ff.remove_nan(np.load(os.path.join(cg.data_dir,'simulated_data_version2',patient_id, timeframe, motion_name,'centerlist.npy'),allow_pickle = True)))
      
        shift = gt_centers[0,:] - motion_centers[0,:]
        # shift the motion image so the most basal slices in gt and in motion are aligned with each other
        _,_,_,M = transform.generate_transform_matrix([shift[0], shift[1],0],[0,0,0],[1,1,1],gt.shape)
        motion = transform.apply_affine_transform(motion, M, order = 0)

        # load pred
        pred_cp = np.load(os.path.join(motion_folder, 'batch_0', 'pred_centers.npy'),allow_pickle = True)
        pred_x = Bspline.control_points(np.linspace(0,1,7), pred_cp[0,:], gt_centers.shape[0])
        pred_y = Bspline.control_points(np.linspace(0,1,7), pred_cp[1,:], gt_centers.shape[0])
        pred_img = np.copy(gt)
        start_slice = [ii for ii in range(0,gt.shape[-1]) if np.sum(gt[:,:,ii])>0][0]
        
        for s in range(start_slice+1, start_slice + gt_centers.shape[0]):
            pred_center = [gt_centers[0,0]  + pred_x[s - start_slice] , gt_centers[0,1]  + pred_y[s - start_slice]]
            shift = [pred_center[0] - gt_centers[s-start_slice,0], pred_center[1] - gt_centers[s-start_slice,1]]
            I = pred_img[:,:,s]
            _,_,_,M = transform.generate_transform_matrix([shift[0], shift[1]],0.0, [1,1], I.shape)
            img_new = transform.apply_affine_transform(I, M, order = 0)
            pred_img[:,:,s] = img_new
        
      
        # dice
        motion_lv_dice = util.np_categorical_dice(motion, gt, 1)
        motion_myo_dice = util.np_categorical_dice(motion, gt, 2)
        pred_lv_dice = util.np_categorical_dice(pred_img, gt, 1)
        pred_myo_dice = util.np_categorical_dice(pred_img, gt, 2)

        # Hausdorff distance
        gt_lv_pixels = np.asarray(ff.count_pixel(gt,1)[1]); gt_myo_pixels = np.asarray(ff.count_pixel(gt,2)[1])
        motion_lv_pixels = np.asarray(ff.count_pixel(motion,1)[1]); motion_myo_pixels = np.asarray(ff.count_pixel(motion,2)[1])
        pred_lv_pixels = np.asarray(ff.count_pixel(pred_img,1)[1]); pred_myo_pixels = np.asarray(ff.count_pixel(pred_img,2)[1])

        motion_lv_dis = directed_hausdorff(gt_lv_pixels, motion_lv_pixels)[0] * 1.25
        motion_myo_dis = directed_hausdorff(gt_myo_pixels, motion_myo_pixels)[0] * 1.25
        pred_lv_dis = directed_hausdorff(gt_lv_pixels, pred_lv_pixels)[0] * 1.25
        pred_myo_dis = directed_hausdorff(gt_myo_pixels, pred_myo_pixels)[0] * 1.25

        # print(motion_lv_dice, motion_myo_dice, pred_lv_dice, pred_myo_dice)

        nb.save(nb.Nifti1Image(pred_img.astype(float), affine), os.path.join(motion_folder,'batch_0','pred_img.nii.gz'))
        Results.append([patient_id, timeframe, motion_name, motion_lv_dice, motion_myo_dice, pred_lv_dice, pred_myo_dice,
                        motion_lv_dis, motion_myo_dis, pred_lv_dis, pred_myo_dis])

df = pd.DataFrame(Results, columns = ['Patient_ID','timeframe','motion_name','motion_lv_dice', 'motion_myo_dice', 'pred_lv_dice', 'pred_myo_dice',
                                       'motion_lv_dis', 'motion_myo_dis', 'pred_lv_dis', 'pred_myo_dis'])
df.to_excel(os.path.join(os.path.dirname(save_folder), 'comparison_DICE_batch0model_cross_val.xlsx'), index = False)