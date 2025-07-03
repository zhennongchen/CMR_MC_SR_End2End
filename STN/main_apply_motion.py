# apply predicted motion correction to the simulated data
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.data_simulation.transformation as transform
import CMR_HFpEF_Analysis.Build_list_data_prepare.Build_list as Build_list
import os
import numpy as np
import pandas as pd
import nibabel as nb
from scipy.spatial.distance import directed_hausdorff
cg = Defaults.Parameters()

trial_name = 'STN_vector_img'
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
    
    save_sub = os.path.join(save_folder,patient_id, timeframe, motion_name)
    print(patient_id, timeframe, motion_name)

    # load gt image
    gt = nb.load(os.path.join(cg.data_dir,'contour_dataset/simulated_data',patient_id, timeframe, 'ds/data_clean.nii.gz'))
    affine = gt.affine; gt_img = gt.get_fdata();  gt_img = np.round(gt_img); gt_img = gt_img.astype(int); gt_img = util.relabel(gt_img, 4, 0)
    # load gt centerpoints
    gt_centers = np.load(gt_center_list[i],allow_pickle = True)
    
    # now load motion image
    motion = nb.load(os.path.join(cg.data_dir,'contour_dataset/simulated_data',patient_id, timeframe, motion_name,'data_clean.nii.gz'))
    motion_img = motion.get_fdata(); motion_img = np.round(motion_img); motion_img = motion_img.astype(int); motion_img = util.relabel(motion_img, 4, 0)
    # load motion centerpoints
    motion_centers = np.load(motion_center_list[i],allow_pickle = True)

    # now load the predicted movement
    pred_movements = np.load(os.path.join(cg.predict_dir, trial_name, 'images',patient_id, timeframe, motion_name,'pred_final.npy'),allow_pickle = True)
    
    # now apply the movement to the motion image
    pred_img = np.zeros(motion_img.shape)
    for row in range(0,12):
        if np.isnan(gt_centers[row,0]) == 1:
            continue
        else:
            I = motion_img[:,:,row]
            translation,rotation,scale,M = transform.generate_transform_matrix(pred_movements[row,:],0.0,[1,1],I.shape)
            pred_img[:,:,row] = transform.apply_affine_transform(I, M, order = 0)
    print(np.unique(pred_img))

    # now calculate the metrics: Dice:
    motion_lv_dice = ff.np_categorical_dice(motion_img, gt_img, 1)
    motion_myo_dice = ff.np_categorical_dice(motion_img, gt_img, 2)

    pred_lv_dice = ff.np_categorical_dice(pred_img, gt_img, 1)
    pred_myo_dice = ff.np_categorical_dice(pred_img, gt_img, 2)
    print( motion_lv_dice, motion_myo_dice, pred_lv_dice, pred_myo_dice)

    # save the predicted image
    nb.save(nb.Nifti1Image(pred_img.astype(float), affine), os.path.join(save_sub,'pred_img.nii.gz'))




        
      
#         # dice
#         motion_lv_dice = util.np_categorical_dice(motion, gt, 1)
#         motion_myo_dice = util.np_categorical_dice(motion, gt, 2)
#         pred_lv_dice = util.np_categorical_dice(pred_img, gt, 1)
#         pred_myo_dice = util.np_categorical_dice(pred_img, gt, 2)

#         # Hausdorff distance
#         gt_lv_pixels = np.asarray(ff.count_pixel(gt,1)[1]); gt_myo_pixels = np.asarray(ff.count_pixel(gt,2)[1])
#         motion_lv_pixels = np.asarray(ff.count_pixel(motion,1)[1]); motion_myo_pixels = np.asarray(ff.count_pixel(motion,2)[1])
#         pred_lv_pixels = np.asarray(ff.count_pixel(pred_img,1)[1]); pred_myo_pixels = np.asarray(ff.count_pixel(pred_img,2)[1])

#         motion_lv_dis = directed_hausdorff(gt_lv_pixels, motion_lv_pixels)[0] * 1.25
#         motion_myo_dis = directed_hausdorff(gt_myo_pixels, motion_myo_pixels)[0] * 1.25
#         pred_lv_dis = directed_hausdorff(gt_lv_pixels, pred_lv_pixels)[0] * 1.25
#         pred_myo_dis = directed_hausdorff(gt_myo_pixels, pred_myo_pixels)[0] * 1.25

#         # print(motion_lv_dice, motion_myo_dice, pred_lv_dice, pred_myo_dice)

#         nb.save(nb.Nifti1Image(pred_img.astype(float), affine), os.path.join(motion_folder,'batch_0','pred_img.nii.gz'))
#         Results.append([patient_id, timeframe, motion_name, motion_lv_dice, motion_myo_dice, pred_lv_dice, pred_myo_dice,
#                         motion_lv_dis, motion_myo_dis, pred_lv_dis, pred_myo_dis])

# df = pd.DataFrame(Results, columns = ['Patient_ID','timeframe','motion_name','motion_lv_dice', 'motion_myo_dice', 'pred_lv_dice', 'pred_myo_dice',
#                                        'motion_lv_dis', 'motion_myo_dis', 'pred_lv_dis', 'pred_myo_dis'])
# df.to_excel(os.path.join(os.path.dirname(save_folder), 'comparison_DICE_batch0model_cross_val.xlsx'), index = False)