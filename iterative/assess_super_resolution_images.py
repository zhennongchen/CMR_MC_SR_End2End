import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.data_simulation.transformation as transform
import CMR_HFpEF_Analysis.motion_correction.Bspline as Bspline
import CMR_HFpEF_Analysis.iterative.Build_list as Build_list
import os
import numpy as np
import pandas as pd
import nibabel as nb
from scipy.spatial.distance import directed_hausdorff
cg = Defaults.Parameters()

###### define study and data
trial_name = 'Iteration_C'
study_set = 'Combined'
iteration_num = 2
save_folder = os.path.join(cg.predict_dir,study_set, trial_name, 'round_'+str(iteration_num), 'images')
ff.make_folder([os.path.dirname(os.path.dirname(save_folder)),os.path.dirname(save_folder),save_folder])

###### define data sheet
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_HR_LVslices_motion_flip_clean_7_slice_10_normal_IterationC.xlsx')

###### build patient list
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [5]
x_list_predict, y_list_predict, patient_id_list, tf_list, motion_name_list, batch_list,_,_ = b.__build__(batch_list = batches)
n = np.arange(0,patient_id_list.shape[0],1)
x_list_predict = x_list_predict[n]

# calculate DICE and HD:
Results = []
for i in range(0,x_list_predict.shape[0]):
    patient_id = str(patient_id_list[n[i]])
    timeframe = tf_list[n[i]]
    motion_name = motion_name_list[n[i]]
    batch = batch_list[n[i]]

    data_path = os.path.join(cg.data_dir,'processed_HR_data', patient_id, timeframe)
    pred_path = os.path.join(save_folder, patient_id, timeframe, motion_name)

    # whether there is data?
    if os.path.isfile(os.path.join(pred_path,'pred_img_HR.nii.gz')) == 0:
        print('not done; skip'); continue

    # load pred 
    pred = nb.load(os.path.join(pred_path,'pred_img_HR.nii.gz')).get_fdata()
    pred = np.round(pred); pred = pred.astype(int)

    # load gt image
    gt = nb.load(os.path.join(data_path, 'HR_'+timeframe+'_crop_60.nii.gz'))
    affine = gt.affine; gt = gt.get_fdata();  gt = np.round(gt); gt = gt.astype(int); gt = util.relabel(gt, 4, 0)

    ##### Calculate DICE
    # move for best Dice
    if os.path.isfile(os.path.join(pred_path, 'move_for_dice_original_orient.npy')) == 0:
        max_dice, ds, pred_for_dice = util.Dice_optim(pred, gt, [1,2], d = 4)
        np.save(os.path.join(pred_path, 'move_for_dice_original_orient.npy'), np.asarray(ds))
    else:
        ds = np.load(os.path.join(pred_path, 'move_for_dice_original_orient.npy'), allow_pickle = True)
        pred_for_dice = util.move_3Dimage(pred, (ds[0], ds[1], 0))
    # dice
    pred_lv_dice = ff.np_categorical_dice(pred_for_dice, gt, 1)
    pred_myo_dice = ff.np_categorical_dice(pred_for_dice, gt, 2)

    ##### Calculate Hausdorff distance 
    # move for HD
    if os.path.isfile(os.path.join(pred_path,'move_for_HD_original_orient.npy')) == 0:
        _, ds2, pred_for_HD = util.HD_optim(pred, gt, min  = False,k_list = [1,2],s d = 4)
        np.save(os.path.join(pred_path, 'move_for_HD_original_orient.npy'), np.asarray(ds2))
    else:
        ds2 = np.load(os.path.join(pred_path,'move_for_HD_original_orient.npy'), allow_pickle = True)
        pred_for_HD = util.move_3Dimage(pred, (ds2[0], ds2[1], 0))

    gt_lv_pixels = np.asarray(ff.count_pixel(gt,1)[1]); gt_myo_pixels = np.asarray(ff.count_pixel(gt,2)[1])
    pred_lv_pixels = np.asarray(ff.count_pixel(pred_for_HD,1)[1]); pred_myo_pixels = np.asarray(ff.count_pixel(pred_for_HD,2)[1])

    pred_lv_dis = ff.HD( pred_lv_pixels, gt_lv_pixels, 1.25, min = False)
    pred_myo_dis = ff.HD( pred_myo_pixels, gt_myo_pixels,1.25, min = False)

    print(patient_id, timeframe, motion_name, pred_lv_dice, pred_myo_dice, pred_lv_dis,pred_myo_dis)
    Results.append([patient_id, timeframe, motion_name,  pred_lv_dice, pred_myo_dice, pred_lv_dis, pred_myo_dis])
    
    df = pd.DataFrame(Results, columns = ['Patient_ID','timeframe','motion_name','pred_lv_dice', 'pred_myo_dice', 'pred_lv_dis', 'pred_myo_dis'])
    df = df.round(decimals=3)
    df.to_excel(os.path.join(os.path.dirname(save_folder), 'comparison_super_resolution_test_original_orient.xlsx'), index = False)
