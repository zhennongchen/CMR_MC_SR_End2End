import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.iterative.Build_list as Build_list
import CMR_HFpEF_Analysis.super_resolution.Generator as Generator
import CMR_HFpEF_Analysis.super_resolution.EDSR as EDSR
import CMR_HFpEF_Analysis.Trained_models.super_resolution_models as super_resolution_models

import argparse
import os
import numpy as np
import nibabel as nb
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2

cg = Defaults.Parameters()
mm = super_resolution_models.trained_models()

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

for i in range(0, x_list_predict.shape[0]):
    patient_id = str(patient_id_list[n[i]])
    timeframe = tf_list[n[i]]
    motion_name = motion_name_list[n[i]]
    batch = batch_list[n[i]]

    data_path = os.path.join(cg.data_dir,'simulated_data_version2', patient_id, timeframe)
    pred_path = os.path.join(save_folder, patient_id, timeframe, motion_name)

    # load lv_slice_condition
    slice_condition = np.load(os.path.join(data_path, 'lv_slice_condition.npy'), allow_pickle = True)
    slice_exclude = slice_condition[1][0]

    # load predict
    pred = nb.load(os.path.join(pred_path,'pred_img_HR_final.nii.gz')).get_fdata()
    pred = np.round(pred); pred = pred.astype(int)

    # move it to the center
    center_mass = util.center_of_mass(pred,0,large = True)
    center_mass = [int(center_mass[0]),int(center_mass[1]),int(center_mass[2])]
    center_image = [ pred.shape[i] // 2 for i in range(0,len(pred.shape))]

    move = [ center_image[i] - center_mass[i] for i in range(0,len(center_mass))]
    img_move = util.move_3Dimage(pred, move)

    # downsample:
    pred_ds = util.downsample_in_z(pred,5)
    
    # save
    affine = nb.load(os.path.join(data_path,'ds/data.nii.gz')).affine
    nb.save(nb.Nifti1Image(pred_ds, affine), os.path.join(pred_path, 'pred_ds.nii.gz'))
    
    # flip and clean
    pred_ds = pred_ds[:,:,[pred_ds.shape[-1] - i for i in range(1,pred_ds.shape[-1] + 1)]]
    for s in slice_exclude:
        pred_ds[:,:,s] = np.zeros((pred_ds.shape[0],pred_ds.shape[1]))
    nb.save(nb.Nifti1Image(pred_ds, affine), os.path.join(pred_path, 'pred_ds_flip_clean.nii.gz'))

    