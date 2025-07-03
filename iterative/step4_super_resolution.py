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
object = 'final'
lv_model, lvmyo_model,threeclass_model = mm.EDSR_collection_ds()
model_file = threeclass_model

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


# # create models
input_shape = cg.input_dim + (3,)
model_inputs = [Input(input_shape)]
model_outputs=[]
final_image = EDSR.main_model(cg.output_dim, cg.num_classes, 128, 5, layer_name = 'edsr')(model_inputs[0])
model_outputs += [final_image]
model = Model(inputs = model_inputs,outputs = model_outputs)
model.load_weights(model_file)

# # pred LV LVmyo or all three classes
for i in range(0,x_list_predict.shape[0]):
    patient_id = str(patient_id_list[n[i]])
    timeframe = tf_list[n[i]]
    motion_name = motion_name_list[n[i]]
    batch = batch_list[n[i]]

    pred_path = os.path.join(save_folder, patient_id, timeframe, motion_name)
    data_path = os.path.join(cg.data_dir,'processed_HR_data', patient_id, timeframe)
    
    print(patient_id, timeframe, motion_name)

    filename = os.path.join(pred_path, 'pred_img_HR_' + object +'.nii.gz')

    # if os.path.isfile(filename) == 1:
    #     print('done;skip'); continue

    x = os.path.join(pred_path,'pred_img.nii.gz')
    y = os.path.join(data_path, 'HR_'+ timeframe+'_crop_60.nii.gz')

    final_pred = nb.load(filename).get_fdata()

    # if os.path.isfile(x) == 0:
    #     print('no image file input;skip'); continue

    # datagen = Generator.DataGenerator(np.asarray([x]),np.asarray([y]),patient_num = 1, 
    #                                 batch_size = cg.batch_size, 
    #                                 num_classes = 3,
    #                                 input_dimension = cg.input_dim,
    #                                 output_dimension = cg.output_dim, 
    #                                 shuffle = False,
    #                                 remove_slices = 'None',
    #                                 remove_label = True,
    #                                 relabel_RV = False,
    #                                 relabel_myo = False,
    #                                 slice_augment = False,
    #                                 num_classes_exist_after_remove = 3,)

    # pred = model.predict_generator(datagen, verbose = 1, steps = 1,)

   
    # final_pred = np.argmax(pred[0], axis = -1).astype(np.uint8)
    # pred_nb = nb.Nifti1Image(final_pred, nb.load(y).affine)
   
    # nb.save(pred_nb, filename) 

    # postprocessing (flip and clean):
    final_pred_flip = final_pred[:,:,[final_pred.shape[-1] - j for j in range(1,final_pred.shape[-1] + 1)]]
    nb.save(nb.Nifti1Image(final_pred_flip, nb.load(y).affine),os.path.join(pred_path, 'pred_img_HR_final_flip.nii.gz'))

    # clean
    slice_condition = np.load(os.path.join(cg.data_dir,'simulated_data_version2',patient_id, timeframe, 'lv_slice_condition.npy'), allow_pickle = True)
    lv_slices = np.asarray(slice_condition[2][0])

    hr_slices = np.arange(((lv_slices[0]+1) * 5 - 1), 60, 1)
    np.save(os.path.join(pred_path, 'HR_slice_condition.npy'), np.asarray(hr_slices))

    final_pred_flip_clean = np.zeros(final_pred_flip.shape)
    final_pred_flip_clean[:,:, hr_slices[0] : hr_slices[-1] + 1] = final_pred_flip[:,:, hr_slices[0] : hr_slices[-1] + 1]    
    nb.save(nb.Nifti1Image(final_pred_flip_clean, nb.load(y).affine),os.path.join(pred_path, 'pred_img_HR_final_flip_clean.nii.gz'))



# postprocessing (for LV and LVmyo):

# for i in range(0, x_list_predict.shape[0]):
#     patient_id = str(patient_id_list[n[i]])
#     timeframe = tf_list[n[i]]
#     motion_name = motion_name_list[n[i]]
#     batch = batch_list[n[i]]

#     pred_path = os.path.join(save_folder, patient_id, timeframe, motion_name)

#     if os.path.isfile(os.path.join(pred_path, 'pred_img_HR_LV.nii.gz')) == 0 or os.path.isfile(os.path.join(pred_path, 'pred_img_HR_LVmyo.nii.gz')) == 0:
#         print('no data; skip'); continue
    
#     lv = nb.load(os.path.join(pred_path, 'pred_img_HR_LV.nii.gz'))
#     affine = lv.affine; header = lv.header
#     lv = lv.get_fdata(); lv = np.round(lv); lv = lv.astype(int)

#     lvmyo = nb.load(os.path.join(pred_path, 'pred_img_HR_LVmyo.nii.gz')).get_data()
#     lv_myo = np.round(lvmyo); lv_myo = lv_myo.astype(int)

#     lvmyo[lvmyo == 1] = 2
#     lvmyo[lv==1] = 1

#     nb.save(nb.Nifti1Image(lvmyo, affine, header), os.path.join(pred_path, 'pred_img_HR_final.nii.gz'))


