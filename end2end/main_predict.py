## Data Preparation 

# You should prepare the following before running this step. Please refer to the `example_data` folder for guidance:

# 1. **the data you want to corect/process** 

# 2. **A patient list** that enumerates all your cases
#    - To understand the standard format, please refer to the file:  
#      `example_data/Patient_list/patient_list.xlsx`

# Docker environment
# Please use `docker`, it will build a tensorflow-based container 
# make sure you have voxelmorph installed in the container

import CMR_MC_SR_End2End.Image_utils as util
import CMR_MC_SR_End2End.functions_collection as ff
import CMR_MC_SR_End2End.Build_list.Build_list as Build_list
import CMR_MC_SR_End2End.end2end.model as end2end
import CMR_MC_SR_End2End.end2end.Generator as Generator

import os
import numpy as np
import nibabel as nb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


# build lists
trial_name = 'end2end'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_simulated_data_15normals.xlsx')
save_folder = os.path.join(cg.predict_dir, trial_name, 'images'); ff.make_folder([os.path.dirname(save_folder), save_folder])

b = Build_list.Build(data_sheet)
patient_id_list,patient_tf_list,motion_name_list,_,_, y_HR_img_list,_,y_LR_img_list,_,_, x_list, center_list  = b.__build__(batch_list = [5])
n = np.arange(0,patient_id_list.shape[0],1)
x_list = x_list[n]; center_list = center_list[n]; y_LR_img_list = y_LR_img_list[n]; y_HR_img_list = y_HR_img_list[n]

# create model
input_shape = (128,128,12) + (1,)
model_inputs = [Input(input_shape)]
model_outputs=[]
combined_slices, final_LR_img, final_HR_img = end2end.main_model(cg.num_classes, cg.output_dim , nb_filters_motion_model = [32,64,128,256], output_num_motion_model = 2)(model_inputs[0])
model_outputs += [combined_slices, final_LR_img, final_HR_img ]
model = Model(inputs = model_inputs,outputs = model_outputs)

model_files = mm.end2end_collection()


results = []
for j in range(0,len(model_files)):
   f = model_files[j]
   print(f)
   model = Model(inputs = model_inputs,outputs = model_outputs)
   model.load_weights(f)

   for i in range(0, x_list.shape[0]):
      patient_id = patient_id_list[n[i]] 
      patient_tf  = patient_tf_list[n[i]]
      motion_name = motion_name_list[n[i]]

      print(patient_id, patient_tf, motion_name)

      folder = os.path.join(save_folder, str(patient_id), patient_tf, motion_name)
      ff.make_folder([os.path.join(save_folder, str(patient_id)), os.path.join(save_folder, str(patient_id), patient_tf), os.path.join(save_folder, str(patient_id), patient_tf, motion_name)])

      if os.path.isfile(os.path.join(folder, 'pred_img_LR_' + str(j + 1) +'.nii.gz')) == 1:
         print('done')
        
      else:
         datagen = Generator.DataGenerator(np.asarray([x_list[i]]),
                                   np.asarray([center_list[i]]),
                                   np.asarray([y_LR_img_list[i]]),
                                   np.asarray([y_HR_img_list[i]]),
                                   patient_num = 1, 
                                   batch_size = 1, 
                                   num_classes = 3,
                                   input_dimension = (128,128,12),
                                   shuffle = False, 
                                   remove_label= True,
                                   relabel_myo = False,
                                   slice_augment = False,
                                   seed = 10,)
         
         pred_movements, pred_LR_img, pred_HR_img = model.predict_generator(datagen, verbose = 1, steps = 1,)
         pred_movements = pred_movements[0,...]
         u_gt_nii = nb.load(y_HR_img_list[i])
         pred_HR_img = np.argmax(pred_HR_img[0,...], axis = -1).astype(np.uint8)
         u_gt_nii_LR = nb.load(y_LR_img_list[i])
         pred_LR_img = np.argmax(pred_LR_img[0,...], axis = -1).astype(np.uint8)
         
         
         # save pred_movements as numpy
         # np.save(os.path.join(folder, 'pred_vector_' + str(j + 1) +'.npy'), pred_movements)

         # save pred_LR_img as nifti
         nb.save(nb.Nifti1Image(pred_LR_img.astype(float), u_gt_nii_LR.affine, u_gt_nii_LR.header), os.path.join(folder, 'pred_img_LR_' + str(j + 1) +'.nii.gz'))
         
         # # save pred_HR_img as nifti
         # nb.save(nb.Nifti1Image(pred_HR_img.astype(float), u_gt_nii.affine, u_gt_nii.header), os.path.join(folder, 'pred_img_HR_' + str(j + 1) +'.nii.gz'))
      




