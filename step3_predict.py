## Data Preparation 

# You should prepare the following before running this step. Please refer to the `example_data` folder for guidance:

# 1. **the data you want to corect/process** 

# 2. **A patient list** that enumerates all your cases
#    - To understand the standard format, please refer to the file:  
#      `example_data/Patient_list/patient_list.xlsx`

# it uses trained model to do CMR motion correction and super-resolution. it save "pred_img_LR.nii.gz" as motion-corrected image (still in low z-resolution) and "pred_img_HR.nii.gz" as super-resolutioned image (in high resolution).



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

main_path = '/mnt/camca_NAS/CMR_processing/' # replace with your own

# set trial name and pretrained model file
trial_name = 'end2end'
pretrained_model_file = os.path.join(main_path, 'example_data/models', trial_name, 'models', 'model-final.hdf5')  # replace with your own
save_folder = os.path.join(main_path,'example_data/models', trial_name, 'pred_images')
ff.make_folder([save_folder])

# set patient list
data_sheet = os.path.join(main_path,'example_data/Patient_list/patient_list.xlsx')

b = Build_list.Build(data_sheet)
patient_id_list,patient_tf_list,motion_name_list,_, y_HR_img_list,y_LR_img_list,_,x_list, center_list  = b.__build__(batch_list = [0])


# create model
input_shape = (128,128,12) + (1,)
model_inputs = [Input(input_shape)]
model_outputs=[]
combined_slices, final_LR_img, final_HR_img = end2end.main_model(nb_class = 3, final_HR_img_dim = (128,128,60) , nb_filters_motion_model = [32,64,128,256], output_num_motion_model = 2)(model_inputs[0])
model_outputs += [combined_slices, final_LR_img, final_HR_img ]
model = Model(inputs = model_inputs,outputs = model_outputs)


model = Model(inputs = model_inputs,outputs = model_outputs)
model.load_weights(pretrained_model_file)

for i in range(0, x_list.shape[0]):
   patient_id = patient_id_list[i] 
   patient_tf  = patient_tf_list[i]
   motion_name = motion_name_list[i]

   print(patient_id, patient_tf, motion_name)

   folder = os.path.join(save_folder, str(patient_id), patient_tf, motion_name)
   ff.make_folder([os.path.join(save_folder, str(patient_id)), os.path.join(save_folder, str(patient_id), patient_tf), os.path.join(save_folder, str(patient_id), patient_tf, motion_name)])

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
   # np.save(os.path.join(folder, 'pred_vector.npy'), pred_movements)

   # save motion_corrected image as nifti
   nb.save(nb.Nifti1Image(pred_LR_img.astype(float), u_gt_nii_LR.affine, u_gt_nii_LR.header), os.path.join(folder, 'pred_img_LR.nii.gz'))
         
   # save pred_HR_img as nifti
   nb.save(nb.Nifti1Image(pred_HR_img.astype(float), u_gt_nii.affine, u_gt_nii.header), os.path.join(folder, 'pred_img_HR.nii.gz'))
      




