#!/usr/bin/env python

import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Build_list_data_prepare.Build_list as Build_list
import CMR_HFpEF_Analysis.STN.model as STN_model
import CMR_HFpEF_Analysis.Trained_models.stn as stn
import Generator

import os
import numpy as np
import nibabel as nb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
mm = stn.trained_models()
cg = Defaults.Parameters()

 
# build lists
trial_name = 'STN_vector_img'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_simulated_data_15normals.xlsx')
save_folder = os.path.join(cg.predict_dir, trial_name, 'images'); ff.make_folder([os.path.dirname(save_folder), save_folder])

b = Build_list.Build(data_sheet)
patient_id_list,patient_tf_list,motion_name_list,_, _,_, _,img_list,_,_, x_list, center_list = b.__build__(batch_list = [5])
n = np.arange(75,76,1)
x_list = x_list[n]; center_list = center_list[n]; img_list = img_list[n]

print(x_list[0:3], center_list[0:3], img_list[0:3])

# create model
input_shape = (128,128,12) + (1,)
model_inputs = [Input(input_shape)]
model_outputs=[]
combined_slices, final_img  = STN_model.main_model(nb_filters = [32,64,128,256], output_num = 2)(model_inputs[0])
model_outputs += [combined_slices, final_img]

model_files = mm.STN_vector_img_models()


results = []
for j in range(0,1):#len(model_files)):
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

      if 1==2:#os.path.isfile(os.path.join(folder, 'pred_vector_' + str(j + 1) +'.npy')) == 1:
         print('done')
         pred_movements = np.load(os.path.join(folder, 'pred_vector_' + str(j + 1) +'.npy'), allow_pickle = True)
      else:
         datagen = Generator.DataGenerator(np.asarray([x_list[i]]),
                                             np.asarray([center_list[i]]),
                                             np.asarray([img_list[i]]),
                                             patient_num = 1,
                                             batch_size = 1,
                                             num_classes = 3,
                                             input_dimension = (128,128,12),
                                             output_dimension = (12,2), 
                                             shuffle = True, 
                                             remove_label= True,
                                             relabel_myo = False,
                                             slice_augment = True,
                                             seed = 10,
                                             )
         
         pred_movements, pred_img = model.predict_generator(datagen, verbose = 1, steps = 1,)
         pred_img = pred_img[0,...]
         pred_movements = pred_movements[0,...]
         # np.save(os.path.join(folder, 'pred_vector_' + str(j + 1) +'.npy'), pred_movements)
         nb.save(nb.Nifti1Image(pred_img, nb.load(img_list[i]).affine), os.path.join(folder,'pred_LR_img.nii.gz'))

      




