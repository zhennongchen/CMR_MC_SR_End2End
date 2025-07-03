#!/usr/bin/env python

import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Build_list_data_prepare.Build_list as Build_list
import CMR_HFpEF_Analysis.end2end.model as end2end
import CMR_HFpEF_Analysis.end2end.Generator as Generator
import CMR_HFpEF_Analysis.Trained_models.end2end as end2end_models

import os
import numpy as np
import nibabel as nb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
mm = end2end_models.trained_models()
cg = Defaults.Parameters()

# build lists
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_SunnyBrooks.xlsx')
save_folder = os.path.join(cg.predict_dir,'Sunny_Brooks' , 'solution_4')
ff.make_folder([save_folder])

# build list
b = Build_list.Build(data_sheet)
patient_id_list,patient_tf_list,motion_name_list,_, _,y_list, _,_,_,_, x_list, _ = b.__build__(batch_list = [5])
n = np.arange(0,patient_id_list.shape[0],1)
x_list = x_list[n]; y_list = y_list[n]

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

      print(patient_id)

      folder = os.path.join(save_folder, patient_id)
      ff.make_folder([folder])

      # if os.path.isfile(os.path.join(folder, 'pred_img_HR_' + str(j + 1) +'.nii.gz')) == 1:
      #    print('done')
      #    continue
        
      x = os.path.join(cg.data_dir, 'Sunny_Brooks/sunnybrooks_nii', patient_id, 'corrected/SAX_ED_corrected_for_DL.nii.gz')
      center = os.path.join(cg.data_dir, 'contour_dataset/simulated_data', '100/ED' , 'ds/centerlist.npy')
      y_HR_img = os.path.join(cg.data_dir, 'contour_dataset/processed_HR_data', '100/ED' , 'HR_ED_zoomed_crop_flip_clean.nii.gz')

      datagen = Generator.DataGenerator(np.asarray([x]),
                                   np.asarray([center]),
                                   np.asarray([x]),
                                   np.asarray([y_HR_img]),
                                   patient_num = 1, 
                                   batch_size = 1, 
                                   num_classes = 3,
                                   input_dimension = (128,128,12),
                                   shuffle = False, 
                                   remove_label= True,
                                   relabel_myo = False,
                                   slice_augment = False,
                                   seed = 10,)
         
      pred_movements, _, pred_HR_img = model.predict_generator(datagen, verbose = 1, steps = 1,)
      pred_movements = pred_movements[0,...]
      u_gt_nii = nb.load(y_HR_img)
      pred_HR_img = np.argmax(pred_HR_img[0,...], axis = -1).astype(np.uint8)
         
      # save pred_movements as numpy
      np.save(os.path.join(folder, 'pred_vector_' + str(j + 1) +'.npy'), pred_movements)
      # save pred_HR_img as nifti
      nb.save(nb.Nifti1Image(pred_HR_img.astype(float), u_gt_nii.affine, u_gt_nii.header), os.path.join(folder, 'pred_img_HR_' + str(j + 1) +'.nii.gz'))
      




