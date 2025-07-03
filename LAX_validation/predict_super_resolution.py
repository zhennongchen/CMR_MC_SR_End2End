import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Build_list_data_prepare.Build_list as Build_list
import CMR_HFpEF_Analysis.Trained_models.super_resolution_models as super_resolution_models
import CMR_HFpEF_Analysis.super_resolution.Generator as Generator
import CMR_HFpEF_Analysis.super_resolution.EDSR  as EDSR

import os
import numpy as np
import nibabel as nb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
mm = super_resolution_models.trained_models()
cg = Defaults.Parameters()

data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_SunnyBrooks.xlsx')
save_folder = os.path.join(cg.predict_dir,'Sunny_Brooks' , 'solution_1')
ff.make_folder([save_folder])

# build list
b = Build_list.Build(data_sheet)
patient_id_list,patient_tf_list,motion_name_list,_, _,y_list, _,_,_,_, x_list, _ = b.__build__(batch_list = [5])
n = np.arange(0, patient_id_list.shape[0],1)
x_list = x_list[n]; y_list = y_list[n]

# create model
input_shape = cg.input_dim + (cg.num_classes,)
model_inputs = [Input(input_shape)]
model_outputs=[]
final_image = EDSR.main_model(cg.output_dim, cg.num_classes, 128, 5, layer_name = 'edsr')(model_inputs[0])
model_outputs += [final_image]
model = Model(inputs = model_inputs,outputs = model_outputs)

# model_files = mm.EDSR_super_resolution()
model_files = mm.EDSR_two_tasks()


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
    #      print('done')

    x = os.path.join(cg.data_dir, 'Sunny_Brooks/sunnybrooks_nii', patient_id, 'SAX_ED_for_DL.nii.gz')
    y = os.path.join(cg.data_dir, 'contour_dataset/processed_HR_data', '100/ED' , 'HR_ED_zoomed_crop_flip_clean.nii.gz')

    datagen = Generator.DataGenerator(np.asarray([x]),np.asarray([y]), 
                                      patient_num = 1,
                                      batch_size = 1, 
                                      num_classes = 3,
                                      input_dimension = cg.input_dim,
                                      output_dimension = cg.output_dim, 
                                      shuffle = False, 
                                      remove_slices = 'None',
                                      remove_label= True,
                                      relabel_RV = False,
                                      relabel_myo = False,
                                      augment = False,
                                      seed = 12,
                                     )

    pred = model.predict_generator(datagen, verbose = 1, steps = 1,)
        
    u_gt_nii = nb.load(y)
    final_pred = np.argmax(pred[0,...], axis = -1).astype(np.uint8)
    print(final_pred.shape, np.unique(final_pred))

    # save image
    nb.save(nb.Nifti1Image(final_pred.astype(float), u_gt_nii.affine), os.path.join(folder, 'pred_img_HR_' + str(j + 1) +'.nii.gz'))

       
   
  
    
 

        
        
