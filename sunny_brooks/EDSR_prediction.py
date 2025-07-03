#!/usr/bin/env python
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.super_resolution.Generator as Generator
import CMR_HFpEF_Analysis.super_resolution.EDSR as EDSR
import CMR_HFpEF_Analysis.Trained_models.super_resolution_models as super_resolution_models
import CMR_HFpEF_Analysis.sunny_brooks.Build_list as Build_list
import os
import numpy as np
import nibabel as nb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


cg = Defaults.Parameters()
mm = super_resolution_models.trained_models()

###### define study and data
trial_name = 'EDSR'
study_set = 'Sunny_Brooks'
save_folder = os.path.join(cg.predict_dir,study_set, trial_name, 'images')
ff.make_folder([os.path.dirname(save_folder),save_folder])

###### define data sheet
data_sheet = os.path.join(cg.data_dir,'Patient_list/Sunny_Brooks.xlsx')

###### define model list:
lv_model, lvmyo_model,threeclass_model = mm.EDSR_collection_normal_motion()
model_file = threeclass_model

###### build patient list
print('Build List...')
b = Build_list.Build(data_sheet)
input_list, patient_id_list, patient_tf_list = b.__build__()
n = np.arange(0,patient_id_list.shape[0],1)
input_list = input_list[n]

###### create model architecture:
input_shape = cg.input_dim + (cg.num_classes,)
model_inputs = [Input(input_shape)]
model_outputs=[]
final_image = EDSR.main_model(cg.output_dim, cg.num_classes, 128, 5, layer_name = 'edsr')(model_inputs[0])
model_outputs += [final_image]
model = Model(inputs = model_inputs,outputs = model_outputs)
model.load_weights(model_file)


###### do prediction
for i in range(0,patient_id_list.shape[0]):
    patient_id = patient_id_list[n[i]]
    timeframe = patient_tf_list[n[i]]

    pred_path = os.path.join(save_folder, patient_id, timeframe)
    ff.make_folder([os.path.join(save_folder, patient_id), pred_path])
    
    print(patient_id, timeframe)

    filename = os.path.join(pred_path, 'pred.nii.gz')

    # if os.path.isfile(filename) == 1:
    #     print('done;skip'); continue

    x = os.path.join(os.path.dirname(input_list[i]), 'img_class_LR_processed.nii.gz')
    y = os.path.join(cg.data_dir, 'processed_HR_data/155/ED/HR_ED_crop_60.nii.gz') # standard random one


    datagen = Generator.DataGenerator(np.asarray([x]),np.asarray([y]),patient_num = 1, 
                                    batch_size = cg.batch_size, 
                                    num_classes = 3,
                                    input_dimension = cg.input_dim,
                                    output_dimension = cg.output_dim, 
                                    shuffle = False,
                                    remove_slices = 'None',
                                    remove_label = True,
                                    relabel_RV = False,
                                    relabel_myo = False,
                                    slice_augment = False,
                                    num_classes_exist_after_remove = 3,)

    pred = model.predict_generator(datagen, verbose = 1, steps = 1,)

   
    final_pred = np.argmax(pred[0], axis = -1).astype(np.uint8)
    pred_nb = nb.Nifti1Image(final_pred, nb.load(y).affine)
   
    nb.save(nb.Nifti1Image(final_pred, nb.load(y).affine), filename) 


