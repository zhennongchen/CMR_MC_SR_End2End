#!/usr/bin/env python
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Trained_models.motion_correction_models as motion_correction_models
import CMR_HFpEF_Analysis.motion_correction.ResNet as resnet
import CMR_HFpEF_Analysis.motion_correction.Generator_motion as Generator_motion
import CMR_HFpEF_Analysis.sunny_brooks.Build_list as Build_list
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


cg = Defaults.Parameters()
mm = motion_correction_models.trained_models()

###### define study and data
trial_name = 'Iteration_C'
study_set = 'Sunny_Brooks'
iteration_num = 1
save_folder = os.path.join(cg.predict_dir,study_set, trial_name, 'round_'+str(iteration_num), 'images')
ff.make_folder([os.path.dirname(os.path.dirname(save_folder)),os.path.dirname(save_folder),save_folder])

###### define data sheet
data_sheet = os.path.join(cg.data_dir,'Patient_list/Sunny_Brooks.xlsx')

###### define model list:
center_x_files, center_y_files = mm.Motion_ResNet_HR_collection()
files = [center_x_files, center_y_files]

###### build patient list
print('Build List...')
b = Build_list.Build(data_sheet)
input_list, patient_id_list, patient_tf_list = b.__build__()
n = np.arange(0,patient_id_list.shape[0],1)
input_list = input_list[n]

###### create model architecture:
input_shape = (128,128,60) + (1,)
model_inputs = [Input(input_shape)]
model_outputs=[]
center_x, center_y = resnet.main_model(nb_filters = [32,64,128,256],output_num = 9)(model_inputs[0])
model_outputs += [center_x, center_y]


###### do prediction
for jj in range(0,len(center_x_files)):
    for j in range(0,len(files)): #center_x: j=0, center_y: j=1,
        f = files[j][jj]
        print(f)
        model= Model(inputs = model_inputs,outputs = model_outputs); model.load_weights(f)
        
        for i in range(0,patient_id_list.shape[0]):
            patient_id = str(patient_id_list[n[i]])
            timeframe = patient_tf_list[n[i]]

            save_sub = os.path.join(save_folder, patient_id, timeframe, 'centers_2' )
            ff.make_folder([os.path.join(save_folder,patient_id), os.path.join(save_folder,patient_id, timeframe), save_sub])

            if j == 0:
                filename = 'center_x_' + str(jj) +'.npy'
            elif j == 1:
                filename = 'center_y_' + str(jj) +'.npy'
            
            # done?
            # if os.path.isfile(os.path.join(save_sub,filename)) == 1:
            #     print('done'); continue
            
            x = os.path.join(save_folder,patient_id,timeframe, 'pred_img_HR_final_flip.nii.gz')
            y = os.path.join(cg.data_dir,'simulated_data_version2/155/ED/ds/HR/HR_centerlist_LV_slices.npy')

            datagen = Generator_motion.DataGenerator(np.asarray([x]), np.asarray([y]),patient_num =1,batch_size = cg.batch_size, 
                                            num_classes = 2,
                                            input_dimension = (128,128,60),
                                            output_dimension = (9,), 
                                            shuffle = False,
                                            remove_slices = 'LV',
                                            remove_pixel_num_threshold = 50,
                                            remove_label= True,
                                            relabel_myo = True,
                                            slice_augment = False,
                                            )
                                            
            pred_delta_x_cp, pred_delta_y_cp = model.predict_generator(datagen, verbose = 1, steps = 1,)

            pred_delta_x = np.concatenate([np.array([0]),np.asarray(pred_delta_x_cp).reshape(-1) / 2 * 128], axis = -1)
            pred_delta_y = np.concatenate([np.array([0]),np.asarray(pred_delta_y_cp).reshape(-1) / 2 * 128], axis = -1)
            print(pred_delta_x)
            print(pred_delta_y)
    

            np.save(os.path.join(save_sub,filename), np.concatenate([pred_delta_x.reshape(1,-1), pred_delta_y.reshape(1,-1)],axis = 0))