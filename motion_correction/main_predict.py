#!/usr/bin/env python
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Trained_models.motion_correction_models as trained_models
import ResNet as resnet
import CMR_HFpEF_Analysis.Build_list_data_prepare.Build_list as Build_list
import Generator_motion 
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

cg = Defaults.Parameters()
mm = trained_models.trained_models()

# build lists
trial_name = 'Motion_ResNet_new'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_simulated_data_15normals.xlsx')
save_folder = os.path.join(cg.predict_dir, trial_name, 'images'); ff.make_folder([os.path.dirname(save_folder), save_folder])

b = Build_list.Build(data_sheet)
patient_id_list, patient_tf_list, motion_name_list, batch_list, _,_, _, _, gt_center_list,  _, x_list, _ = b.__build__(batch_list = [5])
n = np.arange(0,patient_id_list.shape[0],1)
x_list = x_list[n]; gt_center_list = gt_center_list[n]

# create model architecture:
input_shape = (128,128,12) + (1,)
model_inputs = [Input(input_shape)]
model_outputs=[]
center_x, center_y = resnet.main_model(nb_filters = [32,64,128,256], output_num = 6)(model_inputs[0])
model_outputs += [center_x, center_y]

# define model list:
model_files = mm.Motion_ResNet_collection()

# do prediction
for j in range(2,3):#len(model_files)):
  f = model_files[j]
  print(f)
  model= Model(inputs = model_inputs,outputs = model_outputs); model.load_weights(f)
    
  for i in range(0,x_list.shape[0]):
    patient_id = str(patient_id_list[n[i]])
    timeframe = patient_tf_list[n[i]]
    motion_name = motion_name_list[n[i]]
    batch = batch_list[n[i]]
    

    print(batch, patient_id, timeframe, motion_name)
    save_sub = os.path.join(save_folder, patient_id, timeframe, motion_name,'centers' )
    ff.make_folder([os.path.join(save_folder,patient_id), os.path.join(save_folder,patient_id, timeframe), os.path.join(save_folder, patient_id, timeframe, motion_name), save_sub])

    filename = 'pred_centers_' + str(j + 1) + '.npy'
      
    # done?
    if os.path.isfile(os.path.join(save_sub,filename)) == 1:
      print('done'); #continue

    datagen = Generator_motion.DataGenerator(np.asarray([x_list[i]]),
                                            np.asarray([gt_center_list[i]]),
                                            patient_num = 1, 
                                            batch_size = 1, 
                                            num_classes = 3,
                                            input_dimension = (128,128,12),
                                            output_dimension = (6,), 
                                            shuffle = False, 
                                            remove_label= True,
                                            relabel_myo = False,
                                            slice_augment = False,
                                            seed = 10,)
                                     
    pred_delta_x_cp, pred_delta_y_cp = model.predict_generator(datagen, verbose = 1, steps = 1,)

    pred_delta_x = np.concatenate([np.array([0]),np.asarray(pred_delta_x_cp).reshape(-1) / 2 * 128], axis = -1)
    pred_delta_y = np.concatenate([np.array([0]),np.asarray(pred_delta_y_cp).reshape(-1) / 2 * 128], axis = -1)
    predict = np.concatenate([pred_delta_x.reshape(1,-1), pred_delta_y.reshape(1,-1)],axis = 0)
     
    print(predict.shape, predict)

    np.save(os.path.join(save_sub,filename), predict)
