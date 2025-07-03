## Data Preparation 

# You should prepare the following before running this step. Please refer to the `example_data` folder for guidance:

# 1. **paired data** from step 1

# 2. **A patient list** that enumerates all your cases
#    - To understand the standard format, please refer to the file:  
#      `example_data/Patient_list/patient_list.xlsx`


# Docker environment
# Please use `docker`, it will build a tensorflow-based container 
# make sure you have voxelmorph installed in the container



# main code
import os
import sys
sys.path.append('/workspace/Documents')  ### remove this if not needed!

import argparse
import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

import CMR_MC_SR_End2End.Image_utils as util
import CMR_MC_SR_End2End.functions_collection as ff
import CMR_MC_SR_End2End.Hyperparameters as hyper
import CMR_MC_SR_End2End.Build_list.Build_list as Build_list
import CMR_MC_SR_End2End.end2end.model as end2end
import CMR_MC_SR_End2End.end2end.Generator as Generator

main_path = '/mnt/camca_NAS/CMR_processing/' # replace with your own

# set trial name and pretrained model file
trial_name = 'end2end'
pretrained_model_file = None

save_folder = os.path.join(main_path,'example_data/models', trial_name)
ff.make_folder([save_folder, os.path.join(save_folder, 'logs'), os.path.join(save_folder, 'models')])

# set patient list
data_sheet = os.path.join(main_path,'example_data/Patient_list/patient_list.xlsx')

b = Build_list.Build(data_sheet)
_,_,_,_, y_HR_img_list_trn,y_LR_img_list_trn,_, x_list_trn, center_list_trn = b.__build__(batch_list = [0])
_,_,_,_, y_HR_img_list_val,y_LR_img_list_val,_,x_list_val, center_list_val = b.__build__(batch_list = [0]) # same as trn, just as example


# main code, no need to change
# create model
print('Create Model...')
input_shape = (128,128,12) + (1,)
model_inputs = [Input(input_shape)]
model_outputs=[]

# default hyperparameters
combined_slices, final_LR_img, final_HR_img = end2end.main_model(nb_class = 3, final_HR_img_dim = (128,128,60) , nb_filters_motion_model = [32,64,128,256], output_num_motion_model = 2)(model_inputs[0])
model_outputs += [combined_slices, final_LR_img, final_HR_img ]
model = Model(inputs = model_inputs,outputs = model_outputs)

if pretrained_model_file != None:
    print('\n\n',pretrained_model_file)
    model.load_weights(pretrained_model_file)

# compile
print('Compile Model...')
opt = Adam(lr = 1e-4)
weights = [1,1,1]
model.compile(optimizer= opt, 
                loss = ['MAE', hyper.dice_loss_one_class, hyper.dice_loss_one_class],
                loss_weights = weights,)

# set callbacks
print('Set callbacks...')
model_fld = os.path.join(save_folder, 'models')
filepath=os.path.join(model_fld,  'model-{epoch:03d}.hdf5')
  
csv_logger = CSVLogger(os.path.join(save_folder, 'logs','training-log.csv')) 
callbacks = [csv_logger,
            ModelCheckpoint(filepath,          
                            monitor='val_loss',
                            save_best_only=False,),
            LearningRateScheduler(hyper.learning_rate_step_decay_classic),   # learning decay
            ]


datagen = Generator.DataGenerator(x_list_trn,
                                center_list_trn,
                                y_LR_img_list_trn,
                                y_HR_img_list_trn,
                                patient_num = x_list_trn.shape[0], 
                                batch_size = 1,
                                num_classes = 3,
                                input_dimension = (128,128,12),
                                shuffle = True, 
                                remove_label= True,
                                slice_augment = True,
                                seed = 10,)

valgen = Generator.DataGenerator(x_list_val,
                                center_list_val,
                                y_LR_img_list_val,
                                y_HR_img_list_val,
                                patient_num = x_list_val.shape[0], 
                                batch_size = 1, 
                                num_classes = 3,
                                input_dimension = (128,128,12),
                                remove_label= True,
                                slice_augment = False,
                                seed = 10,)

model.fit_generator(generator = datagen,
                        epochs = 200,
                        validation_data = valgen,
                        callbacks = callbacks,
                        verbose = 1,)