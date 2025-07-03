#!/usr/bin/env python

import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Hyperparameters as hyper
import CMR_HFpEF_Analysis.Build_list_data_prepare.Build_list as Build_list
import CMR_HFpEF_Analysis.end2end.model as end2end
import CMR_HFpEF_Analysis.end2end.Generator as Generator

import argparse
import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from contextlib import redirect_stdout

cg = Defaults.Parameters()

def train(val_batch,trial_name,data_sheet, epochs, load_model_file):
 
  # build lists
  print('Build List...')
  batch_list = [0,1,2,3,5]; batch_list.pop(val_batch)
  train_batch = batch_list; print(train_batch)
  b = Build_list.Build(data_sheet)
  _,_,_,_,_, y_HR_img_list_trn,_,y_LR_img_list_trn,_,_, x_list_trn, center_list_trn = b.__build__(batch_list = train_batch)
  _,_,_,_,_, y_HR_img_list_val,_,y_LR_img_list_val,_,_, x_list_val, center_list_val = b.__build__(batch_list = [val_batch])

  # n = np.arange(0,1,1)
  # x_list_trn = x_list_trn[n]; center_list_trn = center_list_trn[n]; y_LR_img_list_trn = y_LR_img_list_trn[n]; y_HR_img_list_trn = y_HR_img_list_trn[n]
  # x_list_val = x_list_trn; center_list_val = center_list_trn; y_LR_img_list_val = y_LR_img_list_trn; y_HR_img_list_val = y_HR_img_list_trn

  print(x_list_trn[0:3],center_list_trn[0:3], y_LR_img_list_trn[0:3],  y_HR_img_list_trn[0:3], x_list_val[0:3],center_list_val[0:3], y_LR_img_list_val[0:3])

  # create model
  print('Create Model...')
  input_shape = (128,128,12) + (1,)
  model_inputs = [Input(input_shape)]
  model_outputs=[]
  combined_slices, final_LR_img, final_HR_img = end2end.main_model(cg.num_classes, cg.output_dim , nb_filters_motion_model = [32,64,128,256], output_num_motion_model = 2)(model_inputs[0])
  model_outputs += [combined_slices, final_LR_img, final_HR_img ]
  model = Model(inputs = model_inputs,outputs = model_outputs)

  if load_model_file != None:
    print('\n\n',load_model_file)
    model.load_weights(load_model_file)

  # compile model
  print('Compile Model...')
  opt = Adam(lr = 1e-4)
  weights = [1,1,1]
  model.compile(optimizer= opt, 
                loss = ['MAE', hyper.dice_loss_one_class, hyper.dice_loss_one_class],
                loss_weights = weights,)
  print(weights)

  # set callbacks
  print('Set callbacks...')
  model_fld = os.path.join(cg.model_dir,trial_name,'models','batch_'+str(val_batch))
  model_name = 'model' 
  filepath=os.path.join(model_fld,  model_name +'-{epoch:03d}.hdf5')
  print('filepath is: ',filepath)
  ff.make_folder([os.path.dirname(os.path.dirname(model_fld)), os.path.dirname(model_fld), model_fld, os.path.join(os.path.dirname(os.path.dirname(model_fld)), 'logs')])
  csv_logger = CSVLogger(os.path.join(os.path.dirname(os.path.dirname(model_fld)), 'logs',model_name + '_batch'+ str(val_batch) + '_training-log.csv')) # log will automatically record the train_accuracy/loss and validation_accuracy/loss in each epoch
  callbacks = [csv_logger,
                    ModelCheckpoint(filepath,          
                                    monitor='val_loss',
                                    save_best_only=False,),
                    LearningRateScheduler(hyper.learning_rate_step_decay_classic),   # learning decay
                    ]

  # save model summnary
  with open(os.path.join(cg.model_dir, trial_name,'model_summary.txt'), 'w') as f:
    with redirect_stdout(f):
      model.summary()

  # Fit
  print('Fit model...')
  datagen = Generator.DataGenerator(x_list_trn,
                                    center_list_trn,
                                    y_LR_img_list_trn,
                                    y_HR_img_list_trn,
                                    patient_num = x_list_trn.shape[0], 
                                    batch_size = cg.batch_size, 
                                    num_classes = 3,
                                    input_dimension = (128,128,12),
                                    shuffle = True, 
                                    remove_label= True,
                                    relabel_myo = False,
                                    slice_augment = True,
                                    seed = 10,
                                     )

  valgen = Generator.DataGenerator(x_list_val,
                                   center_list_val,
                                   y_LR_img_list_val,
                                   y_HR_img_list_val,
                                   patient_num = x_list_val.shape[0], 
                                   batch_size = cg.batch_size, 
                                   num_classes = 3,
                                   input_dimension = (128,128,12),
                                   shuffle = False, 
                                   remove_label= True,
                                   relabel_myo = False,
                                   slice_augment = False,
                                   seed = 10,
                                     )

  model.fit_generator(generator = datagen,
                        epochs = epochs,
                        validation_data = valgen,
                        callbacks = callbacks,
                        verbose = 1,
                        )


def main(val_batch):
    """These are the main training settings. Set each before running
    this file."""
    
    trial_name = 'end2end'
    data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_simulated_data_15normals.xlsx')
    
    load_model_file = os.path.join(cg.model_dir,'end2end','models/batch_1_initial','model-030.hdf5')
   
    epochs = 200

    train(val_batch,trial_name, data_sheet, epochs, load_model_file)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int)
  args = parser.parse_args()
  
  if args.batch is not None:
    assert(0 <= args.batch < 5)

  main(args.batch)
