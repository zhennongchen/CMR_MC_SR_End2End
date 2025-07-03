#!/usr/bin/env python

import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Hyperparameters as hyper
import ResNet as resnet
import CMR_HFpEF_Analysis.Build_list_data_prepare.Build_list as Build_list
import Generator_motion 
import Bspline
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
from contextlib import redirect_stdout

cg = Defaults.Parameters()

def train(val_batch,trial_name,data_sheet, epochs, load_model_file):
 
  # build lists
  print('Build List...')
  batch_list = [0,1,2,3,4]; batch_list.pop(val_batch)
  train_batch = batch_list
  b = Build_list.Build(data_sheet)
  _,_,_,_, _,_,_,_,center_list_trn,_, x_list_trn, _ = b.__build__(batch_list = train_batch)
  _,_,_,_, _,_,_,_,center_list_val,_, x_list_val, _ = b.__build__(batch_list = [val_batch])
  # n = np.arange(0,1,1)
  # x_list_trn = x_list_trn[n]; center_list_trn = center_list_trn[n]
  # x_list_val = x_list_trn[n]; center_list_val = center_list_trn[n]

  print(x_list_trn.shape, x_list_val.shape,x_list_trn[0:3],center_list_trn[0:3],x_list_val[0:3],center_list_val[0:3])

  # create model
  print('Create Model...')
  input_shape = (128,128,12) + (1,)
  model_inputs = [Input(input_shape)]
  model_outputs=[]
  center_x, center_y= resnet.main_model(nb_filters = [32,64,128,256], output_num = 6)(model_inputs[0])
  model_outputs += [center_x, center_y]
  model = Model(inputs = model_inputs,outputs = model_outputs)

  if load_model_file != None:
    print('\n\n',load_model_file)
    model.load_weights(load_model_file)

  # compile model
  print('Compile Model...')
  opt = Adam(lr = 1e-4)
  weights = [1,1]
  model.compile(optimizer= opt, 
                loss = ['MSE','MSE'],
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
  datagen = Generator_motion.DataGenerator(x_list_trn,
                                           center_list_trn,
                                    patient_num = x_list_trn.shape[0], 
                                    batch_size = cg.batch_size, 
                                    num_classes = 3,
                                    input_dimension = (128,128,12),
                                    output_dimension = (6,), 
                                    shuffle = True, 
                                    remove_label= True,
                                    relabel_myo = False,
                                    slice_augment = True,
                                    seed = 10,
                                     )

  valgen = Generator_motion.DataGenerator(x_list_val,
                                          center_list_val,
                                    patient_num = x_list_val.shape[0], 
                                    batch_size = cg.batch_size, 
                                    num_classes = 3,
                                    input_dimension = (128,128,12),
                                    output_dimension = (6,), 
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
    
    trial_name = 'Motion_ResNet_new'
    data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_simulated_data_15normals.xlsx')
    
    load_model_file = None#os.path.join(cg.model_dir,'Motion_HR_14CP','models/batch_0','model-012.hdf5')
   
    epochs = 150

    train(val_batch,trial_name, data_sheet, epochs, load_model_file)

    

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int)
  args = parser.parse_args()
  
  if args.batch is not None:
    assert(0 <= args.batch < 5)

  main(args.batch)
