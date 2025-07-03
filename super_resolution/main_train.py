#!/usr/bin/env python

import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Build_list_data_prepare.Build_list as Build_list
import Generator
import EDSR 

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2

cg = Defaults.Parameters()

def train(val_batch,trial_name,data_sheet, epochs, load_model_file):
    # build lists
    print('Build List...')
    batch_list = [0,1,2,3,4]; batch_list.pop(val_batch)
    train_batch = batch_list
    b = Build_list.Build(data_sheet)
    _,_,_,_, _,y_list_trn, _,x_list_trn,_,_, _, _ = b.__build__(batch_list = train_batch)
    _,_,_,_, _,y_list_val, _,x_list_val,_,_, _, _ = b.__build__(batch_list = [val_batch])
    # _,_,_,_, _,y_list_trn, _,_,_,_, x_list_trn, _ = b.__build__(batch_list = train_batch)
    # _,_,_,_, _,y_list_val, _,_,_,_, x_list_val, _ = b.__build__(batch_list = [val_batch])
    n = np.arange(0,x_list_trn.shape[0],15)
    x_list_trn = x_list_trn[n]; y_list_trn = y_list_trn[n]
    n = np.arange(0,x_list_val.shape[0],15)
    x_list_val = x_list_val[n]; y_list_val = y_list_val[n]

    print(x_list_trn.shape, x_list_val.shape,x_list_trn[0:3],y_list_trn[0:3],x_list_val[0:3],y_list_val[0:3])

    # # create model
    print('Create Model...')
    input_shape = cg.input_dim + (cg.num_classes,)
    model_inputs = [Input(input_shape)]
    model_outputs=[]
    final_image = EDSR.main_model(cg.output_dim, cg.num_classes, 128, 5, layer_name = 'edsr')(model_inputs[0])
    model_outputs += [final_image]
    model = Model(inputs = model_inputs,outputs = model_outputs)

    if load_model_file != None:
        print('\n\n',load_model_file)
        model.load_weights(load_model_file)

    # compile model
    print('Compile Model...')
    opt = Adam(lr = 1e-4)
    losses = {'edsr': 'categorical_crossentropy'} 
    model.compile(optimizer= opt, 
                  loss= losses,
                  metrics= {'edsr':'acc',})

    print(model.summary())

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
                     LearningRateScheduler(EDSR.learning_rate_step_decay_classic),   # learning decay
                    ]

    # Fit
    datagen = Generator.DataGenerator(x_list_trn,y_list_trn,
                                      patient_num = x_list_trn.shape[0], 
                                      batch_size = cg.batch_size, 
                                      num_classes = 3,
                                      input_dimension = cg.input_dim,
                                      output_dimension = cg.output_dim, 
                                      shuffle = True, 
                                      remove_slices = 'None',
                                      remove_label= True,
                                      relabel_RV = False,
                                      relabel_myo = False,
                                      augment = True,
                                      seed = 10,
                                     )

    valgen = Generator.DataGenerator(x_list_val,y_list_val,
                                     patient_num = x_list_val.shape[0], 
                                     batch_size = cg.batch_size, 
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

    model.fit_generator(generator = datagen,
                        epochs = epochs,
                        validation_data = valgen,
                        callbacks = callbacks,
                        verbose = 1,
                        )


def main(val_batch):
    """These are the main training settings. Set each before running
    this file."""
    
    trial_name = 'EDSR_LVmyo_ds_new'
    data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_simulated_data_15normals.xlsx')
    
    load_model_file = None#os.path.join(cg.model_dir,'EDSR_LVmyo_ds_new','models/batch_0_initial','model-029.hdf5')
   
    epochs = 200

    train(val_batch,trial_name, data_sheet, epochs, load_model_file)

    

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int)
  args = parser.parse_args()
  
  if args.batch is not None:
    assert(0 <= args.batch < 5)

  main(args.batch)
