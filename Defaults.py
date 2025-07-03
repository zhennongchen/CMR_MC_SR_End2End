# System
import os

class Parameters():

  def __init__(self):
  
    # # Number of partitions in the crossvalidation.
    # self.num_partitions = int(os.environ['CG_NUM_PARTITIONS'])
    
    # Dimension of padded input, for training.
    self.input_dim = (int(os.environ['CG_INPUT_X']), int(os.environ['CG_INPUT_Y']), int(os.environ['CG_INPUT_Z']))
    self.output_dim = (int(os.environ['CG_OUTPUT_X']), int(os.environ['CG_OUTPUT_Y']), int(os.environ['CG_OUTPUT_Z']))
   
  
    # Seed for randomization.
    self.seed = int(os.environ['CG_SEED'])
      
    # UNet Depth
    self.resblock_num = int(os.environ['CG_RESBLOCK_NUM']) # default = 5
    
    # How many images should be processed in each batch?
    self.batch_size = int(os.environ['CG_BATCH_SIZE'])

    # classes:
    self.num_classes = int(os.environ['CG_NUM_CLASSES'])
    if int(os.environ['CG_RELABEL_RV']) == 1:
      self.relabel_RV = True
    else:
      self.relabel_RV = False

    if int(os.environ['CG_RELABEL_MYO']) == 1:
      self.relabel_myo = True
    else:
      self.relabel_myo = False

    # slice augmentation
    if int(os.environ['CG_SLICE_AUGMENT']) == 1:
      self.slice_augment = True
    else:
      self.slice_augment = False

    # Total number of epochs to train
    self.epochs = int(os.environ['CG_EPOCHS'])
    self.lr_epochs = int(os.environ['CG_LR_EPOCHS'])
    self.initial_power = float(os.environ['CG_INITIAL_POWER'])
    self.start_epoch = int(os.environ['CG_START_EPOCH'])
    self.decay_rate = float(os.environ['CG_DECAY_RATE'])
    self.regularizer_coeff = float(os.environ['CG_REGULARIZER_COEFF'])


    # # folders
    # for VR dataset
    self.nas_dir = os.environ['CG_NAS_DIR']
    self.data_dir = os.environ['CG_DATA_DIR']
    self.model_dir = os.environ['CG_MODEL_DIR']
    self.predict_dir = os.environ['CG_PREDICT_DIR']
    # self.picture_dir = os.environ['CG_PICTURE_DIR']