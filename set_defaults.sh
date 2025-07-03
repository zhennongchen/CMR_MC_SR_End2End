## to run this in terminal, type:
# chmod +x set_defaults.sh
# . ./set_defaults.sh   

## parameters
# define GPU you use
export CUDA_VISIBLE_DEVICES="0"

# volume dimension
export CG_INPUT_X=128 
export CG_INPUT_Y=128 
export CG_INPUT_Z=12

export CG_OUTPUT_X=128 
export CG_OUTPUT_Y=128 
export CG_OUTPUT_Z=60

export CG_BATCH_SIZE=1 

# set number of resblock
export CG_RESBLOCK_NUM=5

# set number of classes
export CG_NUM_CLASSES=3
export CG_RELABEL_RV=0 # from class 4 to class 3
export CG_RELABEL_MYO=0 # from class 2 to class 1

# set learning epochs
export CG_EPOCHS=200
export CG_LR_EPOCHS=25 # the number of epochs for learning rate change 
export CG_START_EPOCH=0
export CG_DECAY_RATE=0.01
export CG_INITIAL_POWER=-4
export CG_REGULARIZER_COEFF=0.2

# set random seed
export CG_SEED=8

# set augmentation
export CG_SLICE_AUGMENT=0


# folders for Zhennong's dataset (change based on your folder paths)
export CG_NAS_DIR="/mnt/camca_NAS/HFpEF/"
export CG_DATA_DIR="${CG_NAS_DIR}data/"
export CG_MODEL_DIR="${CG_NAS_DIR}model/"
export CG_PREDICT_DIR="${CG_NAS_DIR}predict/"
# export CG_PICTURE_DIR="${CG_NAS_DIR}picture_collections/"
