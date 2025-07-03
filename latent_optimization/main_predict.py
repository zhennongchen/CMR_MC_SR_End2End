import os
import nibabel as nb
import numpy as np
from tqdm import tqdm
from image_utils import *
from Get_prediction import *
import torch
import networks
from DegradProcess import batch_degrade, MotionDegrad
from torch import optim
import torch.nn.functional as F
import torch
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.iterative.Build_list as Build_list

gpu_id = 0
device = torch.device("cuda:{:d}".format(gpu_id) if torch.cuda.is_available() else "cpu")

cg = Defaults.Parameters()

# define study name:
trial_name = 'latent_optimization'

# define case_list
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_motion_flip_clean_7_slice_10_normal.xlsx')

###### build patient list
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [5]
x_list_predict, y_list_predict, patient_id_list, tf_list, motion_name_list, batch_list,HRfile_list, LRfile_list = b.__build__(batch_list = batches)
n = ff.get_X_numbers_in_interval(patient_id_list.shape[0], 0, 1, 10)
x_list_predict = x_list_predict[n]
y_list_predict = y_list_predict[n]
LRfile_list = LRfile_list[n]
# print(x_list_predict)


# define model
path = os.path.join(cg.model_dir,'SRHeart/betaVAE')
z_dim, beta = 64, 1e-3
model = networks.GenVAE3D(z_dim=z_dim, img_size=128, depth=64)
model.to(device)
model_path = os.path.join(path, 'VAECE_zdim_{:d}_epoch_100_beta_{:.2E}_alpha.pt'.format(z_dim, beta))
model.load_state_dict(torch.load(model_path, map_location=device))

# make prediction
for i in range(0,x_list_predict.shape[0]):
    patient_id = str(patient_id_list[n[i]])
    timeframe = tf_list[n[i]]
    motion_name = motion_name_list[n[i]]
    batch = batch_list[n[i]]

    print(patient_id, timeframe, motion_name)

    # import HR ground truth (for affine):
    gt_file = os.path.join(cg.data_dir,'processed_HR_data',patient_id, timeframe,'HR_'+timeframe+'_crop_60.nii.gz')
    
    gt_file = nb.load(gt_file)
    gt_affine = gt_file.affine

    save_folder = os.path.join(cg.predict_dir,trial_name,'images', patient_id,timeframe,motion_name)
    print(save_folder)
     
    ff.make_folder([os.path.dirname(os.path.dirname(save_folder)), os.path.dirname(save_folder),save_folder])

    if os.path.isfile(os.path.join(save_folder, 'predict.nii.gz')) == 1:
        print('done')
        continue

    seg_file_path = os.path.join(cg.data_dir, 'simulated_data_version2', patient_id,timeframe, motion_name, 'data.nii.gz')

    seg_nib = nb.load(seg_file_path)
    seg_data = seg_nib.get_fdata()

    seg_data = np.rollaxis(seg_data,2,0)
    seg_data = np.concatenate((seg_data,np.zeros((1,128,128))))
    seg_data = np.round(seg_data)
    seg_data = seg_data.astype(int)

    seg_LR = torch.Tensor(label2onehot(seg_data)[np.newaxis,:]).to(device)

    # predict
    SR_data = get_prediction_latent_optimization(seg_LR, model, z_dim, epochs = 1000, device = device)

    final_result = util.pick_largest(SR_data,[1,2,4])
    final_result = util.crop_or_pad(final_result, [60, 128,128])


    final_nb = nb.Nifti1Image(np.rollaxis(final_result,0,3),gt_affine)
    nb.save(final_nb, os.path.join(save_folder, 'predict.nii.gz'))

    
 
        
        
    

        
