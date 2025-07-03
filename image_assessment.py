import os
import nibabel as nb
import numpy as np
from tqdm import tqdm
import pandas as pd
from latent_optimization.image_utils import *
import Image_utils as util
import functions_collection as ff
import Defaults as Defaults

cg = Defaults.Parameters()

# define case list
trial_name = 'latent_optimization_1'
patient_list = ff.sort_timeframe(ff.find_all_target_files(['*'],os.path.join(cg.predict_dir,trial_name)),0,'/')
case_list = []
for p in patient_list:
    cases = ff.find_all_target_files(['*/*'],p)
    for c in cases:
        case_list.append(c)
print(len(case_list))


result_list = []
for c in case_list:
    study_name = os.path.basename(c)
    tf = os.path.basename(os.path.dirname(c))
    patient_id = os.path.basename(os.path.dirname(os.path.dirname(c)))

    print(patient_id, tf, study_name)

    # load data
    predict_file = nb.load(os.path.join(c,'predict.nii.gz'))
    predict = predict_file.get_fdata()
    predict = np.rollaxis(predict,2,0)

    truth_file = nb.load(os.path.join(cg.data_dir,'processed_HR_data',patient_id,tf,'HR_'+tf+'_crop.nii.gz'))
    truth = truth_file.get_fdata()
    truth = np.rollaxis(truth,2,0)
    truth = np.round(truth); truth = truth.astype(int)

    spacing = truth_file.header.get_zooms()
    spacing = np.asarray([spacing[-1], spacing[0], spacing[1]])

    original_file = nb.load(os.path.join(cg.data_dir,'simulated_data',patient_id,tf,study_name,'data.nii.gz'))
    original = original_file.get_fdata()
    original = np.rollaxis(original,2,0)
    original = np.round(original); original = original.astype(int)

     # check whether label is correct
    rv_i = np.asarray(np.where(truth== 4))
    if rv_i.shape[1] == 0:
        correct_label = False
    else:
        correct_label = True
    print('correct label is: ',correct_label)

    # check whether orientation is correct
    correct_orientation = util.correct_ori(truth) # 0 is False, 1 is True
    print('orientation is: ',correct_orientation)
    

    if correct_orientation == True and correct_label == True:
        # DICE for three organs
        LV_dice, LV_d = np_categorical_dice_optim(predict, truth, k=1)
        MYO_dice, MYO_d = np_categorical_dice_optim(predict, truth, k=2)
        RV_dice, RV_d = np_categorical_dice_optim(predict, truth, k=4)
        print('LV DICE: {:0.2f}\nMYO DICE: {:0.2f}\nRV DICE: {:0.2f}'.format(LV_dice, MYO_dice, RV_dice))
        
        # contour-to-contour distance
        predict_ds = np.rollaxis(predict,0,3)
        predict_ds = np.rollaxis(util.downsample_in_z(predict_ds,5,affine=None),2,0)
        truth_ds = np.rollaxis(truth,0,3)
        truth_ds = np.rollaxis(util.downsample_in_z(truth_ds,5,affine=None),2,0)
        
        predict_distance,_,_ = util.c2c(predict_ds, [spacing[1],spacing[2]],threshold = 1, large = False)
        truth_distance,_,_ = util.c2c(truth_ds,[spacing[1],spacing[2]], threshold = 1, large = False)
        original_distance,_,_ = util.c2c(original,[spacing[1],spacing[2]], threshold = 1, large = False)

        result_list.append([patient_id,tf,study_name, True, True, LV_dice, MYO_dice, RV_dice,  truth_distance, original_distance, predict_distance])

    else:
        a = [patient_id,tf,study_name, correct_orientation, correct_label] + ['']*6
        result_list.append(a)

 

    

df = pd.DataFrame(result_list, columns = ['Patien_ID','timeframe','study_name','orientation correct?','label correct?',  'LV_dice','MYO_dice', 'RV_dice','contour2contour distance_truth','contour2contour distance_input','contour2contour distance_predict'])
df.to_excel(os.path.join(cg.predict_dir,'latent_optimization_1_quantitative','latent_optimization_1_quantitative.xlsx'), index= False)
   


    
