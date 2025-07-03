import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import time
import os
import math
import nibabel as nb
import scipy
from scipy import ndimage
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.motion_correction.Bspline as Bspline

cg = Defaults.Parameters()

# get patient list
patient_list = ff.sort_timeframe(ff.find_all_target_files(['1045'],os.path.join(cg.data_dir,'contour_dataset/simulated_data')),0)
case_list = []
for p in patient_list:
    cases = ff.find_all_target_files(['ED'],p)
    for c in cases:
        case_list.append(c)

# get center line points (for LV + myo)
for case in case_list:
    patient_id = os.path.basename(os.path.dirname(case))
    tf = os.path.basename(case)

    files = ff.find_all_target_files(['ds/data_clean.nii.gz', 'normal_motion*/data_clean.nii.gz'], case)

    for img_file in files:
        print(img_file)
        img = nb.load(img_file).get_fdata()
        img= np.round(img)
        img = img.astype(int)
        img = util.relabel(img,4,0)

        # find centerline points
        slice_list = []; center_list_raw = []
        for i in range(0,img.shape[-1]):
            I = img[:,:,i]
            # no heart:
            if np.where(I > 0)[0].shape[0] < 20 :
                center_list_raw.append(util.center_of_mass(np.zeros((20,20)),0,large = True))
                continue
            slice_list.append(i)
            center_list_raw.append(np.round(util.center_of_mass(I,0,large = True),2))

        np.save(os.path.join(os.path.dirname(img_file), 'centerlist.npy'), np.asarray(center_list_raw))
        ff.txt_writer(os.path.join(os.path.dirname(img_file),'centerlist.txt'), center_list_raw, ['']*len(center_list_raw))
        


