# for LV and LVmyo

import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import Build_list
import Generator

import argparse
import os
import numpy as np
import nibabel as nb

cg = Defaults.Parameters()

trial_name = 'EDSR_LV_LVmyo_normal_motion_dataversion2_b'
data_set = 'Super_resolution'#
save_folder = os.path.join(cg.predict_dir,data_set, trial_name,'images')
ff.make_folder([os.path.dirname(save_folder),save_folder])

# obtain the previous predictions (LVmyo and LV)
patient_list = ff.find_all_target_files(['*/*/*'],os.path.join(cg.predict_dir,data_set,'EDSR_LV_normal_motion_dataversion2_b/images'))

for p in patient_list:
    patient_id = os.path.basename(os.path.dirname(os.path.dirname(p)))
    timeframe = os.path.basename(os.path.dirname(p))
    motion_name = os.path.basename(p)

    # LV segmentation:
    lv = nb.load(os.path.join(p, 'batch_0/pred.nii.gz'))
    affine = lv.affine; header = lv.header
    lv = lv.get_fdata()

    # LV+myo segmentation
    lvmyo = nb.load(os.path.join(cg.predict_dir,data_set,'EDSR_LVmyo_normal_motion_dataversion2_b/images', patient_id, timeframe, motion_name, 'batch_0/pred.nii.gz'))
    lvmyo = lvmyo.get_fdata()
    lvmyo[lvmyo==1] = 2

    lvmyo[lv==1] = 1

    # final image
    final = np.copy(lvmyo)
    ff.make_folder([os.path.join(save_folder,patient_id), os.path.join(save_folder, patient_id, timeframe), os.path.join(save_folder, patient_id, timeframe, motion_name), os.path.join(save_folder, patient_id, timeframe, motion_name, 'batch_0')])
    nb.save(nb.Nifti1Image(final,affine, header = header), os.path.join(save_folder, patient_id, timeframe, motion_name, 'batch_0', 'pred.nii.gz'))
    


    
