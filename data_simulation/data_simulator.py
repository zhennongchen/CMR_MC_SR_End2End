#!/usr/bin/env python

# downsampling
# simulate motion

# %%
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import glob as gb
import nibabel as nb
import shutil
import os
import math

import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.data_simulation.transformation as transform
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import generate_moving_data as gen

cg = Defaults.Parameters()


# define moving range:
t_mu = 2.5 #unit mm
t_sigma = 0.75 # unit mm
t_bar = 5 # unit mm <= mu + 2sigma ~ 5mm (4 pixels)
extreme = False


# define patients
patient_list = ff.sort_timeframe(ff.find_all_target_files(['*'],os.path.join(cg.data_dir,'contour_dataset/processed_HR_data')),0)
case_list = []
for p in patient_list:
    cases = ff.find_all_target_files(['ED'],p)
    for c in cases:
        case_list.append(c)

main_save_folder = os.path.join(cg.data_dir,'contour_dataset/simulated_data')


for case in case_list:
    patient_id = os.path.basename(os.path.dirname(case))
    tf = os.path.basename(case)
    print('patient: ', patient_id, tf)

    # if os.path.isfile(os.path.join(case,'different_orientation.txt')) == 1:
    #     print('this patient needs to be excluded')
    #     continue

    save_folder = os.path.join(main_save_folder,patient_id,tf)
    ff.make_folder([os.path.dirname(save_folder),save_folder])

    if os.path.isfile(os.path.join(save_folder,'normal_motion_15','data.nii.gz')) == 1:
        print('done')
        continue

    img_file = ff.find_all_target_files(['HR_ED_zoomed_crop_flip.nii.gz'],case)
    assert len(img_file) == 1
    img_file = nb.load(img_file[0])
    affine_HR = img_file.affine
    spacing = img_file.header.get_zooms()
    img = img_file.get_fdata()
    print('spacing: ', spacing)
    

    # generate static data (only down-sampling)
    img_ds,new_affine = util.downsample_in_z(img,5,affine=affine_HR)
    assert img_ds.shape[-1] == 12
    # print(img_ds.shape)
    new_spacing = (spacing[0],spacing[1],spacing[2]*5)      
    new_header = img_file.header
    new_header['pixdim'] = [-1, new_spacing[0], new_spacing[1], new_spacing[-1],0,0,0,0]

    
    save_folder_static = os.path.join(save_folder,'ds'); ff.make_folder([save_folder_static])
    if os.path.isfile(os.path.join(save_folder_static,'data.nii.gz')) == 0:
        img_ds_nb = nb.Nifti1Image(img_ds, new_affine, header=new_header)
        nb.save(img_ds_nb, os.path.join(save_folder_static,'data.nii.gz'))

    # generate moving data 
    for i in range(0,15):
        if os.path.isfile(os.path.join(save_folder,'normal_motion_'+str(i+1),'data.nii.gz')) == 1:
            continue

        print('generate normal motion', i+1)
        save_folder_r = os.path.join(save_folder,'normal_motion_'+str(i+1)); ff.make_folder([save_folder_r])

        img_new,record = gen.generate_moving_data(img_ds, t_mu, t_sigma, t_bar, 0, spacing, order = 0, extreme = extreme)
        # save image
        nb.save(nb.Nifti1Image(img_new, new_affine, header=new_header), os.path.join(save_folder_r,'data.nii.gz'))
        # save record
        ff.txt_writer2(os.path.join(save_folder_r, 'motion_record.txt'),record)
        np.save(os.path.join(save_folder_r, 'motion_record.npy'), np.asarray(record))
        




        #####
        # # generate its 2x misalignment:
        # print('now make translation 2x')
        # save_folder_2x = os.path.join(save_folder,'normal_motion_'+str(i+1) +'_2x'); ff.make_folder([save_folder_2x])
        # img_new_2x, record_2x = gen.generate_moving_data_nx(img_ds, 2, 0, record, spacing, order = 0, extreme = extreme)
        # nb.save(nb.Nifti1Image(img_new_2x, new_affine, header=new_header), os.path.join(save_folder_2x,'data.nii.gz'))
        # ff.txt_writer2(os.path.join(save_folder_2x, 'motion_record.txt'),record_2x)
        # np.save(os.path.join(save_folder_2x, 'motion_record.npy'), np.asarray(record_2x))
      

        # #####
        # # generate its 3x misalignment:
        # print('now make 3x')
        # save_folder_3x = os.path.join(save_folder,'normal_motion_'+str(i+1) +'_3x'); ff.make_folder([save_folder_3x])
        # img_new_3x, record_3x = gen.generate_moving_data_nx(img_ds, 3,0 ,record, spacing, order = 0, extreme = extreme)
        # nb.save(nb.Nifti1Image(img_new_3x, new_affine, header=new_header), os.path.join(save_folder_3x,'data.nii.gz'))
        # ff.txt_writer2(os.path.join(save_folder_3x, 'motion_record.txt'),record_3x)
        # np.save(os.path.join(save_folder_3x, 'motion_record.npy'), np.asarray(record_3x))

        # #####
        # # generate its 4x misalignment:
        # print('now make 4x')
        # save_folder_4x = os.path.join(save_folder,'normal_motion_'+str(i+1) +'_4x'); ff.make_folder([save_folder_4x])
        # img_new_4x, record_4x = gen.generate_moving_data_nx(img_ds, 4,0 ,record, spacing, order = 0, extreme = extreme)
        # nb.save(nb.Nifti1Image(img_new_4x, new_affine, header=new_header), os.path.join(save_folder_4x,'data.nii.gz'))
        # ff.txt_writer2(os.path.join(save_folder_4x, 'motion_record.txt'),record_4x)
        # np.save(os.path.join(save_folder_4x, 'motion_record.npy'), np.asarray(record_4x))




    