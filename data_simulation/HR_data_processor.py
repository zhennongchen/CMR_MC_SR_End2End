#!/usr/bin/env python

# 1. move the LV center to image center
# 2. resample pixel size from [1.25, 1.25, 2] to [1,1,2]
# 3. crop the image to 128 x 128 x 60 dimension, HR_ED_zoomed_crop.nii.gz
# 4. flip along z-direction, HR_ED_zoomed_crop_flip.nii.gz
# %%
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import glob as gb
import nibabel as nb
import shutil
import os
import scipy.ndimage

import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.data_simulation.transformation as transform
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util

cg = Defaults.Parameters()

# define final dim:
dim = [128,128,60]

# define patients
patient_list = ff.sort_timeframe(ff.find_all_target_files(['*'],os.path.join(cg.data_dir,'contour_dataset/raw_data')),0)
case_list = []
for p in patient_list:
    cases = ff.find_all_target_files(['HR_*'],p)
    for c in cases:
        case_list.append(c)
# case_list = ff.find_all_target_files(['*/HR_*'],os.path.join(cg.data_dir,'raw_data'))
print(len(case_list))
main_save_folder = os.path.join(cg.data_dir,'contour_dataset/processed_HR_data')


for case in case_list:
    patient_id = os.path.basename(os.path.dirname(case))
    base_name = os.path.basename(case)
    num = [i for i,e in enumerate(base_name) if e== '_'][-1]
    tf = base_name[num+1 : num+3]
    print('patient: ', patient_id, tf)

    save_folder = os.path.join(main_save_folder,patient_id,tf)
    ff.make_folder([os.path.dirname(save_folder),save_folder])

    # if os.path.isfile(os.path.join(save_folder,'HR_'+tf+'_zoomed_crop_flip.nii.gz')) == 1:
    #     print('done')
    #     continue

    # copy the HR data
    # if os.path.isfile(os.path.join(save_folder,os.path.basename(case))) == 0:
    #     shutil.copyfile(case, os.path.join(save_folder,os.path.basename(case)))
    

    img_file = nb.load(case)
    spacing = img_file.header.get_zooms()
    img = img_file.get_fdata()
    print(img.shape,spacing)

    ########
    #  check labels
    ########

    img = np.round(img)
    img= img.astype(int)
    # labels
    img[img== 3] = 4
    correct_label, labels = util.correct_label(img)
    assert correct_label == True
    

    ########
    #  move the center
    ########
    center_mass = util.center_of_mass(img,1,large = False)
    center_mass = [int(center_mass[0]),int(center_mass[1]),int(center_mass[2])]
    center_image = [ img.shape[i] // 2 for i in range(0,len(img.shape))]

    move = [ center_image[i] - center_mass[i] for i in range(0,len(center_mass))]
    img_move = util.move_3Dimage(img, move)

    new_center_mass = util.center_of_mass(img_move,1,large = False)
    assert (int(new_center_mass[0]) >= (center_image[0] - 2)) and (int(new_center_mass[0]) <= (center_image[0] + 2))
    assert (int(new_center_mass[1]) >= (center_image[1] - 2)) and (int(new_center_mass[1]) <= (center_image[1] + 2))
    assert (int(new_center_mass[2]) >= (center_image[2] - 2)) and (int(new_center_mass[2]) <= (center_image[2] + 2))
    
    # img_move_nb = nb.Nifti1Image(img_move, affine_HR, header=header_HR)
    # nb.save(img_move_nb, os.path.join(save_folder,'HR_'+tf+'_move.nii.gz'))

    # save move vector
    np.save(os.path.join(save_folder,'move_heart_center'),np.asarray(move))
    ff.txt_writer(os.path.join(save_folder,'move_heart_center.txt'),[[move[0]],[move[1]],[move[2]]],['x_move','y_move','z_move'])


    ########
    #  resample pixel dimension to [1,1,10]mm, then convert dimension to 128 128 60
    ########

    img_move_zoom = scipy.ndimage.zoom(img_move, [1.25, 1.25, 1], order = 0)
    print(img_move.shape, img_move_zoom.shape)

    apex_buffer = 4
    z_slice_range = [int(img_move_zoom.shape[-1]//2) - (int(dim[-1]/2) + apex_buffer) , int(img_move_zoom.shape[-1]//2) + (int(dim[-1]/2) - apex_buffer)]

    img_move_zoom = img_move_zoom[ :, :, z_slice_range[0]:z_slice_range[1]]
    
    img_move_crop = util.crop_or_pad(img_move_zoom,dim)
    print(z_slice_range[1] - z_slice_range[0], img_move_crop.shape)
    # save
    new_header = img_file.header
    new_header['pixdim'] = [-1, 1, 1,2 ,0,0,0,0]
    new_affine =  img_file.affine
    Transform = np.eye(4); Transform[0:3,0] = np.array([1/1.25,0,0]);  Transform[0:3,1] = np.array([0,1/1.25,0])
    new_affine = np.dot(new_affine,Transform)

    img_move_crop_nb = nb.Nifti1Image(img_move_crop, new_affine, header = new_header)
    nb.save(img_move_crop_nb, os.path.join(save_folder,'HR_'+tf+'_zoomed_crop.nii.gz'))


    ########
    #  check orientation, if wrong, then exclude this case
    ########
   
    a = util.correct_ori(img_move_crop)
    if a == False:
        print('this one has wrong orientation, need to exclude')
        t_file = open(os.path.join(save_folder,'different_orientation.txt'),"w+")
        t_file.write('wrong orientation')
        t_file.close()

    print('\n')


    ########
    # filp 
    ########
    img_flip = img_move_crop[:,:,[img_move_crop.shape[-1] - i for i in range(1,img_move_crop.shape[-1] + 1)]]
    img_flip_nb = nb.Nifti1Image(img_flip, new_affine, header = new_header)
    nb.save(img_flip_nb, os.path.join(save_folder,'HR_'+tf+'_zoomed_crop_flip.nii.gz'))

        

    


    
# %%
