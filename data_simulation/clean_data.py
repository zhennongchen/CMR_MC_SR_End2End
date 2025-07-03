# remove incomplete basal slices

import numpy as np
import glob as gb
import nibabel as nb
import os

import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.data_simulation.transformation as transform
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util

cg = Defaults.Parameters()


# remove the incomplete basal slice in LR
# patient_list = ff.sort_timeframe(ff.find_all_target_files(['*'],os.path.join(cg.data_dir,'contour_dataset/simulated_data')),0)
# case_list = []
# for p in patient_list:
#     cases = ff.find_all_target_files(['*'],p)
#     for c in cases:
#         case_list.append(c)

# for case in case_list:

#     patient_id = os.path.basename(os.path.dirname(case))
#     tf = os.path.basename(case)
#     print('patient: ', patient_id, tf)
#     if os.path.isfile(os.path.join(case,'normal_motion_15','data_clean.nii.gz') ) == 1:
#         print('done')
#         continue

#     # find the incomplete slice:
#     folder = os.path.join(case,'ds')

#     img_file = os.path.join(folder, 'data.nii.gz')
#     img_file = nb.load(img_file)
#     affine_LR = img_file.affine
#     header = img_file.header
#     img = img_file.get_fdata()
#     img= np.round(img);img = img.astype(int);img = util.relabel(img,4,0)

#     heart_mass = np.asarray([int(np.sum(img[:,:,i]) )for i in range(0,img.shape[-1])]).reshape(-1)
#     non_zero_slices = np.where(heart_mass != 0)[0]

#     remove_slice= []
#     for j in range(0,2):
#         mass1 = heart_mass[non_zero_slices[j]]
#         mass2 = heart_mass[non_zero_slices[j + 1]]

#         if mass1 < (mass2 / 1.67) + 100:
#             remove_slice.append(non_zero_slices[j])

#     lv_slice = [ii for ii in range(0,img.shape[-1]) if np.where(img[:,:,ii] == 1)[0].shape[0] != 0]
#     remain_slice = np.copy(np.asarray(lv_slice))
#     for s in remove_slice:
#         remain_slice = np.delete(remain_slice, np.where(remain_slice == s)[0])
#     remain_slice = remain_slice.tolist()
#     print(heart_mass, remove_slice, lv_slice, remain_slice)

#     lv_slice_condition = np.asarray([[lv_slice], [remove_slice], [remain_slice]])
#     np.save(os.path.join(case,'lv_slice_condition'), lv_slice_condition)
 
#     # remove slices
#     folders = ff.find_all_target_files(['ds', 'normal*'],case)

#     for folder in folders:
#         img_file = os.path.join(folder, 'data.nii.gz')
#         img_file = nb.load(img_file)
#         affine_LR = img_file.affine
#         header = img_file.header
#         img = img_file.get_fdata()

#         img_clean = np.copy(img)
#         if len(remove_slice) > 0:
#             for jj in remove_slice:
#                 img_clean[:,:,jj] = 0
        
#         nb.save(nb.Nifti1Image(img_clean, affine = affine_LR, header = header), os.path.join(folder,'data_clean.nii.gz'))


# remove the corresponding basal slices in HR data
patient_list = ff.sort_timeframe(ff.find_all_target_files(['*'],os.path.join(cg.data_dir,'contour_dataset/simulated_data')),0)
case_list = []
for p in patient_list:
    cases = ff.find_all_target_files(['*'],p)
    for c in cases:
        case_list.append(c)

for case in case_list:

    patient_id = os.path.basename(os.path.dirname(case))
    tf = os.path.basename(case)
    print('patient: ', patient_id, tf)
    if os.path.isfile(os.path.join(cg.data_dir,'contour_dataset/processed_HR_data', patient_id, tf,'HR_ED_zoomed_crop_flip_clean.nii.gz')) == 1:
        print('done')
        continue

    # find the clean LR data
    clean_LR = nb.load(os.path.join(case, 'ds/data_clean.nii.gz')).get_fdata()
    clean_LR = np.round(clean_LR).astype(int)
    # find the HR data
    original_HR_file = nb.load(os.path.join(cg.data_dir,'contour_dataset/processed_HR_data', patient_id, tf,'HR_ED_zoomed_crop_flip.nii.gz' ))
    original_HR = np.round(original_HR_file.get_fdata()).astype(int)

    # find which slices are not empty in LR:
    not_empty_slices = [slice_num for slice_num in range(0, clean_LR.shape[-1]) if np.sum(clean_LR[:,:,slice_num]) > 0 ]
    print(not_empty_slices)
    # use this info to make clean HR
    clean_HR = np.zeros(original_HR.shape)
    clean_HR[:,:, not_empty_slices[0] * 5 : (not_empty_slices[-1] * 5 + 5)] = original_HR[:,:, not_empty_slices[0] * 5 : (not_empty_slices[-1] * 5 + 5)]

    nb.save(nb.Nifti1Image(clean_HR, original_HR_file.affine), os.path.join(cg.data_dir,'contour_dataset/processed_HR_data', patient_id, tf,'HR_ED_zoomed_crop_flip_clean.nii.gz'))
 