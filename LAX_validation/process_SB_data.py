# in this script, we will process the data from the SB study
# (1) correct the motion misalignment using LAX
# (2) flip the SAX images so that basal slice is on the first slice
# (3) add blank slices so that all SAX images have the same number of slices = 12
# (4) crop to 128x128

import numpy as np
import os
import nibabel as nb
import scipy.ndimage as ndimage
import pandas as pd

import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.LAX_validation.functions as LAX_func
import CMR_HFpEF_Analysis.Defaults as Defaults
cg = Defaults.Parameters()

# find all cases:
patient_list = ff.find_all_target_files(['*'],os.path.join(cg.data_dir, 'Sunny_Brooks/sunnybrooks_nii'))

# function 1: use LAX to correct the motion misalignment
# for i in range(32,33):#len(patient_list)):
#     patient_folder = patient_list[i]
#     patient_id = os.path.basename(patient_folder)
#     print('processing patient: ' + str(i) + ' '+ patient_id)
    
#     # load all LAX images
#     # find which LAXs we have (2CH, 3CH or 4CH)
#     names = ['LAX_2CH', 'LAX_3CH', 'LAX_4CH']
#     have = [0,0,0]
#     for i in range(0,3):
#         lax_files = ff.find_all_target_files([names[i] + '*'], patient_folder)
#         have[i] = [1 if len(lax_files) > 0 else 0][0]

#     save_folder = os.path.join(patient_folder, 'corrected')
#     ff.make_folder([save_folder])

#     # load sax images and segmentation
#     sax_img = nb.load(os.path.join(patient_folder ,'SAX_ED.nii.gz')).get_fdata()

#     sax_epi_img = nb.load(os.path.join(patient_folder,'SAX_ED_epi.nii.gz')).get_fdata()
#     sax_epi_seg = np.copy(sax_epi_img); sax_epi_seg[sax_epi_seg > 0] = 1

#     sax_endo_img = nb.load(os.path.join(patient_folder, 'SAX_ED_endo.nii.gz')).get_fdata()
#     sax_endo_seg = np.copy(sax_endo_img); sax_endo_seg[sax_endo_seg > 0] = 1

#     sax_affine = nb.load(os.path.join(patient_folder, 'SAX_ED_endo.nii.gz')).affine
   
#     # collect LAX pts and convert them into SAX coordinate
#     LAX_endo_pts_list = []
#     LAX_epi_pts_list = []
#     lax_affine_list = []
#     for i in range(0,3):
#         if have[i] == 0:
#             LAX_endo_pts_list.append([])
#             LAX_epi_pts_list.append([])
#             lax_affine_list.append([])
#         else:
#             # load LAX Epicaridum and Endocardium
#             lax_epi_seg = nb.load(os.path.join(patient_folder, names[i] + '_ED_epi.nii.gz')).get_fdata()
#             if len(lax_epi_seg.shape) == 4:
#                 lax_epi_seg = lax_epi_seg[:,:,0,:]
#             lax_epi_seg[lax_epi_seg > 0] = 1

#             lax_endo_seg = nb.load(os.path.join(patient_folder, names[i] + '_ED_endo.nii.gz')).get_fdata()
#             if len(lax_endo_seg.shape) == 4:
#                 lax_endo_seg = lax_endo_seg[:,:,0,:]
#             lax_endo_seg[lax_endo_seg > 0] = 1

#             # add affine matrix:
#             lax_affine_list.append([nb.load(os.path.join(patient_folder, names[i] + '_ED_epi.nii.gz')).affine])

#             # find LAX LV and myocardium points and convert them into SAX coordinate
#             # endocardium:
#             lax_pts = np.where(lax_endo_seg == 1)
#             grid_points = np.column_stack((lax_pts[0].flatten(), lax_pts[1].flatten(), lax_pts[2].flatten()))
#             converted_LAX_pts = ff.coordinate_convert(grid_points, sax_affine, lax_affine_list[i][0])
#             LAX_endo_pts_list.append(converted_LAX_pts)
#             # epicardium:
#             lax_pts = np.where(lax_epi_seg == 1)
#             grid_points = np.column_stack((lax_pts[0].flatten(), lax_pts[1].flatten(), lax_pts[2].flatten()))
#             converted_LAX_pts = ff.coordinate_convert(grid_points, sax_affine, lax_affine_list[i][0])
#             LAX_epi_pts_list.append(converted_LAX_pts)
#     # save these LAX pts
#     np.save(os.path.join(save_folder, 'converted_LAX_endo_pts_list.npy'), np.asarray(LAX_endo_pts_list))
#     np.save(os.path.join(save_folder, 'converted_LAX_epi_pts_list.npy'), np.asarray(LAX_epi_pts_list))

#     # motion correction:
#     sax_img_corrected, sax_endo_img_corrected, sax_epi_img_corrected, _, _ =  LAX_func.correct_motion_misalignment_based_on_intersection('epi', sax_img, sax_endo_img, sax_epi_img, sax_endo_seg, sax_epi_seg, LAX_endo_pts_list, LAX_epi_pts_list, have)
#     # save these images
#     nb.save(nb.Nifti1Image(sax_img_corrected, sax_affine),os.path.join(save_folder, 'SAX_ED_corrected.nii.gz') )
#     nb.save(nb.Nifti1Image(sax_endo_img_corrected, sax_affine),os.path.join(save_folder, 'SAX_ED_endo_corrected.nii.gz') )
#     nb.save(nb.Nifti1Image(sax_epi_img_corrected, sax_affine),os.path.join(save_folder, 'SAX_ED_epi_corrected.nii.gz') )

Results = []
# function 2: find out whether we need to flip the SAX images so that basal slice is on the first slice
# function 3: find out how to add blank slices so that all SAX images have the same number of slices = 12
for i in range(0,len(patient_list)):
    patient_folder = patient_list[i]
    patient_id = os.path.basename(patient_folder)
    print('looking at patient: ' + str(i) + ' '+ patient_id)

    # load SAX ED endo
    sax_endo_img = nb.load(os.path.join(patient_folder, 'SAX_ED_endo.nii.gz')).get_fdata()
    sax_endo_binary = np.copy(sax_endo_img); sax_endo_binary[sax_endo_binary > 0] = 1

    # find out whether the first slice has larger summation of pixel values than the last slice
    first_slice_sum = np.sum(sax_endo_binary[:,:,0])
    last_slice_sum = np.sum(sax_endo_binary[:,:,-1])
    if first_slice_sum < last_slice_sum:
        # print('start with the apical slice, need to flip')
        flip_required = 1
    else:
        flip_required = 0

    # find out how many slices we need to add, basically we want to add blank slices to the beginning and the end evenly to make total number of slices = 12
    # first let's find out how many slices in the original SAX images
    original_slice_num = sax_endo_img.shape[-1]
    # if want to add blank slices to the beginning and the end as evenly as possible, how many slices do we need in the beginning and the end?
    blank_slice_num = 12 - original_slice_num
    blank_slice_num_begin = int(np.floor(blank_slice_num / 2))
    blank_slice_num_end = blank_slice_num - blank_slice_num_begin
    start_index = blank_slice_num_begin
    end_index = start_index + original_slice_num

    # save these results with patient id
    Results.append([patient_id, flip_required, start_index, end_index])

    # do the processing and save the results
    # load data
    sax_endo_img = nb.load(os.path.join(patient_folder, 'SAX_ED_endo.nii.gz')).get_fdata(); sax_epi_img = nb.load(os.path.join(patient_folder, 'SAX_ED_epi.nii.gz')).get_fdata()
    sax_img = np.copy(sax_epi_img)
    sax_img[sax_img > 0] = 2
    sax_img[sax_endo_img > 0] = 1

    sax_endo_img_c = nb.load(os.path.join(patient_folder, 'corrected/SAX_ED_endo_corrected.nii.gz')).get_fdata(); sax_epi_img_c = nb.load(os.path.join(patient_folder, 'corrected/SAX_ED_epi_corrected.nii.gz')).get_fdata()
    sax_img_c = np.copy(sax_epi_img_c)
    sax_img_c[sax_img_c > 0] = 2
    sax_img_c[sax_endo_img_c > 0] = 1

    # do flipping
    if flip_required == 1:
        # flip the images
        sax_img = np.flip(sax_img, axis = 2)
        sax_img_c = np.flip(sax_img_c, axis = 2)

    # crop to 128x128
    sax_img_crop = util.crop_or_pad(sax_img, ([128,128,sax_img.shape[-1]]))
       
    sax_img_c_crop = util.crop_or_pad(sax_img_c, ([128,128,sax_img.shape[-1]]))
    
    # add blank slices
    sax_img_final = np.zeros((sax_img_crop.shape[0], sax_img_crop.shape[1], 12))
    sax_img_final[:,:,start_index:end_index] = sax_img_crop

    sax_img_c_final = np.zeros((sax_img_c_crop.shape[0], sax_img_c_crop.shape[1], 12))
    sax_img_c_final[:,:,start_index:end_index] = sax_img_c_crop

    # save the results
    nb.save(nb.Nifti1Image(sax_img_final, nb.load(os.path.join(patient_folder, 'SAX_ED_endo.nii.gz')).affine), os.path.join(patient_folder, 'SAX_ED_for_DL.nii.gz'))
    nb.save(nb.Nifti1Image(sax_img_c_final, nb.load(os.path.join(patient_folder, 'SAX_ED_endo.nii.gz')).affine), os.path.join(patient_folder, 'corrected', 'SAX_ED_corrected_for_DL.nii.gz'))
    
    # turn Results into a excel file using pandas
    # df = pd.DataFrame(Results, colta_dumns = ['patient_id', 'flip_required', 'start_index', 'end_index'])
    # df.to_excel(os.path.join(cg.dair, 'Sunny_Brooks/Sunny_Brooks_processing.xlsx'), index = False)

  

        


    


