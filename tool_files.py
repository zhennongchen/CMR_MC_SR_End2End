import Defaults
import functions_collection as ff

import os
import numpy as np
import glob
import nibabel as nb
import shutil
import pandas as pd
import SimpleITK as sitk
import cv2
from skimage.measure import label, regionprops

cg = Defaults.Parameters()

######## check dimension
# file_list = ff.find_all_target_files(['SC-HYP-01/img-nii-raw/*.nii.gz'],os.path.join(cg.data_dir,'Sunny_Brooks/LAX/nii-images/'))

# for i in range(0,file_list.shape[0]):
#     print(file_list[i])
#     a = nb.load(file_list[i])
#     if len(a.shape) > 3:
#         b = a.get_fdata()[:,:,0,:]
#     else:
#         b = np.copy(a.get_fdata())
#     print(a.get_fdata().shape)
#     print(a.header.get_zooms())
#     save_folder = os.path.join(os.path.dirname(os.path.dirname(file_list[i])), 'img-nii')
#     ff.make_folder([save_folder])
#     nb.save(nb.Nifti1Image(b, affine = a.affine, header = a.header), os.path.join(save_folder,os.path.basename(file_list[i])))
#     b = nb.load(os.path.join(save_folder,os.path.basename(file_list[i]))); print(b.shape); print(b.header.get_zooms())


######## delete files
# file_list = ff.find_all_target_files(['*/corrected/SAX_ED_for_DL.nii.gz'],os.path.join(cg.data_dir,'Sunny_Brooks/sunnybrooks_nii/'))
# for f in file_list:
#     print(f)
#     os.remove(f)
    # shutil.rmtree(f)
        
######## change file name
# def find_all_target_files(target_file_name,main_folder):
#     F = np.array([])
#     for i in target_file_name:
#         f = np.array(sorted(glob.glob(os.path.join(main_folder, os.path.normpath(i)))))
#         F = np.concatenate((F,f))
#     return F
# def XX_to_ID_00XX(num):
#     if num < 10:
#         return 'ID_000' + str(num)
#     elif num>= 10 and num< 100:
#         return 'ID_00' + str(num)
#     elif num>= 100 and num < 1000:
#         return 'ID_0' + str(num)
#     elif num >= 1000:
#         return 'ID_' + str(num)
    
# # main_path = '/Volumes/TOSHIBA_4TB/MGH/HFpEF_zhennong/nnrd'
# # sheet = pd.read_excel(os.path.join(os.path.dirname(main_path),'Important_HFpEF_Patient_list_unique_patient_w_notes.xlsx'), sheet_name = 'Sheet1')
# nas_drive = '/Volumes/IRB2020P002624-DATA/zhennongchen/HFpEF/data/HFpEF_data/unchecked/contours'
# patients = find_all_target_files(['ID*'],nas_drive)

# for i in range(0,len(patients)):
#     patient_id = os.path.basename(patients[i])
#     print(patient_id)
    
#     folder = os.path.join(main_path,patient_id)
#     if os.path.isdir(folder) == 0:
#         print('oh no patient!!')
#     if os.path.isdir(folder) == 1:
#         os.rename(folder, os.path.join(main_path,'need_2_' + patient_id))
  

    
# folder_list = find_all_target_files(['*'],main_path)
# folder_names = ['SAX_ED_endo', 'SAX_ED_epi', 'SAX_ES_endo', 'SAX_ES_epi']
# for folder in folder_list:
#     for folder_name in folder_names:
#         specific_folder = find_all_target_files([folder_name + '*'],folder)
#         a = specific_folder[0].item()
#         f = os.path.dirname(a)
#         os.rename(a, os.path.join(f, folder_name))

######## copy files/folders
# folder_list = ff.find_all_target_files(['*/*/*'],os.path.join(cg.predict_dir,'Combined/Iteration_A/round_1/images'))
# save_f = os.path.join(cg.predict_dir,'Combined/Iteration_D/round_1/images')
# for f in folder_list:
#     random_name = os.path.basename(f)
#     patient_subid = os.path.basename(os.path.dirname(f))
#     patient_id = os.path.basename(os.path.dirname(os.path.dirname(f)))

#     ff.make_folder([os.path.join(save_f,patient_id),os.path.join(save_f,patient_id,patient_subid),os.path.join(save_f,patient_id,patient_subid,random_name), os.path.join(save_f,patient_id,patient_subid,random_name,'centers')])

#     file = os.path.join(f,'pred_img.nii.gz')
#     shutil.copy(file,os.path.join(save_f,patient_id,patient_subid,random_name,'pred_img.nii.gz'))

#     file = os.path.join(f,'centers/pred_final.npy')
#     shutil.copy(file,os.path.join(save_f,patient_id,patient_subid,random_name,'centers', 'pred_final.npy'))
    

######## zip folder
# shutil.make_archive(os.path.join(cg.data_dir,'HFpEF_data/Patient_list/individual_reports'), 'zip', os.path.join(cg.data_dir,'HFpEF_data/Patient_list/individual_reports'))
# shutil.make_archive(os.path.join(cg.nas_dir,'picture_collections/snapshot_motion_free'), 'zip', os.path.join(cg.nas_dir,'picture_collections/snapshot_motion_free'))
# shutil.make_archive(os.path.join(cg.nas_dir,'picture_collections/snapshot_simulated_new'), 'zip', os.path.join(cg.nas_dir,'picture_collections/snapshot_simulated'))

####### convert nrrd to nii with known nii affine and header
# spreadsheet = pd.read_excel(os.path.join(cg.data_dir, 'HFpEF_data/Patient_list', 'Important_HFpEF_Patient_list_unique_patient_w_readmission_finalized.xlsx' ))
# spreadsheet = spreadsheet.iloc[0:200]

# # patient_list = ff.find_all_target_files(['ID*'], os.path.join(cg.data_dir, 'HFpEF_Data/nii_manual_seg'))
# for p in range(0,spreadsheet.shape[0]):
#     patient_id = ff.XX_to_ID_00XX(spreadsheet['OurID'].iloc[p])
#     print(patient_id)

#     # load one nii file for reference affine and header
#     nii_file = os.path.join(cg.data_dir,'HFpEF_Data/nii_img_for_affines_reference', patient_id, 'Org3D_frame1.nii.gz')
#     # get the pixel dimension in header
#     seg = nb.load(nii_file)
#     header = seg.header
#     affine = seg.affine

#     # load nrrd files
#     img_folder = os.path.join(cg.data_dir,'HFpEF_Data/nrrd', 'need_' + patient_id)
#     img_files = ff.sort_timeframe(ff.find_all_target_files(['*nrrd'],img_folder), 1, start_signal='e')

#     if os.path.isfile(os.path.join(cg.data_dir,'HFpEF_Data/nii_img',patient_id, 'Org3D_frame25.nii.gz')) == 1:
#         print('done, continue');continue
                
#     for i in range(0,img_files.shape[0]):
#         img_file = img_files[i]
#         print(img_file)
#         frame = sitk.ReadImage(img_file)
#         spacing = frame.GetSpacing()
#         nrrd = sitk.GetArrayFromImage(frame)
#         nrrd = ff.nrrd_to_nii_orientation(nrrd, format = 'nrrd')
#         # save into nii file
#         nrrd_img = nb.Nifti1Image(nrrd, affine=affine, header=header)
#         ff.make_folder([os.path.join(cg.data_dir,'HFpEF_Data/nii_img',patient_id)])
#         nb.save(nrrd_img, os.path.join(cg.data_dir,'HFpEF_Data/nii_img',patient_id, 'Org3D_frame' + str(i + 1) + '.nii.gz'))



######## convert endo and epi segmentation into a single segmentation   
# patient_list = ff.find_all_target_files(['ID_*'], os.path.join(cg.data_dir, 'HFpEF_Data/nii_manual_seg'))
# for p in patient_list:
#     patient_id = os.path.basename(p)
#     print(patient_id)

#     if os.path.isfile(os.path.join(p, 'SAX_ES_seg.nii.gz')) == 1:
#         continue 
    
#     # load one nii file 
#     nii_ED_endo = nb.load(os.path.join(p, 'SAX_ED_endo.nii.gz'))
#     nii_ED_epi = nb.load(os.path.join(p, 'SAX_ED_epi.nii.gz'))
#     nii_ES_endo = nb.load(os.path.join(p, 'SAX_ES_endo.nii.gz'))
#     nii_ES_epi = nb.load(os.path.join(p, 'SAX_ES_epi.nii.gz'))

#     ed_seg = np.zeros(nii_ED_endo.get_fdata().shape)
#     ed_seg_epi = nii_ED_epi.get_fdata()>0
#     ed_seg_epi = ff.erode_and_dilate(ed_seg_epi, (2,2), erode = True, dilate = True)

#     ed_seg_endo = nii_ED_endo.get_fdata()>0
#     # ed_seg_endo = ff.erode_and_dilate(ed_seg_endo, (2,2), erode = True, dilate = True)
    
#     ed_seg[ed_seg_epi>0] = 2
#     ed_seg[ed_seg_endo>0] = 1

#     # turn the scattered epi (inside endo) into endo
#     refined_ed_seg = np.copy(ed_seg)
#     for i in range(0,ed_seg.shape[2]):
#         if np.sum(ed_seg[:,:,i]) == 0:
#             continue
#         else:
#             labeled_image = label(ed_seg[:,:,i] == 2)
#             regions = regionprops(labeled_image)
#             region_sizes = [region.area for region in regions]
#             largest_region_label = np.argmax(region_sizes) + 1
       
#             result_image = np.copy(ed_seg[:,:,i])
#             for j in range(0, len(regions) + 1):
#                 if j != largest_region_label and j != 0:
#                     region_mask = (labeled_image == j)
#                     result_image[region_mask] = 1
#             refined_ed_seg[:,:,i] = result_image
#     ed_seg = np.copy(refined_ed_seg)


#     es_seg = np.zeros(nii_ES_endo.get_fdata().shape)
#     es_seg_epi = nii_ES_epi.get_fdata()>0
#     es_seg_epi = ff.erode_and_dilate(es_seg_epi, (2,2), erode = True, dilate = True)
#     es_seg_endo = nii_ES_endo.get_fdata()>0
#     # es_seg_endo = ff.erode_and_dilate(es_seg_endo, (2,2), erode = True, dilate = True)
#     es_seg[es_seg_epi>0] = 2
#     es_seg[es_seg_endo>0] = 1

#     # turn the scattered epi (inside endo) into endo
#     refined_es_seg = np.copy(es_seg)
#     for i in range(0,es_seg.shape[2]):
#         if np.sum(es_seg[:,:,i]) == 0:
#             continue
#         else:
#             labeled_image = label(es_seg[:,:,i] == 2)
#             regions = regionprops(labeled_image)
#             region_sizes = [region.area for region in regions]
#             largest_region_label = np.argmax(region_sizes) + 1
       
#             result_image = np.copy(es_seg[:,:,i])
#             for j in range(0, len(regions) + 1):
#                 if j != largest_region_label and j != 0:
#                     region_mask = (labeled_image == j)
#                     result_image[region_mask] = 1
#             refined_es_seg[:,:,i] = result_image
#     es_seg = np.copy(refined_es_seg)

#     # if patient_id == 'ID_0468':
#     #     ed_seg = np.flip(np.flip(ed_seg, axis = 0),axis = 2)  # ID-0468
#     ed_seg = nb.Nifti1Image(ed_seg, nii_ED_endo.affine, nii_ED_endo.header)
#     nb.save(ed_seg, os.path.join(p, 'SAX_ED_seg.nii.gz'))

#     es_seg = nb.Nifti1Image(es_seg, nii_ED_endo.affine, nii_ED_endo.header)
#     nb.save(es_seg, os.path.join(p, 'SAX_ES_seg.nii.gz'))


#### convert for other timeframes
# patient_list = ff.find_all_target_files(['ID_0284'], os.path.join(cg.data_dir, 'HFpEF_Data/nii_manual_seg'))
# for p in patient_list:
#     patient_id = os.path.basename(p)
#     print(patient_id)

#     # find all the time frames
#     tf_files = ff.sort_timeframe(ff.find_all_target_files(['SAX_TF*_endo*'],p),0,'F','_')
    
#     for tf_file in tf_files:
#         tf = ff.find_timeframe(tf_file, 0 , 'F','_')

#         if tf != 25:
#             # load one nii file 
#             nii_endo = nb.load(tf_file)
#             nii_epi = nb.load(os.path.join(p, 'SAX_TF' + str(tf) + '_epi.nii.gz'))

#             seg = np.zeros(nii_endo.get_fdata().shape)
#             seg_epi = nii_epi.get_fdata()>0
#             seg_epi = ff.erode_and_dilate(seg_epi, (2,2), erode = True, dilate = True)

#             seg_endo = nii_endo.get_fdata()>0
#             # seg_endo = ff.erode_and_dilate(seg_endo, (2,2), erode = True, dilate = True)

#             seg[seg_epi>0] = 2
#             seg[seg_endo>0] = 1

#             # turn the scattered epi (inside endo) into endo
#             refined_seg = np.copy(seg)
#             for i in range(0,seg.shape[2]):
#                 if np.sum(seg[:,:,i]) == 0:
#                     continue
#                 else:
#                     labeled_image = label(seg[:,:,i] == 2)
#                     regions = regionprops(labeled_image)
#                     region_sizes = [region.area for region in regions]
#                     largest_region_label = np.argmax(region_sizes) + 1
                
#                     result_image = np.copy(seg[:,:,i])
#                     for j in range(0, len(regions) + 1):
#                         if j != largest_region_label and j != 0:
#                             region_mask = (labeled_image == j)
#                             result_image[region_mask] = 1
#                     refined_seg[:,:,i] = result_image
#             seg = np.copy(refined_seg)

#             seg = nb.Nifti1Image(seg, nii_endo.affine, nii_endo.header)
#             nb.save(seg, os.path.join(p, 'SAX_TF' + str(tf) + '_seg.nii.gz'))

#         if tf == 25:
#             nii_endo = nb.load(tf_file)
#             seg = np.zeros(nii_endo.get_fdata().shape)
#             seg_endo = nii_endo.get_fdata()>0
#             seg[seg_endo>0] = 1

#             seg = nb.Nifti1Image(seg, nii_endo.affine, nii_endo.header)
#             nb.save(seg, os.path.join(p, 'SAX_TF' + str(tf) + '_seg.nii.gz'))
   


    

    