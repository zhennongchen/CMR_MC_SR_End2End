# compare our predicted volume with LAX by re-slicing the predicted volume

import numpy as np
import os
import nibabel as nb
import scipy.ndimage as ndimage
import pandas as pd

import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.Build_list_data_prepare.Build_list as Build_list
import CMR_HFpEF_Analysis.LAX_validation.functions as LAX_func
import CMR_HFpEF_Analysis.Defaults as Defaults
cg = Defaults.Parameters()

data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_SunnyBrooks.xlsx')
data_folder = os.path.join(cg.predict_dir,'Sunny_Brooks' , 'solution_3')
process_info = pd.read_excel(os.path.join(cg.data_dir, 'Sunny_Brooks', 'Sunny_Brooks_processing.xlsx'))

# build list
b = Build_list.Build(data_sheet)
patient_id_list,patient_tf_list,motion_name_list,_, _,y_list, _,_,_,_, x_list, _ = b.__build__(batch_list = [5])
n = np.arange(0,patient_id_list.shape[0],1)
x_list = x_list[n]; y_list = y_list[n]

# main script
Results = []
for i in range(0, x_list.shape[0]):
    patient_id = patient_id_list[n[i]] 
    print(patient_id)  

    if 1==2:#os.path.isfile(os.path.join(data_folder, patient_id, 'LAX_dice.npy')) == 1:
        print('done --> load')
        lv_dice_list = np.load(os.path.join(data_folder, patient_id, 'LAX_dice.npy'))[0,:]
        myo_dice_list = np.load(os.path.join(data_folder, patient_id, 'LAX_dice.npy'))[1,:]
        lv_hd_list = np.load(os.path.join(data_folder, patient_id, 'LAX_HD.npy'))[0,:]
        myo_hd_list = np.load(os.path.join(data_folder, patient_id, 'LAX_HD.npy'))[1,:]
        lv_hd_one_direct_list = np.load(os.path.join(data_folder, patient_id, 'LAX_HD.npy'))[2,:]
        myo_hd_one_direct_list = np.load(os.path.join(data_folder, patient_id, 'LAX_HD.npy'))[3,:]
        
    else:
        sax_files = ff.find_all_target_files(['pred_img_HR*'],  os.path.join(data_folder, patient_id_list[n[i]]))
        lv_dice_list = []
        myo_dice_list = []
        lv_hd_list =[]
        myo_hd_list = []
        lv_hd_one_direct_list = []
        myo_hd_one_direct_list = []

        for sax_file in sax_files:
            sax_data = nb.load(sax_file).get_data()

            # find out its row in process_info according to the column "patient_id"
            row = process_info.loc[process_info['patient_id'] == patient_id]
        
            # process the predicted volume
            # first remove the added blank slices
            sax_data = sax_data[:,:, row['start_index'].values[0] * 5 : row['end_index'].values[0] * 5]
            # then flip
            if row['flip_required'].values[0] == 1:
                sax_data = np.flip(sax_data, axis = 2)
            # then pad to 256x256
            sax_data = util.crop_or_pad(sax_data, (256,256,sax_data.shape[2]))

            # new SAX affine:
            scale = [1,1, 1 / 5]
            S = np.diag([scale[0], scale[1], scale[2], 1])
            sax_affine_new = np.matmul(nb.load(os.path.join(cg.data_dir, 'Sunny_Brooks/sunnybrooks_nii', patient_id, 'SAX_ED_endo.nii.gz')).affine, S)

            # find which LAXs we have (2CH, 3CH or 4CH)
            names = ['LAX_2CH', 'LAX_3CH', 'LAX_4CH']
            have = [0,0,0]
            for ii in range(0,3):
                lax_files = ff.find_all_target_files([names[ii] + '*'], os.path.join(cg.data_dir, 'Sunny_Brooks/sunnybrooks_nii', patient_id))
                have[ii] = [1 if len(lax_files) > 0 else 0][0]

            # validate on LAX
            for ii in range(0,3):
                if have[ii] == 0:
                    lv_dice_list.append(np.nan)
                    myo_dice_list.append(np.nan)
                    lv_hd_list.append(np.nan)
                    myo_hd_list.append(np.nan)
                    lv_hd_one_direct_list.append(np.nan)
                    myo_hd_one_direct_list.append(np.nan)
                    
                else:
                    # load LAX Epicaridum and Endocardium
                    lax_epi_seg = nb.load(os.path.join(cg.data_dir, 'Sunny_Brooks/sunnybrooks_nii', patient_id, names[ii] + '_ED_epi.nii.gz')).get_fdata()
                    if len(lax_epi_seg.shape) == 4:
                        lax_epi_seg = lax_epi_seg[:,:,0,:]
                    lax_epi_seg[lax_epi_seg > 0] = 1

                    lax_endo_seg = nb.load(os.path.join(cg.data_dir, 'Sunny_Brooks/sunnybrooks_nii', patient_id, names[ii] + '_ED_endo.nii.gz')).get_fdata()
                    if len(lax_endo_seg.shape) == 4:
                        lax_endo_seg = lax_endo_seg[:,:,0,:]
                    lax_endo_seg[lax_endo_seg > 0] = 1

                    # lax contours:
                    lax_contour = np.copy(lax_epi_seg)
                    lax_contour[lax_contour> 0] = 2
                    lax_contour[lax_endo_seg > 0] = 1

                    # find out which slice in LAX has my segmentation:
                    lax_slice_num = np.where(np.sum(lax_contour, axis = (0,1)) > 0)[0][0]
                    X, Y, Z = np.meshgrid(np.arange(lax_contour.shape[0]), np.arange(lax_contour.shape[1]), np.arange(lax_slice_num, lax_slice_num+1))
                    grid_points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

                    converted_LAX_pts = ff.coordinate_convert(grid_points, sax_affine_new, nb.load(os.path.join(cg.data_dir, 'Sunny_Brooks/sunnybrooks_nii', patient_id, names[ii] + '_ED.nii.gz')).affine)

                    interpolation = ff.define_interpolation(sax_data, Fill_value=0, Method='nearest')
                    reslice = interpolation(converted_LAX_pts).reshape([lax_contour.shape[0], lax_contour.shape[1]]).T

                    # find the smallest and largest x and y coordinate in reslice that has pixel value > 0          
                    if np.where(reslice == 1)[0].shape[0] == 0 or np.where(reslice == 2)[0].shape[0] == 1:
                        print('no intersection points found')
                        lv_dice_list.append(np.nan)
                        myo_dice_list.append(np.nan)
                        lv_hd_list.append(np.nan)
                        myo_hd_list.append(np.nan)
                        lv_hd_one_direct_list.append(np.nan)
                        myo_hd_one_direct_list.append(np.nan)
                        continue
                    
                    x_min = np.min(np.where(reslice > 0)[0]); y_min = np.min(np.where(reslice > 0)[1])
                    x_max = np.max(np.where(reslice > 0)[0]); y_max = np.max(np.where(reslice > 0)[1])

                    # calculate dice
                    max_dice, ds, best_lax = LAX_func.LAX_Dice_optim(lax_contour[:,:,lax_slice_num], reslice, [1,2] , x_min, x_max, y_min, y_max, d = 20, threeDimage = False)
                    
                    lv_dice = ff.np_categorical_dice(reslice, best_lax, [1])
                    myo_dice = ff.np_categorical_dice(reslice, best_lax, [2])
                    lv_dice_list.append(lv_dice)
                    myo_dice_list.append(myo_dice)

                    # calculate HD
                    min_mean_dis, ds, best_lax = LAX_func.LAX_HD_optim(lax_contour[:,:,lax_slice_num], reslice, True, [1,2], x_min, x_max, y_min, y_max, d = 20, pixel_size = 1.0, threeDimage = False)

                    lax_lv_pixels = np.asarray(ff.count_pixel(best_lax,1)[1]); lax_myo_pixels = np.asarray(ff.count_pixel(best_lax,2)[1])
                    reslice_lv_pixels = np.asarray(ff.count_pixel(reslice,1)[1]); reslice_myo_pixels = np.asarray(ff.count_pixel(reslice,2)[1])

                    lv_hd = ff.HD(reslice_lv_pixels, lax_lv_pixels, 1.0, min = True); lv_hd_list.append(lv_hd)
                    myo_hd = ff.HD(reslice_myo_pixels, lax_myo_pixels,1.0, min = True); myo_hd_list.append(myo_hd)
                    lv_hd_one_direct = ff.HD(reslice_lv_pixels, lax_lv_pixels, 1.0, min = False); lv_hd_one_direct_list.append(lv_hd_one_direct)
                    myo_hd_one_direct = ff.HD( reslice_myo_pixels, lax_myo_pixels,1.0, min = False) ; myo_hd_one_direct_list.append(myo_hd_one_direct)

        # concatenate lv_dice_list and myo_dice_list
        lv_dice_list = np.asarray(lv_dice_list)
        myo_dice_list = np.asarray(myo_dice_list)
        ll = np.concatenate((np.reshape(lv_dice_list,(1,-1)), np.reshape(myo_dice_list,(1,-1))), axis = 0)
        np.save(os.path.join(data_folder, patient_id, 'LAX_dice.npy'), ll)

        # concatenate HD list
        lv_hd_list = np.asarray(lv_hd_list)
        myo_hd_list = np.asarray(myo_hd_list)
        lv_hd_one_direct_list = np.asarray(lv_hd_one_direct_list)
        myo_hd_one_direct_list = np.asarray(myo_hd_one_direct_list)
        lll = np.concatenate((np.reshape(lv_hd_list,(1,-1)), np.reshape(myo_hd_list,(1,-1)), np.reshape(lv_hd_one_direct_list,(1,-1)), np.reshape(myo_hd_one_direct_list,(1,-1))), axis = 0)
        np.save(os.path.join(data_folder, patient_id, 'LAX_HD.npy'), ll)


    lv_dice_list_process = lv_dice_list[~np.isnan(lv_dice_list)]
    myo_dice_list_process = myo_dice_list[~np.isnan(myo_dice_list)]
    lv_hd_list_process = lv_hd_list[~np.isnan(lv_hd_list)]
    myo_hd_list_process = myo_hd_list[~np.isnan(myo_hd_list)]
    lv_hd_one_direct_list_process = lv_hd_one_direct_list[~np.isnan(lv_hd_one_direct_list)]
    myo_hd_one_direct_list_process = myo_hd_one_direct_list[~np.isnan(myo_hd_one_direct_list)]

    print('LV Dice: ', list(lv_dice_list), np.max(lv_dice_list_process),np.mean(lv_dice_list_process))
    print('Myo Dice: ', list(myo_dice_list), np.max(myo_dice_list_process), np.mean(myo_dice_list_process))
    print('LV HD: ', list(lv_hd_list), np.min(lv_hd_list_process), np.mean(lv_hd_list_process))
    print('Myo HD: ', list(myo_hd_list), np.min(myo_hd_list_process), np.mean(myo_hd_list_process))
    print('LV HD one direct: ', list(lv_hd_one_direct_list), np.min(lv_hd_one_direct_list_process), np.mean(lv_hd_one_direct_list_process))
    print('Myo HD one direct: ', list(myo_hd_one_direct_list), np.min(myo_hd_one_direct_list_process), np.mean(myo_hd_one_direct_list_process))

    Results.append([patient_id, np.max(lv_dice_list_process),  np.mean(lv_dice_list_process), np.max(myo_dice_list_process), np.mean(myo_dice_list_process),
                    np.min(lv_hd_list_process), np.mean(lv_hd_list_process), np.min(myo_hd_list_process), np.mean(myo_hd_list_process),
                    np.min(lv_hd_one_direct_list_process), np.mean(lv_hd_one_direct_list_process), np.min(myo_hd_one_direct_list_process), np.mean(myo_hd_one_direct_list_process)])
    df = pd.DataFrame(Results, columns = ['patient_id', 'lv_dice_max', 'lv_dice_mean', 'myo_dice_max', 'myo_dice_mean', 'lv_hd_min', 'lv_hd_mean', 'myo_hd_min', 'myo_hd_mean', 'lv_hd_one_direct_min', 'lv_hd_one_direct_mean', 'myo_hd_one_direct_min', 'myo_hd_one_direct_mean'])
    df.to_excel(os.path.join(data_folder, 'LAX_results.xlsx'), index = False)


    
    

    