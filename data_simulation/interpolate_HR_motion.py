import numpy as np
import os
import nibabel as nb
import math
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.Defaults as Defaults
from scipy import interpolate
import CMR_HFpEF_Analysis.data_simulation.transformation as transform

cg = Defaults.Parameters()

patient_list = ff.sort_timeframe(ff.find_all_target_files(['*'],os.path.join(cg.data_dir,'simulated_data_version2')),0)
case_list = []
for p in patient_list:
    cases = ff.find_all_target_files(['*'],p)
    for c in cases:
        case_list.append(c)

for case in case_list:
    patient_id = str(os.path.basename(os.path.dirname(case)))
    tf = os.path.basename(case)
    print('patient: ', patient_id, tf)

    # load a HR case:
    original_file = nb.load(os.path.join(cg.data_dir,'processed_HR_data', patient_id,tf, 'HR_'+tf+'_crop_60.nii.gz'))
    header = original_file.header
    spacing = original_file.header.get_zooms()
    affine = original_file.affine
    original_img = original_file.get_fdata()
    original_img= np.round(original_img).astype(int)
    original_img = util.relabel(original_img,4,0)


    for i in range(0,11):
        if i > 0:
            folder = os.path.join(case, 'normal_motion_' + str(i))
        else: 
            folder = os.path.join(case, 'ds')
        print(folder)

        save_f = os.path.join(folder,'HR'); ff.make_folder([save_f])

        if i > 0:
            motion_record = np.load(os.path.join(folder,'motion_record.npy'),allow_pickle = True)

            motion_x = [motion_record[i][3][0] for i in range(0, motion_record.shape[0])]
            motion_y = [motion_record[i][3][1] for i in range(0, motion_record.shape[0])]
            x = np.arange(0,60,5)
            y = np.asarray(motion_x)
            t, c, k = interpolate.splrep(x, y, s=0, k=1)
            spline_x = interpolate.BSpline(t, c, k, extrapolate=False)
            motion_x_interpolate = [spline_x(l) for l in np.arange(np.min(x), np.max(x)+1,1)]
            motion_x_interpolate += [motion_x_interpolate[-1]] * 4

            y = np.asarray(motion_y)
            t, c, k = interpolate.splrep(x, y, s=0, k=1)
            spline_y = interpolate.BSpline(t, c, k, extrapolate=False)
            motion_y_interpolate = [spline_y(l) for l in np.arange(np.min(x), np.max(x)+1,1)]
            motion_y_interpolate += [motion_y_interpolate[-1]] * 4

            img_t = np.copy(original_img)
            record = []
            for j in range(0, original_img.shape[-1]):
                I = original_img[:,:,j]
                t_x = np.round(motion_x_interpolate[j])
                t_y = np.round(motion_y_interpolate[j])

                center_mass = util.center_of_mass(I,0,large = True)
                if math.isnan(center_mass[0]) == 0:
                    center_mass = [int(center_mass[0]),int(center_mass[1])]
                    translation,rotation,scale,M = transform.generate_transform_matrix([t_x, t_y],0.0,[1,1],I.shape)
                    M = transform.transform_full_matrix_offset_heart(M, center_mass)
                    II = transform.apply_affine_transform(I, M, order = 0)
                    img_t[:,:,j] = II
                record.append([t_x, t_y])

            np.save(os.path.join(save_f, 'HR_motion_record.npy'),np.asarray(record))
            nb.save(nb.Nifti1Image(img_t, affine,header),os.path.join(save_f, 'HR_data.nii.gz'))
        else:
            img_t = np.copy(original_img)  # no motion


        # flip
        img_f= img_t[:,:,[img_t.shape[-1] - j for j in range(1,img_t.shape[-1] + 1)]]
        nb.save(nb.Nifti1Image(img_f, affine,header),os.path.join(save_f, 'HR_data_flip.nii.gz'))

        # clean
        slice_condition = np.load(os.path.join(os.path.dirname(folder), 'lv_slice_condition.npy'), allow_pickle = True)
        lv_slices = np.asarray(slice_condition[2][0])

        hr_slices = np.arange(((lv_slices[0]+1) * 5 - 1), 60, 1)
        # print(hr_slices)
        np.save(os.path.join(save_f, 'HR_slice_condition.npy'), np.asarray(hr_slices))
        img_f_c = np.zeros(img_f.shape)
        img_f_c[:,:, hr_slices[0] : hr_slices[-1] + 1] = img_f[:,:, hr_slices[0] : hr_slices[-1] + 1]    
        nb.save(nb.Nifti1Image(img_f_c, affine,header),os.path.join(save_f, 'HR_data_flip_clean.nii.gz'))

        # center list
        lv_slice_list = []; lv_center_list_raw = []
        for ss in range(0,img_f_c.shape[-1]):
            I = img_f_c[:,:,ss]
            ##### no LV:
            if np.where(I == 1)[0].shape[0] < 50 :#####
                lv_center_list_raw.append(util.center_of_mass(np.zeros((20,20)),0,large = True))
                continue
            lv_slice_list.append(ss)
            lv_center_list_raw.append(np.round(util.center_of_mass(I,0,large = True),2))
        # print(lv_slice_list,  len(lv_slice_list))
        lv_center_list = ff.remove_nan(lv_center_list_raw)
        # print(lv_center_list, len(lv_center_list))
        np.save(os.path.join(save_f, 'HR_centerlist_LV_slices.npy'), np.asarray(lv_center_list))
        np.save(os.path.join(save_f, 'HR_centerlist_LV_slice_num_ref.npy'), np.asarray(lv_slice_list))
        ff.txt_writer(os.path.join(save_f,'HR_centerlist_LV_slices.txt'), lv_center_list_raw, ['']*len(lv_center_list_raw))

        slice_list = []; center_list_raw = []
        for ss in range(0,img_f_c.shape[-1]):
            I = img_f_c[:,:,ss]
            ##### no heart:
            II = np.copy(I); II[I == 2] = 1
            if np.sum(II) < 200 :  #####
                center_list_raw.append(util.center_of_mass(np.zeros((20,20)),0,large = True))
                continue
            slice_list.append(ss)
            center_list_raw.append(np.round(util.center_of_mass(I,0,large = True),2))
        # print(slice_list,  len(slice_list))
        center_list = ff.remove_nan(center_list_raw)
        # print(center_list,len(center_list))

        np.save(os.path.join(save_f, 'HR_centerlist_heart_slices.npy'), np.asarray(center_list))
        np.save(os.path.join(save_f, 'HR_centerlist_heart_slice_num_ref.npy'), np.asarray(slice_list))
        ff.txt_writer(os.path.join(save_f,'HR_centerlist_heart_slices.txt'), center_list_raw, ['']*len(center_list_raw))
