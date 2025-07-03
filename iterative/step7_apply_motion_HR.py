import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.data_simulation.transformation as transform
import CMR_HFpEF_Analysis.motion_correction.Bspline as Bspline
import CMR_HFpEF_Analysis.iterative.Build_list as Build_list
import os
import numpy as np
import pandas as pd
import nibabel as nb
cg = Defaults.Parameters()

###### define study and data
trial_name = 'Iteration_C'
study_set = 'Combined'
iteration_num = 2
save_folder = os.path.join(cg.predict_dir,study_set, trial_name, 'round_'+str(iteration_num), 'images')
ff.make_folder([os.path.dirname(os.path.dirname(save_folder)),os.path.dirname(save_folder),save_folder])

###### define data sheet
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_HR_LVslices_motion_flip_clean_7_slice_10_normal_IterationC.xlsx')

###### build patient list
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [5]
x_list_predict, y_list_predict, patient_id_list, tf_list, motion_name_list, batch_list,_,_ = b.__build__(batch_list = batches)
n = np.arange(0,patient_id_list.shape[0],1)
x_list_predict = x_list_predict[n]

##### apply motion
for i in range(0,x_list_predict.shape[0]):
    patient_id = str(patient_id_list[n[i]])
    timeframe = tf_list[n[i]]
    motion_name = motion_name_list[n[i]]
    batch = batch_list[n[i]]
    print(patient_id, timeframe, motion_name)

    data_path1 = os.path.join(cg.data_dir,'simulated_data_version2', patient_id, timeframe,motion_name)
    data_path2 = os.path.join(cg.predict_dir,study_set, trial_name, 'round_'+str(iteration_num - 1), 'images', patient_id, timeframe, motion_name)
    pred_path = os.path.join(save_folder, patient_id, timeframe, motion_name)

    # load HR_slice_condition 
    # the slices in the data_clean (slice w/ LV) --> where the motion parameters were predicted from
    slice_clean = np.load(os.path.join(os.path.dirname(data_path1), 'ds/HR/HR_centerlist_LV_slice_num_ref.npy'), allow_pickle = True)
    # print(slice_clean, slice_clean.shape)
    
    # whether it has motion parameters?
    if os.path.isfile(os.path.join(pred_path, 'centers/pred_final.npy')) == 0:
        print('haven not had motion final; skip'); continue

    # # whether it has been done?
    # # if os.path.isfile(os.path.join(pred_path,'pred_img.nii.gz')) == 1:
    # #     print('done; skip'); continue
    
    # do for clean version first
    # # load motion image
    motion = nb.load(os.path.join(data_path2, 'pred_img_HR_final_flip_clean.nii.gz')); affine = motion.affine; header = motion.header
    motion = motion.get_fdata()
    motion = np.round(motion); motion = motion.astype(int); motion = util.relabel(motion, 4, 0)
    # heart slices
    heart_slices = np.asarray([ii for ii in range(0,motion.shape[-1]) if np.sum(motion[:,:,ii]) > 0])

    # calculate motion centerline
    motion_centers = []
    for ss in range(0,motion.shape[-1]):
        I = motion[:,:,ss]
        ##### no heart:
        if np.sum(I) <= 0 :#####
            motion_centers.append(util.center_of_mass(np.zeros((20,20)),0,large = True))
            continue
        motion_centers.append(np.round(util.center_of_mass(I,0,large = True),2))
    motion_centers = np.asarray(motion_centers)
    motion_centers = np.asarray(ff.remove_nan(motion_centers))
    # print(motion_centers)
    
    # load predicted motion
    pred = np.load(os.path.join(pred_path, 'centers/pred_final.npy'), allow_pickle = True)
    pred_x = pred[0,:]; pred_y = pred[1,:]
    pred_x = Bspline.control_points(np.linspace(0,1,10), pred_x, slice_clean.shape[0])
    pred_y = Bspline.control_points(np.linspace(0,1,10), pred_y, slice_clean.shape[0])

    pred_img = np.copy(motion)

    # load ground truth (if there is):
    gt_file = os.path.join(os.path.dirname(data_path1), 'ds/HR/HR_centerlist_LV_slices.npy')
    if os.path.isfile(gt_file) == 1:
        have_gt = 1
        gt_centers = np.asarray(ff.remove_nan(np.load(gt_file,allow_pickle = True)))
        motion_base = gt_centers[0,:]
      
    else:
        have_gt = 0
        motion_base = motion_centers[0,:]  
    
    # do slice shift
    for j in range(0,heart_slices.shape[0]):
        s = heart_slices[j]
        # print('current_slice: ', s)

        # imcomplete basal:
        if s not in slice_clean and j < 5:
            continue
            # print('imcomplete basal, skip'); continue
        
        if s in slice_clean:
            index = np.where(slice_clean == s)[0][0]
            motion_c = [motion_centers[j,0], motion_centers[j,1]]
            # print(index, slice_clean[index], motion_c)

        # apex
        if s not in slice_clean and j > 30:
            # use the previous slice's motion
            index = -1
            motion_c = np.round(util.center_of_mass(motion[:,:,s],0,large = True),2)
            # print(index, slice_clean[index], motion_c)

        # calculate new center and shift
        new_center = [motion_base[0]+ pred_x[index] , motion_base[1] + pred_y[index]]
        # print('new_center: ', new_center)
        shift =  [motion_c[0] - new_center[0] ,  motion_c[1] - new_center[1]]
        # print(motion_c, shift)

        # do transformation
        if shift[0] == 0 and shift[1] == 0:
            continue
        else:
            I = pred_img[:,:,s]
            _,_,_,M = transform.generate_transform_matrix([shift[0], shift[1]],0.0, [1,1], I.shape)  
            img_new = transform.apply_affine_transform(I, M, order = 0)
            pred_img[:,:,s] = img_new
          
    # move prediction to the center
    if have_gt == 0:
        center_mass = util.center_of_mass(pred_img,0,large = True)
        center_mass = [int(center_mass[0]),int(center_mass[1]),int(center_mass[2])]
        center_image = [ pred_img.shape[i] // 2 for i in range(0,len(pred_img.shape))]

        move = [ center_image[i] - center_mass[i] for i in range(0,len(center_mass))]
        pred_img_move = util.move_3Dimage(pred_img, move)
        print('move the center')
    else:
        pred_img_move = np.copy(pred_img)

    # save
    filename = os.path.join(pred_path,'pred_img_HR_flip_clean.nii.gz')
    nb.save(nb.Nifti1Image(pred_img_move, affine, header), filename)



    # do for original version
    motion2 = nb.load(os.path.join(data_path2, 'pred_img_HR_final_flip.nii.gz'))
    motion2 = motion2.get_fdata()
    motion2 = np.round(motion2); motion2 = motion2.astype(int)
    # heart slices
    heart_slices2 = np.asarray([ii for ii in range(0,motion2.shape[-1]) if np.sum(motion2[:,:,ii]) > 0])
  
    pred_img_ori = np.copy(motion2)
    for j in range(0,heart_slices2.shape[0]):
        s = heart_slices2[j]
        # print('current_slice: ', s)

        if s>= slice_clean[0]:
            # print('in slice clean, directly copy')
            pred_img_ori[:,:,s] = pred_img[:,:,s]

    # do slice flip
    pred_img_ori = pred_img_ori[:,:,[pred_img_ori.shape[-1] - i for i in range(1,pred_img_ori.shape[-1] + 1)]]

    # move prediction to the center
    if have_gt == 0:
        center_mass = util.center_of_mass(pred_img_ori,0,large = True)
        center_mass = [int(center_mass[0]),int(center_mass[1]),int(center_mass[2])]
        center_image = [ pred_img_ori.shape[i] // 2 for i in range(0,len(pred_img_ori.shape))]

        move = [ center_image[i] - center_mass[i] for i in range(0,len(center_mass))]
        pred_img_ori_move = util.move_3Dimage(pred_img_ori, move)
        print('move the center')
    else:
        pred_img_ori_move = np.copy(pred_img_ori)

    # save
    filename = os.path.join(pred_path,'pred_img_HR.nii.gz')
    nb.save(nb.Nifti1Image(pred_img_ori_move, affine, header), filename)
