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
iteration_num = 1
save_folder = os.path.join(cg.predict_dir,study_set, trial_name, 'round_'+str(iteration_num), 'images')
ff.make_folder([os.path.dirname(os.path.dirname(save_folder)),os.path.dirname(save_folder),save_folder])

###### define data sheet
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_motion_flip_clean_7_slice_10_normal.xlsx')

###### build patient list
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [0,1,2,3,4]
x_list_predict, y_list_predict, patient_id_list, tf_list, motion_name_list, batch_list,_,_ = b.__build__(batch_list = batches)
n = np.arange(0,patient_id_list.shape[0],1)
x_list_predict = x_list_predict[n]

##### apply motion
for i in range(0,x_list_predict.shape[0]):
    patient_id = str(patient_id_list[n[i]])
    timeframe = tf_list[n[i]]
    motion_name = motion_name_list[n[i]]
    batch = batch_list[n[i]]

    data_path = os.path.join(cg.data_dir,'simulated_data_version2', patient_id, timeframe,motion_name)
    pred_path = os.path.join(save_folder, patient_id, timeframe, motion_name)

    # load lv_slice_condition
    slice_condition = np.load(os.path.join(os.path.dirname(data_path), 'lv_slice_condition.npy'), allow_pickle = True)
    # the slices in the data_clean (slice w/ LV) --> where the motion parameters were predicted from
    slice_clean = np.asarray(slice_condition[2][0])
    # print('slice clean: ', slice_clean)

    # whether it has motion parameters?
    if os.path.isfile(os.path.join(pred_path, 'centers/pred_final.npy')) == 0:
        print('haven not had motion final; skip'); continue

    # whether it has been done?
    # if os.path.isfile(os.path.join(pred_path,'pred_img.nii.gz')) == 1:
    #     print('done; skip'); continue

    # load motion image
    motion = nb.load(os.path.join(data_path, 'data_flip.nii.gz')); affine = motion.affine; header = motion.header
    motion = motion.get_fdata()
    motion = np.round(motion); motion = motion.astype(int); motion = util.relabel(motion, 4, 0)
    # load motion centerline
    motion_centers = np.asarray(ff.remove_nan(np.load(os.path.join(data_path,'centerlist.npy'),allow_pickle = True)))
    # print(motion_centers)
    # heart slices
    heart_slices = np.asarray([ii for ii in range(0,motion.shape[-1]) if np.sum(motion[:,:,ii]) > 0])
    # print('heart_slices: ', heart_slices)
    
    # load predicted motion
    pred = np.load(os.path.join(pred_path, 'centers/pred_final.npy'), allow_pickle = True)
    pred_x = pred[0,:]; pred_y = pred[1,:]
    pred_x = Bspline.control_points(np.linspace(0,1,7), pred_x, len(slice_clean))
    pred_y = Bspline.control_points(np.linspace(0,1,7), pred_y, len(slice_clean))

    pred_img = np.copy(motion)

    # load ground truth (if there is):
    gt_file = os.path.join(os.path.dirname(data_path), 'ds/centerlist.npy')
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
        if s not in slice_clean and j < 3:
            continue
            # print('imcomplete basal, skip'); continue
        
        if s in slice_clean:
            index = np.where(slice_clean == s)[0][0]
            motion_c = [motion_centers[index,0], motion_centers[index,1]]
            # print(index, slice_clean[index], motion_c)

        # apex
        if s not in slice_clean and j > 4:
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
          

    # do slice flip
    pred_img= pred_img[:,:,[pred_img.shape[-1] - i for i in range(1,pred_img.shape[-1] + 1)]]

    # move prediction to the center
    if have_gt == 0:
        center_mass = util.center_of_mass(pred_img,0,large = True)
        center_mass = [int(center_mass[0]),int(center_mass[1]),int(center_mass[2])]
        center_image = [ pred_img.shape[i] // 2 for i in range(0,len(pred_img.shape))]

        move = [ center_image[i] - center_mass[i] for i in range(0,len(center_mass))]
        pred_img = util.move_3Dimage(pred_img, move)
        print('move the center')

    # save
    filename = os.path.join(pred_path,'pred_img.nii.gz')
    nb.save(nb.Nifti1Image(pred_img, affine, header), filename)

    # also save the flip-clean version
    pred_flip = pred_img[:,:,[pred_img.shape[-1] - i for i in range(1,pred_img.shape[-1] + 1)]]  # flip
    pred_flip_clean = np.zeros(pred_flip.shape)
    pred_flip_clean[:,:,slice_clean[0]:] = pred_flip[:,:,slice_clean[0]:] # clean
    nb.save(nb.Nifti1Image(pred_flip_clean, affine, header),  os.path.join(pred_path,'pred_img_flip_clean.nii.gz'))

            


        




  

    