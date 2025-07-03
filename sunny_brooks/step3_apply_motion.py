import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.data_simulation.transformation as transform
import CMR_HFpEF_Analysis.motion_correction.Bspline as Bspline
import CMR_HFpEF_Analysis.sunny_brooks.Build_list as Build_list
import os
import numpy as np
import pandas as pd
import nibabel as nb
cg = Defaults.Parameters()

###### define study and data
trial_name = 'Iteration_C'
study_set = 'Sunny_Brooks'
iteration_num = 1
save_folder = os.path.join(cg.predict_dir,study_set, trial_name, 'round_'+str(iteration_num), 'images')
ff.make_folder([os.path.dirname(os.path.dirname(save_folder)),os.path.dirname(save_folder),save_folder])

###### define data sheet
data_sheet = os.path.join(cg.data_dir,'Patient_list/Sunny_Brooks.xlsx')

###### build patient list
print('Build List...')
b = Build_list.Build(data_sheet)
input_list, patient_id_list, patient_tf_list = b.__build__()
n = np.arange(0,patient_id_list.shape[0],1)
input_list = input_list[n]

##### apply motion
for i in range(0,patient_id_list.shape[0]):
    patient_id = patient_id_list[n[i]]
    timeframe = patient_tf_list[n[i]]
   
    pred_path = os.path.join(save_folder, patient_id, timeframe)

    # load lv_slice_condition
    # lv_slice = [ii for ii in range(0,img.shape[-1]) if np.where(img[:,:,ii] == 1)[0].shape[0] != 0]
  

    # whether it has motion parameters?
    if os.path.isfile(os.path.join(pred_path, 'centers/pred_final.npy')) == 0:
        print('haven not had motion final; skip'); continue

    # whether it has been done?
    # if os.path.isfile(os.path.join(pred_path,'pred_img.nii.gz')) == 1:
    #     print('done; skip'); continue

    # load motion image
    motion = nb.load(input_list[i]); affine = motion.affine; header = motion.header
    motion = motion.get_fdata()
    motion = np.round(motion); motion = motion.astype(int); motion = util.relabel(motion, 4, 0)
    # original motion centerline
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
    print(motion_centers.shape)

    
    # heart slices
    heart_slices = np.asarray([ii for ii in range(0,motion.shape[-1]) if np.sum(motion[:,:,ii]) > 0])
    lv_slices = [ii for ii in range(0,motion.shape[-1]) if np.where(motion[:,:,ii] == 1)[0].shape[0] > 20]
    print('heart_slices: ', len(heart_slices))
    print('lv_slices: ',len(lv_slices))
    
    # load predicted motion
    pred = np.load(os.path.join(pred_path, 'centers/pred_final.npy'), allow_pickle = True)
    pred_x = pred[0,:]; pred_y = pred[1,:]
    pred_x = Bspline.control_points(np.linspace(0,1,7), pred_x, len(lv_slices))
    pred_y = Bspline.control_points(np.linspace(0,1,7), pred_y, len(lv_slices))
   
    # do slice shift
    pred_img = np.copy(motion)
    motion_base = motion_centers[0,:]


    for j in range(0,heart_slices.shape[0]):
        s = heart_slices[j]
        # print('current_slice: ', s)

        # imcomplete basal:
        if s not in lv_slices and j < 3:
            continue
            # print('imcomplete basal, skip'); continue
        
        if s in lv_slices:
            index = np.where(lv_slices == s)[0][0]
            motion_c = [motion_centers[j,0], motion_centers[j,1]]
            # print(index, lv_slices[index], motion_c)

        # apex
        if s not in lv_slices and j > 4:
            # use the previous slice's motion
            index = -1
            motion_c = np.round(util.center_of_mass(motion[:,:,s],0,large = True),2)
            # print(index,lv_slices[index], motion_c)

        # calculate new center and shift
        new_center = [motion_base[0]+ pred_x[index] , motion_base[1] + pred_y[index]]
        print('new center supposed: ', new_center)
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
            print('real new center: ',np.round(util.center_of_mass(img_new,0,large = True),2))
            pred_img[:,:,s] = img_new
          

    # do slice flip
    pred_img= pred_img[:,:,[pred_img.shape[-1] - i for i in range(1,pred_img.shape[-1] + 1)]]

    # # move prediction to the center
    # if have_gt == 0:
    #     center_mass = util.center_of_mass(pred_img,0,large = True)
    #     center_mass = [int(center_mass[0]),int(center_mass[1]),int(center_mass[2])]
    #     center_image = [ pred_img.shape[i] // 2 for i in range(0,len(pred_img.shape))]

    #     move = [ center_image[i] - center_mass[i] for i in range(0,len(center_mass))]
    #     pred_img = util.move_3Dimage(pred_img, move)
    #     print('move the center')

    # save
    filename = os.path.join(pred_path,'pred_img.nii.gz')
    nb.save(nb.Nifti1Image(pred_img, affine, header), filename)

   

            


        




  

    