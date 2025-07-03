import numpy as np
from scipy.ndimage import zoom
import CMR_HFpEF_Analysis.data_simulation.transformation as transform
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Image_utils as util

# function 1: get the LV axis
def long_axis(img,original_apex_slice, original_base_slice, incre_unit = 3):
    # find mid point
    original_apex_mid = util.center_of_mass(img[:,:,original_apex_slice],0 ,large = True)
    original_apex_mid = np.round(np.asarray([original_apex_mid[0], original_apex_mid[1], original_apex_slice])).astype(int)
    original_base_mid = util.center_of_mass(img[:,:,original_base_slice],0, large = True)
    original_base_mid = np.round(np.asarray([original_base_mid[0], original_base_mid[1], original_base_slice])).astype(int)

    incre = [incre_unit if original_base_mid[i] > original_apex_mid[i] else -incre_unit for i in range(0,2)]

    L_list = []; x_list = []; y_list = []
    base_point_list = []
    for base_x in np.arange(original_apex_mid[0], original_base_mid[0]+incre[0]+1,incre[0]):
        for base_y in np.arange(original_apex_mid[1], original_base_mid[1]+incre[1]+1,incre[1]):
            base_point_list.append([base_x, base_y, original_base_mid[-1]])
            L = np.array([base_x - original_apex_mid[0], base_y - original_apex_mid[1], original_base_slice - original_apex_slice])
            L = ff.normalize(L)
            a = [0,-1,0]
            x = ff.normalize(np.cross(L, a))
            y = ff.normalize(np.cross(L, x))
            L_list.append(L); x_list.append(x); y_list.append(y)
            # print([base_x, base_y], L, ff.normalize(L))

    # first in L_list is the z-axis of the image
    # last in L_list is the long-axis
    return original_apex_mid, original_base_mid, L_list, x_list, y_list, base_point_list


# function 2: resample volume
def resample_img(img, original_apex_mid, L, x, y):
    centers  = []
    count_a = (original_apex_mid[-1] - 0) // L[-1]
    count_b = (img.shape[-1] - original_apex_mid[-1]) // L[-1]

    for i in range(int(-count_a),int(count_b)+1):
        c = original_apex_mid + L * i
        centers.append(c)

    # define interpolation
    interpolation = ff.define_interpolation(img)

    # reslice
    img_new = np.zeros((img.shape[0],img.shape[1], len(centers)))
    for i in range(0, len(centers)):
        img_new[:,:,i] = ff.reslice_mpr(np.zeros([img.shape[0], img.shape[1]]), centers[i],x,y,1,1,interpolation)
    
    return img_new


# function 3:
def compare(img_new, gt, new_apex_mid, pred_pre_crop = 0, apex_x_list = np.arange(-5,6,1), apex_y_list = np.arange(-5,6,1), r_list = np.arange(-90,95,5), crop_base_list = np.arange(1,5,1)):
    gt_slices = np.asarray([ii for ii in range(0,gt.shape[-1]) if np.sum(gt[:,ii]) > 0])
    max_mean_dice = 0
    min_mean_HD = 1000
    optim_mean_dice = [0,0,0,0]
    optim_mean_HD = [0,0,0,0]
    for apex_x in apex_x_list:
        for apex_y in apex_y_list:
            for r in r_list:
                apex_m = new_apex_mid - np.asarray([apex_x, apex_y, 0])
                translation,rotation,scale,M = transform.generate_transform_matrix([0,0,0],[0,0,r / 180 * np.pi],[1,1,1],img_new.shape)
                M = transform.transform_full_matrix_offset_heart(M, np.array([apex_m[0], apex_m[1], img_new.shape[-1]//2]))
                img_t = transform.apply_affine_transform(img_new, M, order = 0)

                pred = img_t[apex_m[0],:,:]
                pred = util.crop_or_pad(pred,[128,128])

                # move to center
                pred = util.move_heart_center_to_image_center(pred)

                # crop base
                heart_slice = np.asarray([ii for ii in range(0,pred.shape[1]) if np.sum(pred[:,ii]) > 0])
               

                for crop_base in crop_base_list:
                    pred[:,(heart_slice[-1] - crop_base - pred_pre_crop):] = 0

                    # make the slice number equal to gt
                    pred_slices = np.asarray([ii for ii in range(0,pred.shape[-1]) if np.sum(pred[:,ii]) > 0])
                    pred_copy = np.copy(pred)
                    pred_copy = zoom(pred_copy, [1, gt_slices.shape[0] / pred_slices.shape[0]], order = 0)
                    pred_copy_slices = np.asarray([ii for ii in range(0,pred_copy.shape[-1]) if np.sum(pred_copy[:,ii]) > 0])
                    pred = np.zeros(pred.shape)
                    pred[:, pred_copy_slices[0]: pred_copy_slices[-1]+1] = pred_copy[:, pred_copy_slices[0]: pred_copy_slices[-1]+1]
                    pred_slices = np.asarray([ii for ii in range(0,pred.shape[-1]) if np.sum(pred[:,ii]) > 0])

                    # make the base slice overlap
                    move = [0, gt_slices[-1] - pred_slices[-1]]
                    pred = util.move_3Dimage(pred, move)

                    ##### dice
                    # pred_lv_dice, ds, _ = util.Dice_optim(pred, gt, [1], d = 5, threeDimage=False)
                    # pred_myo_dice, ds, _ = util.Dice_optim(pred, gt, [2], d = 5, threeDimage=False)
                    
                    # if pred_lv_dice > max_lv_dice:
                    #     optim_lv_dice = [apex_x, apex_y, r,crop_base]
                    #     max_lv_dice = pred_lv_dice
                    #     # print('best lv DICE: ',apex_x, apex_y, r, crop_base, pred_lv_dice)

                    # if pred_myo_dice > max_myo_dice:
                    #     optim_myo_dice = [apex_x, apex_y, r,crop_base]
                    #     max_myo_dice = pred_myo_dice
                    #     # print('best myo DICE: ',apex_x, apex_y, r, crop_base, pred_myo_dice)

                    pred_mean_dice, ds, pred_for_dice = util.Dice_optim(pred, gt, [1,2], d = 5, threeDimage=False)
                    if pred_mean_dice > max_mean_dice:
                        max_mean_dice = pred_mean_dice
                        optim_mean_dice = [apex_x, apex_y, r, crop_base]
                        pred_mean_lv_dice = ff.np_categorical_dice(pred_for_dice, gt, 1)
                        pred_mean_myo_dice = ff.np_categorical_dice(pred_for_dice, gt, 2)
                        print('best mean DICE: ',apex_x, apex_y, r, crop_base, pred_mean_lv_dice, pred_mean_myo_dice, pred_mean_dice)

                    ##### HD
                    # pred_lv_HD, _, _= util.HD_optim(pred, gt,  min = False, k_list = [1], d = 5, pixel_size = 1.367, threeDimage = False)
                    # pred_myo_HD, _, _= util.HD_optim(pred, gt,  min = False, k_list = [2], d = 5, pixel_size = 1.367, threeDimage = False) 
                    
                    # if pred_lv_HD < min_lv_HD:
                    #     optim_lv_HD = [apex_x, apex_y, r,crop_base]
                    #     min_lv_HD = pred_lv_HD
                    #     # print('best lv HD: ',apex_x, apex_y, r, crop_base, pred_lv_HD)

                    # if pred_myo_HD < min_myo_HD:
                    #     optim_myo_HD = [apex_x, apex_y, r,crop_base]
                    #     min_myo_HD = pred_myo_HD
                    #     # print('best myo HD: ',apex_x, apex_y, r, crop_base, pred_myo_HD)

                    pred_mean_HD, ds, pred_for_HD = util.HD_optim(pred, gt, min = False, k_list = [1,2], d = 5, pixel_size = 1.367, threeDimage=False)
                    if pred_mean_HD < min_mean_HD:
                        optim_mean_HD = [apex_x, apex_y, r, crop_base]
                        min_mean_HD = pred_mean_HD
                        gt_lv_pixels = np.asarray(ff.count_pixel(gt,1)[1]); gt_myo_pixels = np.asarray(ff.count_pixel(gt,2)[1])
                        pred_lv_pixels = np.asarray(ff.count_pixel(pred_for_HD,1)[1]); pred_myo_pixels = np.asarray(ff.count_pixel(pred_for_HD,2)[1])

                        pred_mean_lv_HD = ff.HD( pred_lv_pixels, gt_lv_pixels, pixel_size = 1.367, min = False)
                        pred_mean_myo_HD = ff.HD( pred_myo_pixels, gt_myo_pixels,pixel_size = 1.367, min = False)
                        
                        print('best mean HD: ',apex_x, apex_y, r, crop_base, pred_mean_lv_HD, pred_mean_myo_HD, pred_mean_HD)
    return optim_mean_dice, optim_mean_HD, max_mean_dice, pred_mean_lv_dice, pred_mean_myo_dice, min_mean_HD, pred_mean_lv_HD, pred_mean_myo_HD
