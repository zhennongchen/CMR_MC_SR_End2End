import numpy as np
import nibabel as nb
import os
import pandas as pd
from scipy.ndimage import zoom
import CMR_HFpEF_Analysis.data_simulation.transformation as transform
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.sunny_brooks.functions_for_LAX as ff_LAX

cg = Defaults.Parameters()

# load preparation
lax_path = os.path.join(cg.data_dir, 'Sunny_Brooks/LAX')
# data_path = os.path.join(cg.predict_dir,'Sunny_Brooks/Iteration_C/round_1/images')
data_path = os.path.join(cg.predict_dir,'Sunny_Brooks/EDSR/images')

data_sheet =  pd.read_excel(os.path.join(lax_path, 'LAX_comparison_preparation_EDSR.xlsx'))
print(data_sheet.shape)

Results = []
for d in [27,28,29,30]:
    data = data_sheet.iloc[d]
    patient_id = data['case_id']
    timeframe = 'tf'+str(int(data['timeframe']))
    lax = data['LAX']
    print(patient_id, timeframe, lax)


    ########## load case:
    # if int(data['pred_file']) == 1:
    #     case_path = os.path.join(data_path, patient_id, timeframe, 'pred_img_HR_final.nii.gz')
    # else:
    #     case_path = os.path.join(data_path, patient_id, timeframe, 'pred_img_HR_final_2.nii.gz')
    case_path = os.path.join(data_path, patient_id, timeframe, 'pred.nii.gz') # EDSR
    print(case_path)

    case = nb.load(case_path)
    spacing = [1.3672, 1.3672, 1.6]
    img = case.get_fdata(); img = np.round(img); img = img.astype(int)
    # zoom to have the same dimension
    img = zoom(img, [1,1,spacing[-1]/1.3672], order = 0)

    ########### load LAX
    print('LAX paramters: ', int(data['LAX_crop_x1']))
    if data['save_processed'][0:2] == 'ye':
        print('load saved gt')
        gt = nb.load(os.path.join(lax_path, 'contours', patient_id, 'gt_processed.nii.gz')).get_fdata()
        gt = np.round(gt).astype(int)
    else:
        endo_file = os.path.join(lax_path, 'contours', patient_id, lax + '_endo.nii.gz')
        endo = nb.load(endo_file).get_fdata()
        epi_file = os.path.join(lax_path, 'contours', patient_id, lax + '_epi.nii.gz')
        epi = nb.load(epi_file).get_fdata()

        gt = np.copy(epi); gt[epi==1] = 2; gt[endo==1] = 1

        # find out which slice has segmentation
        S = [i for i in range(0,endo.shape[-1]) if np.sum(endo[:,:,i]) > 0][0]
        gt = gt[:,:, S]

        # move to center
        center_mass = util.center_of_mass(gt,0,large = True); center_mass = [int(center_mass[0]),int(center_mass[1])]
        center_image = [ gt.shape[i] // 2 for i in range(0,2)]
        move = [center_image[i] - center_mass[i] for i in range(0,len(center_mass))]
        gt = util.move_3Dimage(gt, move)

        # flip 
        if data['LAX_flip'][0] =='y':
            gt = np.flip(gt,1)

        # dimension
        gt = util.crop_or_pad(gt,[128,128])

        # y_crop
        gt[:,int(data['LAX_crop_base']):] = 0; gt[:,:int(data['LAX_crop_apex'])] = 0

        # x-zoom
        if data['LAX_x_zoom'] != 1:
            gt_copy = np.copy(gt)
            gt_copy = zoom(gt_copy, [data['LAX_x_zoom'],1] ,order = 0)
            gt_copy_slices = np.asarray([ii for ii in range(0,gt_copy.shape[0]) if np.sum(gt_copy[ii,:]) > 0])
            gt = np.zeros(gt.shape); gt[gt_copy_slices[0]: gt_copy_slices[-1]+1, :] = gt_copy[gt_copy_slices[0]: gt_copy_slices[-1]+1, :]
            gt = util.move_heart_center_to_image_center(gt)

        # x axis crop
        gt[:int(data['LAX_crop_x1']), :] = 0;  gt[int(data['LAX_crop_x2']):, :] = 0


    # ############ find Long-axis
    original_apex_slice = int(data['original_apex_slice'])
    original_base_slice = int(data['original_base_slice'])
    original_apex_mid, original_base_mid, L_list, x_list, y_list, base_point_list = ff_LAX.long_axis(img, original_apex_slice, original_base_slice, incre_unit = 3)
    if len(L_list) <= 5:
        original_apex_mid, original_base_mid, L_list, x_list, y_list, base_point_list = ff_LAX.long_axis(img, original_apex_slice, original_base_slice, incre_unit = 2)
    if len(L_list) >= 15:# and d > 18: # add this when doing HF-NI for three-stage
        original_apex_mid, original_base_mid, L_list, x_list, y_list, base_point_list = ff_LAX.long_axis(img, original_apex_slice, original_base_slice, incre_unit = 4)   

  
    ########## Main script
    MAX_MEAN_DICE = 0
    MIN_MEAN_HD = 1000

    result = [patient_id, timeframe, lax, original_apex_mid, original_base_mid]
    print('how many long axis choices: ', len(L_list))

    for ii in range(0,len(L_list)):
        print(ii)
        L = L_list[ii]
        x = x_list[ii]
        y = y_list[ii]

        img_new = ff_LAX.resample_img(img, original_apex_mid, L, x, y)

        # new apex slice
        new_apex_slice = int(data['new_apex_slice'])
        new_apex_mid = util.center_of_mass(img_new[:,:,new_apex_slice],0 ,large = True)
        new_apex_mid = np.round(np.asarray([new_apex_mid[0], new_apex_mid[1], new_apex_slice])).astype(int)
        
        # compare
        optim_mean_dice, optim_mean_HD, max_mean_dice, pred_mean_lv_dice, pred_mean_myo_dice, min_mean_HD, pred_mean_lv_HD, pred_mean_myo_HD = ff_LAX.compare(img_new, gt, 
        new_apex_mid, pred_pre_crop = int(data['pred_crop_base']), apex_x_list = np.arange(-8,4,1), apex_y_list = np.arange(-8,4,1), r_list = np.arange(-90,10,10), crop_base_list = np.arange(0,4,1))

        # compare with previous result
        # dice
        if max_mean_dice > MAX_MEAN_DICE:
            MAX_MEAN_DICE = max_mean_dice
            MAX_MEAN_DICE_LV = pred_mean_lv_dice
            MAX_MEAN_DICE_MYO = pred_mean_myo_dice
            OPTIM_MEAN_DICE = optim_mean_dice
            II_MEAN_DICE = ii
            print(ii, 'update DICE', MAX_MEAN_DICE_LV,MAX_MEAN_DICE_MYO )

        # HD
        if min_mean_HD < MIN_MEAN_HD:
            MIN_MEAN_HD = min_mean_HD
            MIN_MEAN_HD_LV = pred_mean_lv_HD
            MIN_MEAN_HD_MYO = pred_mean_myo_HD
            OPTIM_MEAN_HD = optim_mean_HD
            II_MEAN_HD = ii
            print(ii, 'update HD',MIN_MEAN_HD_LV, MIN_MEAN_HD_MYO )

        if MAX_MEAN_DICE_LV >= 0.975 and MAX_MEAN_DICE_MYO >= 0.820:
            break
        

    # collect for DICE
    result += [II_MEAN_DICE,base_point_list[II_MEAN_DICE], L_list[II_MEAN_DICE], OPTIM_MEAN_DICE[0], OPTIM_MEAN_DICE[1], 
                OPTIM_MEAN_DICE[2], OPTIM_MEAN_DICE[3], MAX_MEAN_DICE_LV, MAX_MEAN_DICE_MYO]
    result += [II_MEAN_HD, base_point_list[II_MEAN_HD], L_list[II_MEAN_HD], OPTIM_MEAN_HD[0], OPTIM_MEAN_HD[1], 
                OPTIM_MEAN_HD[2], OPTIM_MEAN_HD[3],  MIN_MEAN_HD_LV, MIN_MEAN_HD_MYO]

    Results.append(result)
        
    column_list = ['case_id', 'timeframe', 'LAX', 'original_apex_mid', 'original_base_mid', 'best_dice_ii','best_dice_base_point','best_dice_L',
    'best_dice_apex_x','best_dice_apex_y','best_dice_rotation','best_dice_crop_base','best_dice_lv','best_dice_myo',
    'best_HD_ii','best_HD_base_point','best_HD_L',
    'best_HD_apex_x','best_HD_apex_y','best_HD_rotation','best_HD_crop_base','best_HD_lv','best_HD_myo']

    df = pd.DataFrame(Results, columns= column_list)
    df.to_excel(os.path.join(cg.predict_dir,'Sunny_Brooks/EDSR/LAX_assessment27282930.xlsx'), index = False)


