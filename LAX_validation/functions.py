import numpy as np
import scipy.ndimage as ndimage

import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Image_utils as util

def find_LAX_on_SAX(converted_LAX, sax_slice, sax_img):
    lax_on_sax = np.round(converted_LAX[np.abs(converted_LAX[:, 2] - sax_slice) <= 0.30]).astype(np.int16)

    if lax_on_sax.shape[0] == 0:
        return np.nan, np.nan, np.nan, sax_img
    lax_on_sax = lax_on_sax[lax_on_sax[:,0] < sax_img.shape[0]]
    lax_on_sax = lax_on_sax[lax_on_sax[:,1] < sax_img.shape[1]]
    
    sax_new = np.copy(sax_img)
    sax_new[lax_on_sax[:,0], lax_on_sax[:,1], lax_on_sax[:,2]] = np.max(sax_img) + 1
    sax_new[lax_on_sax[0,:][0], lax_on_sax[0,:][1], lax_on_sax[0,:][2]] = np.max(sax_img) + 3
    sax_new[lax_on_sax[-1,:][0], lax_on_sax[-1,:][1], lax_on_sax[-1,:][2]] = np.max(sax_img) + 3

    # the intersection is a line 
    one_end = lax_on_sax[0,:]
    other_end = lax_on_sax[-1,:]

    return lax_on_sax, one_end, other_end, sax_new

def correct_motion_misalignment_based_on_intersection(use, sax_img, sax_endo_img, sax_epi_img, sax_endo_seg, sax_epi_seg, LAX_endo_pts_list, LAX_epi_pts_list, have):
    # use = 'endo' or 'epi' or 'both'
    sax_img_c = np.copy(sax_img)
    sax_endo_img_c = np.zeros_like(sax_endo_img)
    sax_epi_img_c = np.zeros_like(sax_epi_img)
    sax_endo_seg_c = np.zeros_like(sax_endo_seg)
    sax_epi_seg_c = np.zeros_like(sax_epi_seg)

    for sax_slice_num in range(0, sax_img.shape[-1]):
        # original sax slice
        sax_img_slice = np.copy(sax_img[:,:,sax_slice_num])
        sax_endo_img_slice = np.copy(sax_endo_img[:,:,sax_slice_num])
        sax_epi_img_slice = np.copy(sax_epi_img[:,:,sax_slice_num])

        # find intersection points
        lax_endo_boundaries = []
        for list_i in range(0, len(LAX_endo_pts_list)):
            if have[list_i] == 1:
                _, intesect_end_1, intersect_end_2, _ = find_LAX_on_SAX(LAX_endo_pts_list[list_i], sax_slice_num, sax_endo_seg)
                if np.isnan(intesect_end_1).any():
                    # no intersection points found
                    continue
                lax_endo_boundaries.append(intesect_end_1[0:2]); lax_endo_boundaries.append(intersect_end_2[0:2])
        lax_endo_boundaries = np.asarray(lax_endo_boundaries)

        lax_epi_boundaries = []
        for list_i in range(0, len(LAX_epi_pts_list)):
            if have[list_i] == 1:
                _, intesect_end_1, intersect_end_2, _ = find_LAX_on_SAX(LAX_epi_pts_list[list_i], sax_slice_num, sax_epi_seg)
                if np.isnan(intesect_end_1).any():
                    # no intersection points found
                    continue
                lax_epi_boundaries.append(intesect_end_1[0:2]); lax_epi_boundaries.append(intersect_end_2[0:2])
        lax_epi_boundaries = np.asarray(lax_epi_boundaries)
        
        
        # now let's do motion correction:
        sax_endo_seg_slice = np.copy(sax_endo_seg[:,:,sax_slice_num])
        sax_epi_seg_slice = np.copy(sax_epi_seg[:,:,sax_slice_num])
        smallest_dist = 100000
        count = 0

        while True:
            #find the boundary points of SAX
            sax_endo_boundaries = util.mask_to_contourpts(sax_endo_seg_slice, ['Endo'], [1])['Endo'][:,[1,0]]
            sax_epi_boundaries = util.mask_to_contourpts(sax_epi_seg_slice, ['Endo'], [1])['Endo'][:,[1,0]]
            
            motion_vectors = []
            dist_list = []
            # do motion correction based on endocardium
            if use == 'endo' or use == 'both':
                for k in range(0, lax_endo_boundaries.shape[0]):
                    distances = np.linalg.norm(sax_endo_boundaries - lax_endo_boundaries[k,:], axis=1)
                    closest_sax_point = sax_endo_boundaries[np.argmin(distances)]
                    if distances[np.argmin(distances)] > 15:
                        # print('this intersection point is too far away')
                        continue
                    dist_list.append(distances[np.argmin(distances)])
                    motion_vectors.append(lax_endo_boundaries[k,:] - closest_sax_point)

            # based on epicardium
            if use == 'epi' or use == 'both':
                for k in range(0, lax_epi_boundaries.shape[0]):
                    distances = np.linalg.norm(sax_epi_boundaries - lax_epi_boundaries[k,:], axis=1)
                    closest_sax_point = sax_epi_boundaries[np.argmin(distances)]
                    if distances[np.argmin(distances)] > 15:
                        # print('this intersection point is too far away')
                        continue
                    dist_list.append(distances[np.argmin(distances)])
                    motion_vectors.append(lax_epi_boundaries[k,:] - closest_sax_point)

            # use results from both endocardium and epicardium
            if len(dist_list) == 0:
                print('no intersection points found')
                break

            D = sum(dist_list) / len(dist_list)
            M = np.mean(np.asarray(motion_vectors),axis = 0)
            # print(sax_slice_num, count, D, M, [int(np.round(M)[0]), int(np.round(M)[1])])

            # stop until no improvement in the distance between SAX and LAX
            if D < smallest_dist:
                sax_endo_seg_slice = util.move_3Dimage(sax_endo_seg_slice, [int(np.round(M)[0]), int(np.round(M)[1])])
                sax_epi_seg_slice = util.move_3Dimage(sax_epi_seg_slice, [int(np.round(M)[0]), int(np.round(M)[1])])
                sax_img_slice = util.move_3Dimage(sax_img_slice, [int(np.round(M)[0]), int(np.round(M)[1])])
                sax_endo_img_slice = util.move_3Dimage(sax_endo_img_slice, [int(np.round(M)[0]), int(np.round(M)[1])])
                sax_epi_img_slice = util.move_3Dimage(sax_epi_img_slice, [int(np.round(M)[0]), int(np.round(M)[1])])
                smallest_dist = D
                count += 1
                    
            elif D >= smallest_dist or count > 20:
                break

        sax_img_c[:,:,sax_slice_num] = sax_img_slice
        sax_endo_img_c[:,:,sax_slice_num] = sax_endo_img_slice
        sax_epi_img_c[:,:, sax_slice_num] = sax_epi_img_slice
        sax_endo_seg_c[:,:,sax_slice_num] = sax_endo_seg_slice
        sax_epi_seg_c[:,:,sax_slice_num] = sax_epi_seg_slice

    return sax_img_c, sax_endo_img_c, sax_epi_img_c, sax_endo_seg_c, sax_epi_seg_c

def LAX_Dice_optim(lax, reslice, k_list , x_min, x_max, y_min, y_max, d = 5, threeDimage = False):
    max_dice = 0
    best_dx, best_dy = 0, 0
    for dx in range(-d, d):
        for dy in range(-d, d):
            if threeDimage == True:
                lax_1 = util.move_3Dimage(np.copy(lax), (dx, dy, 0))
            else:
                lax_1 = util.move_3Dimage(np.copy(lax), (dx, dy))

            lax_1_partial = np.zeros_like(lax_1)
            lax_1_partial [x_min: x_max, y_min:y_max] = lax_1[x_min: x_max, y_min:y_max]
            dice_1 = ff.np_mean_dice(reslice, lax_1_partial, k_list)
              
            if dice_1 > max_dice:
                best_lax = np.copy(lax_1_partial)
                best_dx, best_dy = dx, dy
                max_dice = dice_1
    ds = [best_dx, best_dy]
    return max_dice, ds, best_lax

def LAX_HD_optim(lax, reslice,  min, k_list, x_min, x_max, y_min, y_max, d = 5, pixel_size = 1.0, threeDimage = True):
    reslice_pixels = []
    for jj in range(0,len(k_list)):
        pp = np.asarray(ff.count_pixel(reslice,k_list[jj])[1])
        reslice_pixels.append(pp)
    
    
    min_mean_dis = 10000
    best_dx, best_dy = 0, 0
    best_lax = np.zeros(reslice.shape)
    for dx in range(-d, d):
        for dy in range(-d, d):
            if threeDimage == True:
                lax_1 = util.move_3Dimage(np.copy(lax), (dx, dy, 0))
            else:
                lax_1 = util.move_3Dimage(np.copy(lax), (dx, dy))

            mean_dis = []
            lax_1_partial = np.zeros_like(lax_1)
            lax_1_partial[x_min: x_max, y_min:y_max] = lax_1[x_min: x_max, y_min:y_max]

            for jj in range(0,len(k_list)):
                lax_pixels = np.asarray(ff.count_pixel(lax_1_partial, k_list[jj])[1]); 
                if lax_pixels.shape[0] ==0 : # no pixels in this class
                    continue
                mean_dis.append(ff.HD(reslice_pixels[jj],lax_pixels, pixel_size, min))

            if len(mean_dis) == len(k_list): # have pixels from both classes
                mean_dis = np.mean(np.asarray(mean_dis))
                
                if mean_dis < min_mean_dis:
                    best_lax = np.copy(lax_1_partial)
                    best_dx, best_dy = dx, dy
                    min_mean_dis = mean_dis

    ds = [best_dx, best_dy]
    return min_mean_dis, ds, best_lax