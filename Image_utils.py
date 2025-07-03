import matplotlib.pyplot as plt
import numpy as np
import math
import nibabel as nb
from skimage.measure import label   
from scipy import ndimage
from scipy import interpolate
from scipy.spatial import ConvexHull
import cv2
from skimage.measure import label, regionprops
import CMR_MC_SR_End2End.data_simulation.transformation as transform
import CMR_MC_SR_End2End.functions_collection as ff
 
# check orientation
# when image is in [x,y,z], the x-plane must have RV at left and LV at right
def correct_ori(data):
 
    for j in range(data.shape[0]//2 - 14,data.shape[0]):
        s = data[j,:,:]
        # make sure it has RV and LV
        rv_i = np.asarray(np.where(s == 4))
        lv_i = np.asarray(np.where(s == 1))

        if (rv_i.shape[1]) == 0 or (lv_i.shape[1]) == 0:
            continue
        else:
            # compare their y axis 
            lv_i_x = np.mean(lv_i[0,:])
            rv_i_x = np.mean(rv_i[0,:])
    

            if rv_i_x >= lv_i_x:
                # print('wrong orientation')
                return False
            else:
                # print('correct orientation')
                return True

# check labels
# background = 0, LV = 1, myo = 2, RV = 4
def correct_label(data):
    labels = np.unique(data)
  
    if (4 not in labels) or labels.shape[0] != 4:
        return False, labels
    else:
        return True, labels


# center_of_mass
def center_of_mass(I,threshold,large = True): 
    II = np.copy(I)
    if large == True:
        II[II<=threshold] = 0
        II[II > threshold] = 1
    else:
        II[II!= threshold] = 0
        II[II==threshold] = 1
    
    center = ndimage.measurements.center_of_mass(II)
    return center


# move 3D image:
def move_3Dimage(image, d):
    if len(d) == 3:  # 3D

        d0, d1, d2 = d
        S0, S1, S2 = image.shape

        start0, end0 = 0 - d0, S0 - d0
        start1, end1 = 0 - d1, S1 - d1
        start2, end2 = 0 - d2, S2 - d2

        start0_, end0_ = max(start0, 0), min(end0, S0)
        start1_, end1_ = max(start1, 0), min(end1, S1)
        start2_, end2_ = max(start2, 0), min(end2, S2)

        # Crop the image
        crop = image[start0_: end0_, start1_: end1_, start2_: end2_]
        crop = np.pad(crop,
                        ((start0_ - start0, end0 - end0_), (start1_ - start1, end1 - end1_),
                        (start2_ - start2, end2 - end2_)),
                        'constant')

    if len(d) == 2: # 2D
        d0, d1 = d
        S0, S1 = image.shape

        start0, end0 = 0 - d0, S0 - d0
        start1, end1 = 0 - d1, S1 - d1

        start0_, end0_ = max(start0, 0), min(end0, S0)
        start1_, end1_ = max(start1, 0), min(end1, S1)

        # Crop the image
        crop = image[start0_: end0_, start1_: end1_]
        crop = np.pad(crop,
                        ((start0_ - start0, end0 - end0_), (start1_ - start1, end1 - end1_)),
                        'constant')

    return crop


# make heart center to image center:
def move_heart_center_to_image_center(img, threshold = 0, large = True, save_move_info = False):
    center_mass = center_of_mass(img, threshold,large)
    center_mass = [int(i) for i in center_mass]
    center_image = [ img.shape[i] // 2 for i in range(0,len(img.shape))]
    move = [ center_image[i] - center_mass[i] for i in range(0,len(center_mass))]

    img_m =  move_3Dimage(img, move)  
    if save_move_info == False:
        return img_m
    else:
        return img_m, move


# down-sample image in z-direction:
def downsample_in_z(img,factor = 5, affine = None): 
    if img.shape[-1]%factor == 0:
        z_num = img.shape[-1]//factor
    else:
        # z_num = img.shape[-1]//factor + 1
        z_num = img.shape[-1]//factor 
        
    img_ds = np.zeros((img.shape[0],img.shape[1],z_num))

    for i in range(0,z_num):
        s = factor * i + 0
        img_ds[:,:,i] = img[:,:,s]

    if affine is None:
        return img_ds
    else:
        Transform = np.eye(4)
        Transform[0:3,2] = np.array([0,0,1]) * factor
        new_affine = np.dot(affine,Transform)
        return img_ds, new_affine


# Get Largest component
def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def pick_largest(raw_image,labels = [1,2,4]):
    blank = np.zeros((raw_image.shape))

    for l in labels:
        answer = np.copy(raw_image)
        answer_tem = np.copy(answer)
        answer_tem[answer_tem != l] = 0
        a = getLargestCC(answer_tem)
        blank[a==True] = l

    return blank


# relabel
def relabel(x,original_label = 4,new_label = 3):
    x[x==original_label] = new_label
    return x


# one-hot encode
# def one_hot(image, num_classes):
#     """
#   One-hot encode an image
#   """
#     return np.reshape(
#         to_categorical(image.flatten(), num_classes=num_classes),
#         image.shape + (num_classes,),
#     )

def one_hot(image, num_classes):
    # Reshape the image to a 2D array
    image_2d = image.reshape(-1)

    # Perform one-hot encoding using NumPy's eye function
    encoded_image = np.eye(num_classes, dtype=np.uint8)[image_2d]

    # Reshape the encoded image back to the original shape
    encoded_image = encoded_image.reshape(image.shape + (num_classes,))

    return encoded_image


# remove some labels 
def change_num_labels(img, new_num_classes):
    img_new = np.copy(img)

    class_list = np.sort(np.unique(img))
    final_class_list = class_list[0: new_num_classes]
 
    for c in class_list:
        if c not in final_class_list:
            img_new[img_new == c] = 0

    correct, labels = correct_label(img_new)
   
    assert labels.shape[0] == new_num_classes

    return img_new, labels

# remove particular (non-LV / non-heart) slices
def remove_z_slices(img, category, pixel_num_threshold = 0):
    if category[0:2] == 'LV':
        slices = [i for i in range(0,img.shape[-1]) if np.where(img[:,:,i] == 1)[0].shape[0] >= pixel_num_threshold]
    if category[0:2] == 'He': # Heart
        slices = [i for i in range(0,img.shape[-1]) if np.where((img[:,:,i] == 1) | (img[:,:,i] == 2))[0].shape[0] >= pixel_num_threshold]
    
    for i in range(0,img.shape[-1]):
        if i not in slices:
            img[:,:,i] = 0
    return img


# input and output adapter for the DL model
def adapt(x, num_img_classes, num_hot_classes, remove_slices , remove_pixel_num_threshold,  remove_label = False, do_relabel_RV = True, do_relabel_myo = True, crop_target_size = None, do_one_hot = None, expand = False):
    # load data
    img = nb.load(x).get_fdata()
    # make sure it's integer
    img= np.round(img)
    img = img.astype(np.int32)

    # only keep slices with LV
    if remove_slices[0:2] == 'LV':
        img = remove_z_slices(img, 'LV', remove_pixel_num_threshold)
    elif remove_slices[0:2] == 'He':
        img = remove_z_slices(img, 'Heart', remove_pixel_num_threshold)

    # relabel RV
    if do_relabel_RV == True:
        img = relabel(img,4 ,3)

    # remove some labels if needed: e.g. remove RV
    if remove_label == True:
        img,labels = change_num_labels(img,num_img_classes)
        # print('after remove labels: ',labels)
  
    # relabel myocardium
    if do_relabel_myo == True:
        img = relabel(img,2, 1)
        # print('after relabel: ',np.unique(img))

    # crop
    if crop_target_size is not None:
        img = crop_or_pad(img,crop_target_size)
        
    # one-hot
    if do_one_hot == True:
        img = one_hot(img,num_hot_classes)
    # expand
    if expand == True:
        img = np.expand_dims(img,axis = -1)

    # print('after adapt, shape of x is: ',img.shape)
    return img.astype(np.int32)



# get contour-to-contour distance
def c2c(image,spacing, threshold = 1, large = False ):
    assert len(spacing) == 2
    centers = []; distance = []
    for i in range(0,image.shape[0]):
        s = image[i,:,:]
        center = center_of_mass(s, threshold, large)
        if math.isnan(center[0]) == 0:
            centers.append(center)
            if len(centers) > 1:
                distance.append(math.sqrt(((centers[-1][0] - centers[-2][0])*spacing[0])**2 + ((centers[-1][1] - centers[-2][1])*spacing[1])**2))
    mean_distance  = sum(distance) / len(distance)
    return mean_distance, sum(distance), len(distance)
    

# crop or pad
def crop_or_pad(array, target, value=0):
    # Pad each axis to at least the target.
    margin = target - np.array(array.shape)
    padding = [(0, max(x, 0)) for x in margin]
    array = np.pad(array, padding, mode="constant", constant_values=value)
    for i, x in enumerate(margin):
        array = np.roll(array, shift=+(x // 2), axis=i)

    if type(target) == int:
        target = [target] * array.ndim

    ind = tuple([slice(0, t) for t in target])
    return array[ind]


# optim means we can move the prediction in a preset range, and find the moved one with highest dice.
def Dice_optim(pred, gt, k_list , d = 5, threeDimage = True):
    pred_copy = np.copy(pred)
    max_dice = 0
    best_dx, best_dy = 0, 0
    for dx in range(-d, d):
        for dy in range(-d, d):
            if threeDimage == True:
                pred_1 = move_3Dimage(pred_copy, (dx, dy, 0))
            else:
                pred_1 = move_3Dimage(pred_copy, (dx, dy))

            dice_1 = ff.np_mean_dice(pred_1, gt, k_list)
              
            if dice_1 > max_dice:
                best_pred = np.copy(pred_1)
                best_dx, best_dy = dx, dy
                max_dice = dice_1
    ds = [best_dx, best_dy]
    return max_dice, ds, best_pred


# optim means we can move the prediction in a preset range, and find the moved one with highest HD.
def HD_optim(pred, gt,  min, k_list, d = 5, pixel_size = 1.0, threeDimage = True):
    pred_copy = np.copy(pred)
    # gt pixels
    gt_pixels = []
    for jj in range(0,len(k_list)):
        gt_pixels.append(np.asarray(ff.count_pixel(gt,k_list[jj])[1]))
    
    min_mean_dis = 10000
    best_dx, best_dy = 0, 0
    for dx in range(-d, d):
        for dy in range(-d, d):
     
            if threeDimage == True:
                pred_1 = move_3Dimage(pred_copy, (dx, dy, 0))
            else:
                pred_1 = move_3Dimage(pred_copy, (dx, dy))

            mean_dis = []
            for jj in range(0,len(k_list)):
                pred_pixels = np.asarray(ff.count_pixel(pred_1, k_list[jj])[1]); 
                mean_dis.append(ff.HD(pred_pixels,gt_pixels[jj], pixel_size, min))

            mean_dis = np.mean(np.asarray(mean_dis))
              
            if mean_dis < min_mean_dis:
                best_pred = np.copy(pred_1)
                best_dx, best_dy = dx, dy
                min_mean_dis = mean_dis

    ds = [best_dx, best_dy]
    return min_mean_dis, ds, best_pred




''' slice volume and plot'''
def vol3view(vol, clim=(0,4), cmap='viridis', slices = [32,64,64]):
    " input volume: 64 x 128 x 128"
    plt.style.use('dark_background')
    fig, axs = plt.subplots(nrows=3, ncols=1, frameon=False,
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    fig.set_size_inches([1, 3])
    [ax.set_axis_off() for ax in axs.ravel()]

    view1 = vol[slices[0], :, :]
    view2 = vol[:, slices[1], :]
    view3 = vol[:, :, slices[-1]]

    axs[0].imshow(view1, clim=clim, cmap=cmap)
    axs[1].imshow(view2, clim=clim, cmap=cmap)
    axs[2].imshow(view3, clim=clim, cmap=cmap)

    return fig

# make some z slice as 0
def make_z_slice_blank(img, keep_slice_num):
    for i in range(0,img.shape[-1]):
        if i not in keep_slice_num:
            img[:,:,i] = 0
        
    return img


# make data augmentation (pick consecutive slices in image)
def make_slice_augmentation(img_low, img_high, remove_slice_num = 2, do_high_resolution = None, factor = 5, fill_apex = None):
    assert len(img_low.shape) == 3
    lv_slice = [i for i in range(0,img_low.shape[-1]) if np.where(img_low[:,:,i] == 1)[0].shape[0] != 0]
    # remove some consecutive slices (at most n slices), the final slice number (LV slice) should be not smaller than 7 slices
    aug_slice_num_list = [len(lv_slice ) - i for i in range(0,remove_slice_num + 1) if (len(lv_slice) - i) >= 7]
    # in case there is low lv slice number (below 7 slices):
    if len(aug_slice_num_list) == 0:
        aug_slice_num_list = [len(lv_slice)]
    # randomly pick one number from aug_slice_num_list
    aug_slice_num = aug_slice_num_list[math.floor(np.random.rand() * len(aug_slice_num_list))]
    
    # start slice index candidates
    start_slice_candidates = lv_slice[0: len(lv_slice) - aug_slice_num + 1]
    start_slice_i = math.floor(np.random.rand() * len(start_slice_candidates))
    lv_slice_augment = lv_slice[start_slice_i : start_slice_i + aug_slice_num]

    # make augmented low-resolution and high-resolution
    # low-resolution:
    img_low_aug = np.copy(img_low)
    if lv_slice_augment[0] == lv_slice[0]:
        # does LV apex include if we start from the first LV slice
        if fill_apex == True:
            apex_include_rand = np.random.rand()
            if apex_include_rand >= 0.5:
                lv_slice_augment = np.arange(0, lv_slice_augment[-1]+1).tolist()
    img_low_aug = make_z_slice_blank(img_low_aug, lv_slice_augment)

    # high-resolution
    if do_high_resolution == True:
        high_slice_augment = np.arange((lv_slice_augment[0] * factor + 0) , (lv_slice_augment[-1] * factor + factor)).tolist()
        img_high_aug = np.copy(img_high)
        img_high_aug = make_z_slice_blank(img_high_aug, high_slice_augment)
        return img_low_aug, img_high_aug, lv_slice, aug_slice_num, lv_slice_augment
    elif do_high_resolution == False:
        return img_low_aug, lv_slice, aug_slice_num, lv_slice_augment

# convert categorical mask (LV, RV, and myo) to contours
def mask_to_contourpts(mask, class_name_list, class_value_list):
    
    cts_dict = {"Endo": None, "Epi": None, "RV": None}
    
    for i in range(0,len(class_name_list)):
        name = class_name_list[i]
        
        if name[0:2] != 'Ep': # LV or RV
            contours_pred, _ = cv2.findContours(
                (mask == class_value_list[i]).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
        else:
            binary_mask = np.zeros_like(mask)
            binary_mask[(mask == class_value_list[i]) | (mask == class_value_list[i-1])] = 1  # only get epi
            contours_pred, _ = cv2.findContours(
                binary_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )


        # choose largest component
        shape = np.shape(contours_pred)


        if shape[0] > 1:
            pt_max = 0
            for i in range(shape[0]):
                if pt_max < np.shape(contours_pred[i])[0]:
                    pt_max = np.shape(contours_pred[i])[0]
                    out = np.reshape(contours_pred[i], (pt_max, 2))
        elif shape[0] == 0:
            continue
        else:
            out = np.reshape(contours_pred, (shape[1], 2))

            
        cts_dict[name] = out
        
    return cts_dict

# convert cnotour control points to categorical masks
def contourpts_to_mask(cts_dict,img, class_name_list, class_value_list, sample_rate = 1):

    final_img = np.zeros_like(img)

    for i in range(0,len(class_name_list)):
        name = class_name_list[i]
        value = class_value_list[i]
        if name[0:2] == 'LV':
            contours = np.copy(cts_dict['Endo'][::sample_rate])

        if name[0:2] == 'My':
            contours = np.copy(cts_dict['Epi'][::sample_rate])

            blank = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
            cv2.fillPoly(blank, [contours], color=(255, 255, 255))
            indices = np.where(blank != 0)
            coordinates = np.asarray(list(zip(indices[0], indices[1])))
            final_img2 = np.zeros_like(final_img)
            final_img2[coordinates[:,0], coordinates[:,1]] = class_value_list[i]
            final_img2[final_img== class_value_list[i-1]] = class_value_list[i-1]
            final_img = np.copy(final_img2)
            continue

        if name[0:2] == 'RV':
            contours = np.copy(cts_dict['RV'][::sample_rate])
        
        blank = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
        cv2.fillPoly(blank, [contours], color=(255, 255, 255))
        indices = np.where(blank != 0)
        coordinates = np.asarray(list(zip(indices[0], indices[1])))
        final_img[coordinates[:,0], coordinates[:,1]] = class_value_list[i]

    return final_img

# function: calculate circularity index
def circularity_index_cal(img):
    '''img should be a binary image'''
    # find the contour
    pts = mask_to_contourpts(img, ['Endo'], [1])
    contour_pts = pts['Endo']
    pts = np.copy(contour_pts)

    # calculate the perimeter
    total_length = 0
    used_pts = []
    start_pt = pts[0]
    while True:
        # stop when used_pts have all the points
        if len(used_pts) == len(contour_pts) - 1:
            break
        # delete the start point from pts
        start_idx = np.where((pts == start_pt).all(axis=1))[0][0]
        # deletet this  point from pts
        pts = np.delete(pts, start_idx, 0)

        # find the closest point to the start point except the itself
        dist = np.sqrt(np.sum((pts - start_pt)**2, axis = 1))
        min_dist = np.min(dist)
        min_idx = np.argmin(dist)
        # update the total length
        total_length += min_dist
        # put the start points into used_pts
        used_pts.append(start_pt)
        # update the start points
        start_pt = pts[min_idx]

    # calculate the area
    area = np.sum(img)

    # calculate the circularity index
    circularity_index = 4 * np.pi * area / total_length**2
    return circularity_index, contour_pts

# function: get circularity index for each slice in an image
def circularity_index_img(img_binary, heart_slices):
    circularity_index_list = []
    for s in range(0, len(heart_slices)):
        img_slice = img_binary[:,:,heart_slices[s]]
        circularity_index, contour_pts = circularity_index_cal(img_slice)
        circularity_index_list.append(circularity_index)
    
    return np.asarray(circularity_index_list)

# function: get LV center and deltas
def get_centers(img_binary, heart_slices, make_plot = False):
    # find the center of mass on each heart slice
    centers = []
    for s in range(0, len(heart_slices)):
        c = center_of_mass(img_binary[:,:,heart_slices[s]], 0, True)
        centers.append(c)

    centers = ff.remove_nan(centers) 

    centers_d = []
    for s in range(0, centers.shape[0]):
        if s == 0:
            centers_d.append([0,0])
        else:
            centers_d.append(centers[s] - centers[0])
    centers_d = np.array(centers_d)

    # plot centers_d as a 2D plot, each datapoint should be a red dot on the curve
    if make_plot:
        plt.figure()
        plt.plot(centers_d[:,0], centers_d[:,1], 'r--')
        plt.plot(centers_d[:,0], centers_d[:,1], 'ro')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    return centers_d, np.asarray(centers)

# function: get area enclosed by the centerpoints
def centers_enclosed_area(centers_d, make_plot = False):
    # calculate the enclosed area by centerpoints using convex hull
    # Assume points is a list of tuples representing (x, y) coordinates
    points = list(zip(np.round(centers_d[:,0]).astype(np.int), np.round(centers_d[:,1]).astype(np.int)))

    # Compute the Convex Hull
    hull = ConvexHull(points)

    # Get the coordinates of the convex polygon vertices
    vertices = np.append(hull.vertices, hull.vertices[0])  # Close the polygon loop

    # Calculate the area of the convex polygon
    area = hull.volume

    # Plot the points and the convex hull
    if make_plot:
        plt.figure()
        plt.plot(*zip(*points), 'o')
        plt.plot(np.array(points)[vertices, 0], np.array(points)[vertices, 1], 'k-')
        plt.show()
        print(f"Area of Convex Hull: {area}")
    return area

def major_minor_axis_len(img_binary):
    labeled_image = label(img_binary)
    props = regionprops(labeled_image)
    for obj in props:
        # Calculate major and minor axis lengths of the object
        major_axis_length = obj.major_axis_length
        minor_axis_length = obj.minor_axis_length
    return major_axis_length   , minor_axis_length




    


    
