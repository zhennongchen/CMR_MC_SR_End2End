import numpy as np
import glob 
import os
from PIL import Image
import math
from scipy import ndimage
import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import RegularGridInterpolator
from nibabel.affines import apply_affine
import re
import cv2 

# function: match nii img orientation to nrrd/dicom img orientation
def nii_to_nrrd_orientation(nii_data):
    return np.flip(np.rollaxis(np.flip(nii_data, axis=2),1,0), axis = 0)

# function: match nrrd/dicom img orientation to nii img orientation
def nrrd_to_nii_orientation(nrrd_data, format = 'nrrd'):
    if format[0:4] == 'nrrd':
        nrrd_data = np.rollaxis(nrrd_data,0,3)
    return np.rollaxis(np.flip(np.rollaxis(np.flip(nrrd_data, axis=0), -2, 2), axis = 2),1,0)

# function: get first X numbers
# if we have 1000 numbers, how to get the X number of every interval numbers?
def get_X_numbers_in_interval(total_number, start_number, end_number , interval = 10):
    n = []
    for i in range(0, total_number, interval):
      n += [i + a for a in range(start_number,end_number)]
    n = np.asarray(n)
    return n


# function: set window level and windth
def set_window(image,level,width):
    if len(image.shape) == 3:
        image = image.reshape(image.shape[0],image.shape[1])
    new = np.copy(image)
    high = level + width
    low = level - width
    # normalize
    unit = (1-0) / (width*2)
    image[image>high] = high
    image[image<low] = low
    new = (image - low) * unit 
    return new

# function: save itk
def save_itk(img, save_file_name, previous_file, new_voxel_dim = None, new_affine = None):
    image = sitk.GetImageFromArray(img)

    image.SetDirection(previous_file.GetDirection())
    image.SetOrigin(previous_file.GetOrigin())

    if new_voxel_dim is None:
        image.SetSpacing(previous_file.GetSpacing())
    else:
        image.SetSpacing(new_voxel_dim)

    if new_affine is None:
        image.SetMetaData("TransformMatrix", previous_file.GetMetaData("TransformMatrix"))
    else:
        affine_matrix_str = np.array2string(new_affine, separator=',')
        image.SetMetaData("TransformMatrix", affine_matrix_str)


    sitk.WriteImage(image, save_file_name)


# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(glob.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F

# function: find time frame of a file
def find_timeframe(file,num_of_dots,start_signal = '/',end_signal = '.'):
    k = list(file)

    if num_of_dots == 0: 
        num = [i for i,e in enumerate(k) if e== start_signal][-1]
        num1 = [i for i, e in enumerate(k) if e == end_signal][-1]
        kk = k[num+1:num1]
    
    else:
        if num_of_dots == 1: #.png
            num1 = [i for i, e in enumerate(k) if e == end_signal][-1]
        elif num_of_dots == 2: #.nii.gz
            num1 = [i for i, e in enumerate(k) if e == end_signal][-2]
        num2 = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk=k[num2+1:num1]

    total = 0
    for i in range(0,len(kk)):
        total += int(kk[i]) * (10 ** (len(kk) - 1 -i))
    return total


# function: sort files based on their time frames
def sort_timeframe(files,num_of_dots,start_signal = '/',end_signal = '.'):
    time=[]
    time_s=[]
    
    for i in files:
        a = find_timeframe(i,num_of_dots,start_signal,end_signal)
        time.append(a)
        time_s.append(a)
    time_s.sort()
    new_files=[]
    for i in range(0,len(time_s)):
        j = time.index(time_s[i])
        new_files.append(files[j])
    new_files = np.asarray(new_files)
    return new_files

# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)


# function: save grayscale image
def save_grayscale_image(a,save_path,normalize = True):
    I = np.zeros((a.shape[0],a.shape[1],3))
    # normalize
    if normalize == True:
        a = (a-np.min(a)) / (np.max(a) - np.min(a))

    for i in range(0,3):
        I[:,:,i] = a
    
    Image.fromarray((I*255).astype('uint8')).save(save_path)

# function: remove nan from list:
def remove_nan(l, show_row_index = False):
    l_new = []
    a = np.sum(np.isnan(l),axis = 1)
    non_nan_row_index = []
    for i in range(0,a.shape[0]):
        if a[i] == 0:
            l_new.append(l[i])
            non_nan_row_index.append(i)
    l_new = np.asarray(l_new)
    non_nan_row_index = np.asarray(non_nan_row_index)
    if show_row_index == True:
        return l_new, non_nan_row_index
    else:
        return l_new

# function: eucliean distance excluding nan:
def ED_no_nan(pred, gt):
    ED = []
    for row in range(0,gt.shape[0]):
        if np.isnan(gt[row,0]) == 1:
            continue
        else:
            ED.append(math.sqrt((gt[row,0] - pred[row,0]) ** 2 +  (gt[row,1] - pred[row,1]) ** 2))
    return sum(ED) / len(ED)

        
# function: normalize a vector:
def normalize(x):
    x_scale = np.linalg.norm(x)
    return np.asarray([i/x_scale for i in x])

# function: count pixels belonged to one class
def count_pixel(seg,target_val):
    index_list = np.where(seg == target_val)
    count = index_list[0].shape[0]
    pixels = []
    for i in range(0,count):
        p = []
        for j in range(0,len(index_list)):
            p.append(index_list[j][i])
        pixels.append(p)
    return count,pixels

# function: optimize the prediction
def optimize(pred, true, mode = [0,1,2]):
    true = np.reshape(true, -1)
    assert pred.shape[1] == true.shape[0]

    final_answer_list = []

    # average
    final_answer_average = np.zeros(pred.shape[1])
    for j in range(0,pred.shape[1]):
        col = pred[:,j]
        col_rank = np.sort(col)
        final_answer_average[j] = np.mean(col_rank)
    
    if 0 in mode:
        final_answer_list.append(final_answer_average)

    # pick best model (w/lowest MAE)
    errors = np.mean(np.abs(pred - true), axis=1)
    min_index = np.argmin(errors)
    final_answer_model_mae = pred[min_index]
    if 1 in mode:
        final_answer_list.append(final_answer_model_mae)
    
    # pick best model (w/lowest largest difference)
    errors = np.max(np.abs(pred - true), axis=1)
    min_index = np.argmin(errors)
    final_answer_model_diff = pred[min_index]
    if 2 in mode:
        return final_answer_model_diff
        # final_answer_list.append(final_answer_model_diff)

    # pick element-wise
    mae = np.abs(pred - true)
    # Find the index of the element with the smallest MAE in each column
    index = np.argmin(mae, axis=0)
    
    # Select the corresponding elements from each column of "pred"
    final_answer_model_element = pred[index, np.arange(pred.shape[1])]
    if 3 in mode:
        return final_answer_model_element
    
    final_answer_list = np.reshape(np.asarray(final_answer_list), (-1, pred.shape[1]))
    errors = np.mean(np.abs(final_answer_list - true), axis=1)
    min_index = np.argmin(errors)
   
    return final_answer_list[min_index]


# Dice calculation
def np_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))


def np_mean_dice(pred, truth, k_list = [1,2]):
    """ Dice mean metric """
    dsc = []
    for k in k_list:
        dsc.append(np_categorical_dice(pred, truth, k))
    return np.mean(dsc)


def HD(pred,gt, pixel_size, min ):
    hd1 = directed_hausdorff(pred, gt)[0] * pixel_size
    hd2 = directed_hausdorff(gt, pred)[0] * pixel_size
    if min == False:
        return hd1
    else:
        return np.min(np.array([hd1, hd2]))

# function: coordinate conversion according to the affine matrix
def coordinate_convert(grid_points, target_affine, original_affine):
    return apply_affine(np.linalg.inv(target_affine).dot(original_affine), grid_points)

# function: interpolation
def define_interpolation(data,Fill_value=0,Method='nearest'):
    shape = data.shape
    [x,y,z] = [np.linspace(0,shape[0]-1,shape[0]),np.linspace(0,shape[1]-1,shape[1]),np.linspace(0,shape[-1]-1,shape[-1])]
    interpolation = RegularGridInterpolator((x,y,z),data,method=Method,bounds_error=False,fill_value=Fill_value)
    return interpolation


# function: reslice a mpr
def reslice_mpr(mpr_data,plane_center,x,y,x_s,y_s,interpolation):
    # plane_center is the center of a plane in the coordinate of the whole volume
    mpr_shape = mpr_data.shape
    new_mpr=[]
    centerpoint = np.array([(mpr_shape[0]-1)/2,(mpr_shape[1]-1)/2,0])
    for i in range(0,mpr_shape[0]):
        for j in range(0,mpr_shape[1]):
            delta = np.array([i,j,0])-centerpoint
            v = plane_center + (x*x_s)*delta[0]+(y*y_s)*delta[1]
            new_mpr.append(v)
    new_mpr=interpolation(new_mpr).reshape(mpr_shape)
    return new_mpr


# switch two classes in one image:
def switch_class(img, class1, class2):
    new_img = np.where(img == class1, class2, np.where(img == class2, class1, img))
    return new_img


# function: write txt file
def txt_writer(save_path,parameters,names):
    t_file = open(save_path,"w+")
    for i in range(0,len(parameters)):
        t_file.write(names[i] + ': ')
        for ii in range(0,len(parameters[i])):
            t_file.write(str(round(parameters[i][ii],2))+' ')
        t_file.write('\n')
    t_file.close()

def txt_writer2(save_path,record):
    t_file = open(save_path,"w+")
    for i in range(0,len(record)):
        r = record[i]
        t_file.write('slice '+ str(r[0]) + ', total_distance: ' + str(round(r[1],2)) + 'mm, vector mm: ' 
                        + str(round(r[2][0],2)) + ' ' + str(round(r[2][1],2)) + ' vector pixel: ' + str(round(r[3][0],2)) + ' ' + str(round(r[3][1],2))
                        + ' rotation: '+str(r[4]) + ' degree' )
        if i != (len(record) - 1):
            t_file.write('\n')
    t_file.close()


# function: from ID_00XX to XX:
def ID_00XX_to_XX(input_string):
    # Find the first non-zero number in the string
    match = re.search(r'\d+', input_string)
    if match:
        number_string = match.group()
        return int(number_string)
    else:
        return None
    
# function: from XX to ID_00XX:
def XX_to_ID_00XX(num):
    if num < 10:
        return 'ID_000' + str(num)
    elif num>= 10 and num< 100:
        return 'ID_00' + str(num)
    elif num>= 100 and num < 1000:
        return 'ID_0' + str(num)
    elif num >= 1000:
        return 'ID_' + str(num)

# function: make movies
def make_movies(save_path,pngs,fps):
    mpr_array=[]
    i = cv2.imread(pngs[0])
    h,w,l = i.shape

    for j in pngs:
        img = cv2.imread(j)
        mpr_array.append(img)


    # save movies
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    for j in range(len(mpr_array)):
        out.write(mpr_array[j])
    out.release()


# function: erode and dilate
def erode_and_dilate(img_binary, kernel_size, erode = None, dilate = None):
    img_binary = img_binary.astype(np.uint8)

    kernel = np.ones(kernel_size, np.uint8)  

    if dilate is True:
        img_binary = cv2.dilate(img_binary, kernel, iterations = 1)

    # if erode is True:
    #     img_binary = cv2.erode(img_binary, kernel, iterations = 1)
    return img_binary

    
