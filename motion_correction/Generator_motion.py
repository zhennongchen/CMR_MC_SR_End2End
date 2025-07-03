## tutorial: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import math
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.motion_correction.Bspline as Bspline
import CMR_HFpEF_Analysis.motion_correction.data_augmentation as aug

cg = Defaults.Parameters()


class DataGenerator(Sequence):

    def __init__(self,X,Y,
        patient_num = None, 
        batch_size = None, 
        num_classes = None,
        input_dimension = (128,128,12),
        output_dimension = None,
        shuffle = None,
        remove_slices = 'None',
        remove_pixel_num_threshold = None,
        remove_label = None,
        relabel_myo = None,
        slice_augment = None,
        seed = 10):

        self.X = X
        self.Y = Y
        self.patient_num = patient_num
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.remove_slices = remove_slices
        self.remove_pixel_num_threshold = remove_pixel_num_threshold
        self.shuffle = shuffle
        self.remove_label = remove_label
        self.relabel_myo = relabel_myo
        self.slice_augment = slice_augment
        self.seed = seed

        self.on_epoch_end()
        
    def __len__(self):
        
        return self.X.shape[0]// self.batch_size

    def on_epoch_end(self):
        
        self.seed += 1
        # print('seed is: ',self.seed)

        patient_list = np.random.permutation(self.patient_num)
                
        self.indices = np.asarray(patient_list)
        # print('all indexes: ', self.indices,len(self.indices))

    def __getitem__(self,index):
        'Generate one batch of data'

        'Generate indexes of the batch'
        total_cases = self.patient_num 
        
        current_index = (index * self.batch_size) % total_cases
        if total_cases > current_index + self.batch_size:   # the total number of cases is adequate for next loop#
            current_batch_size = self.batch_size
        else:
            current_batch_size = total_cases - current_index  # approaching to the tend, not adequate, should reduce the batch size
       
        indexes = self.indices[current_index : current_index + current_batch_size]
        
        # print('indexes in this batch: ',indexes)

        # allocate memory
        batch_x = np.zeros(tuple([current_batch_size]) + self.input_dimension + (1,))
        batch_y1 = np.zeros(tuple([current_batch_size]) + self.output_dimension)
        batch_y2 = np.zeros(tuple([current_batch_size]) + self.output_dimension)
        

        for i,j in enumerate(indexes):
            # path to input
            x = self.X[j]
            y = self.Y[j]
            
            x = util.adapt(x, num_img_classes= self.num_classes, num_hot_classes = self.num_classes,  
                               remove_slices = self.remove_slices, remove_pixel_num_threshold = self.remove_pixel_num_threshold,
                               remove_label = self.remove_label, do_relabel_RV = False, do_relabel_myo = self.relabel_myo,  do_one_hot = False, expand = False)
            
            if self.slice_augment == True:
                aug_t, _,_,_ = aug.displacement_generator()
                x, _ = aug.augmentation(x, aug_t)
            x = np.expand_dims(x, axis = -1)
            # print('x shape: ', x.shape, ' x value: ', np.unique(x))

            # y - center points
            center_list = ff.remove_nan(np.load(y, allow_pickle = True))
            center_x_CP, center_x = Bspline.bspline_and_control_points(center_list, self.output_dimension[0] + 1 , x_or_y = 'x')
            center_y_CP,center_y = Bspline.bspline_and_control_points(center_list, self.output_dimension[0] + 1 , x_or_y = 'y')

            delta_x, delta_x_n = Bspline.delta_for_centerline(center_x_CP,anchor_index = 0, image_shape = 128)
            delta_y, delta_y_n = Bspline.delta_for_centerline(center_y_CP, anchor_index = 0,image_shape = 128)            

            # print('center list: ', center_list)
            # print('center_x: ', center_y)
            # print('center_x_cp: ',center_y_CP)
            # print('delta_x: ',delta_y)


            batch_x[i] = x
            batch_y1[i] = delta_x_n[1:]
            batch_y2[i] = delta_y_n[1:]
           

        return batch_x, [batch_y1, batch_y2]
    


# # previous slice num augmentation:
# img_low = util.adapt(x, num_img_classes=self.num_classes + 1, num_hot_classes = self.num_classes, only_keep_LV_slices=True, remove_label = self.remove_label, do_relabel_RV = False, do_relabel_myo = False, do_one_hot=False)
               
# x, lv_slice, aug_slice_num, lv_slice_augment = util.make_slice_augmentation(img_low,
#                                                                  np.zeros([0]), remove_slice_num=1, do_high_resolution = False, factor = 5, fill_apex = False)
                
# if self.relabel_myo == True:
#     x = util.relabel(x,2,1)
#     x = np.expand_dims(x,axis = -1)
# center_list = np.load(y, allow_pickle = True)
# center_list_aug = ff.remove_nan(center_list[lv_slice_augment])
# center_x_CP, center_x = Bspline.bspline_and_control_points(center_list_aug, 7, x_or_y = 'x')
# center_y_CP, _ = Bspline.bspline_and_control_points(center_list_aug, 7, x_or_y = 'y')

# delta_x, delta_x_n = Bspline.delta_for_centerline(center_x_CP, anchor_index = 0, image_shape = 128)
#  _, delta_y_n = Bspline.delta_for_centerline(center_y_CP, anchor_index = 0, image_shape = 128)