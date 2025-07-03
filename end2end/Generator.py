## tutorial: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import os
from tensorflow.keras.utils import Sequence
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.motion_correction.data_augmentation as aug

cg = Defaults.Parameters()


class DataGenerator(Sequence):

    def __init__(self,
                 X,
                 Y_center,
                 Y_LR_img,
                 Y_HR_img,
        patient_num = None, 
        batch_size = None, 
        num_classes = None,
        input_dimension = (128,128,12),
        output_vector_dimension = (12,2),
        output_img_dimension = (128,128,60),
        shuffle = None,
        remove_slices = 'None',
        remove_pixel_num_threshold = None,
        remove_label = None,
        relabel_myo = None,
        slice_augment = None,
        seed = 10):

        self.X = X
        self.Y_center = Y_center
        self.Y_LR_img = Y_LR_img
        self.Y_HR_img = Y_HR_img

        self.patient_num = patient_num
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_dimension = input_dimension
        self.output_vector_dimension = output_vector_dimension
        self.output_img_dimension = output_img_dimension

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
        batch_y_center = np.zeros(tuple([current_batch_size]) + self.output_vector_dimension)
        batch_y_LR_img = np.zeros(tuple([current_batch_size]) + self.input_dimension + tuple([self.num_classes]))
        batch_y_HR_img = np.zeros(tuple([current_batch_size]) + self.output_img_dimension + tuple([self.num_classes]))
        

        for i,j in enumerate(indexes):
            x = self.X[j]
            y_center = self.Y_center[j]
            y_LR_img = self.Y_LR_img[j]
            y_HR_img = self.Y_HR_img[j]
            # print('paths: ', x, y_center, y_LR_img, y_HR_img)

            # imgs
            x = util.adapt(x, num_img_classes= self.num_classes, num_hot_classes = self.num_classes,  
                               remove_slices = self.remove_slices, remove_pixel_num_threshold = self.remove_pixel_num_threshold,
                               remove_label = self.remove_label, do_relabel_RV = False, do_relabel_myo = self.relabel_myo,  do_one_hot = False, expand = False)
            y_LR_img = util.adapt(y_LR_img, num_img_classes= self.num_classes, num_hot_classes = self.num_classes,  
                               remove_slices = self.remove_slices, remove_pixel_num_threshold = self.remove_pixel_num_threshold,
                               remove_label = self.remove_label, do_relabel_RV = False, do_relabel_myo = self.relabel_myo,  do_one_hot = False, expand = False)
            
            y_HR_img = util.adapt(y_HR_img, num_img_classes=self.num_classes, num_hot_classes = self.num_classes, 
                            remove_slices = self.remove_slices, remove_pixel_num_threshold = self.remove_pixel_num_threshold,
                            remove_label = self.remove_label, do_relabel_RV = False, do_relabel_myo = self.relabel_myo,  do_one_hot=False, expand = False)
            
            if self.slice_augment == True:
                aug_t, _,_,_ = aug.displacement_generator()
  
                x, _ = aug.augmentation(x, aug_t)
                y_LR_img,_ = aug.augmentation(y_LR_img, aug_t)
                y_HR_img,_ = aug.augmentation(y_HR_img,aug_t )
                # print('aug: ', aug_t, x.shape, np.unique(x), y_LR_img.shape, np.unique(y_LR_img), y_HR_img.shape, np.unique(y_HR_img))
            
            x = np.expand_dims(x, axis = -1)
            y_LR_img = util.one_hot(y_LR_img, self.num_classes)
            y_HR_img = util.one_hot(y_HR_img, self.num_classes)
            # print('finally: ', x.shape, np.unique(x), y_LR_img.shape, np.unique(y_LR_img), y_HR_img.shape, np.unique(y_HR_img))
   
           # center points
            motion_centers = np.load(y_center, allow_pickle = True)
            gt_centers = np.load(os.path.join(os.path.dirname(os.path.dirname(y_center)), 'ds/centerlist.npy'),allow_pickle = True)

            assert motion_centers.shape[0] == 12; assert gt_centers.shape[0] == 12

            movements = np.zeros([12,2])
            for row in range(0,12):
                if np.isnan(gt_centers[row,0]) == 1:
                    movements[row,:] = [0,0]
                else:
                    movements[row,0] = motion_centers[row,0] - gt_centers[row,0]    
                    movements[row,1] = motion_centers[row,1] - gt_centers[row,1]   
      
            
            batch_x[i] = x
            batch_y_center[i] = movements
            batch_y_LR_img[i] = y_LR_img
            batch_y_HR_img[i] = y_HR_img
            

        return batch_x, [batch_y_center, batch_y_LR_img, batch_y_HR_img]