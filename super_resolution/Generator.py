## tutorial: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
from tensorflow.keras.utils import Sequence
import CMR_HFpEF_Analysis.Defaults as Defaults
import CMR_HFpEF_Analysis.Image_utils as util
import CMR_HFpEF_Analysis.motion_correction.data_augmentation as aug

cg = Defaults.Parameters()

class DataGenerator(Sequence):

    def __init__(self,X,Y,
        patient_num = None, 
        batch_size = None, 
        num_classes = None,
        input_dimension = [128,128,12],
        output_dimension = [128,128,60],
        shuffle = None,
        remove_slices = None,
        remove_pixel_num_threshold = None,
        remove_label = None,
        relabel_RV = None,
        relabel_myo = None,
        augment = None,
        # num_classes_exist_after_remove = None,
        seed = 10):

        self.X = X
        self.Y = Y
        self.patient_num = patient_num
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.shuffle = shuffle
        self.remove_slices = remove_slices
        self.remove_pixel_num_threshold = remove_pixel_num_threshold
        self.remove_label = remove_label
        self.relabel_RV = relabel_RV
        self.relabel_myo = relabel_myo
        self.augment = augment
        # self.num_classes_exist = num_classes_exist_after_remove
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
        if total_cases > current_index + self.batch_size:   # the total number of cases is adequate for next loop
            current_batch_size = self.batch_size
        else:
            current_batch_size = total_cases - current_index  # approaching to the tend, not adequate, should reduce the batch size
       
        indexes = self.indices[current_index : current_index + current_batch_size]
        

        # allocate memory
        batch_x = np.zeros(tuple([current_batch_size]) + tuple([self.input_dimension[0],self.input_dimension[1], self.input_dimension[2]]) + tuple([self.num_classes]))
        batch_y= np.zeros(tuple([current_batch_size]) + tuple([self.output_dimension[0],self.output_dimension[1], self.output_dimension[2]]) + tuple([self.num_classes]))
        

        for i,j in enumerate(indexes):
            # path to input
            x = self.X[j]
            y = self.Y[j]
        
            x = util.adapt(x, num_img_classes=self.num_classes, num_hot_classes = self.num_classes, 
                            remove_slices = self.remove_slices, remove_pixel_num_threshold = self.remove_pixel_num_threshold, 
                            remove_label = self.remove_label, do_relabel_RV = self.relabel_RV, do_relabel_myo = self.relabel_myo,  do_one_hot=False)
            y = util.adapt(y, num_img_classes=self.num_classes, num_hot_classes = self.num_classes, 
                            remove_slices = self.remove_slices, remove_pixel_num_threshold = self.remove_pixel_num_threshold,
                            remove_label = self.remove_label, do_relabel_RV = self.relabel_RV, do_relabel_myo = self.relabel_myo,  do_one_hot=False)
            
            if self.augment == True:
                aug_t, _,_,_ = aug.displacement_generator()
                x, _ = aug.augmentation(x, aug_t)
                y, _ = aug.augmentation(y, aug_t)
               

            x = util.one_hot(x, self.num_classes)
            y = util.one_hot(y, self.num_classes)
            
            batch_x[i] = x
            batch_y[i] = y

        return batch_x,batch_y
    

# previous slice number augmentation
 # else:
            #     img_low = util.adapt(x, num_img_classes=self.num_classes, num_hot_classes = self.num_classes, 
            #                         remove_slices = self.remove_slices, remove_pixel_num_threshold = self.remove_pixel_num_threshold, 
            #                         remove_label = self.remove_label, do_relabel_RV = self.relabel_RV, do_relabel_myo = False, do_one_hot=False)
            #     img_high = util.adapt(y, num_img_classes=self.num_classes, num_hot_classes = self.num_classes, 
            #                         remove_slices = self.remove_slices, remove_pixel_num_threshold = self.remove_pixel_num_threshold, 
            #                         remove_label = self.remove_label, do_relabel_RV = self.relabel_RV, do_relabel_myo = False,  do_one_hot=False)

            #     x, y, lv_slice, aug_slice_num, lv_slice_augment = util.make_slice_augmentation(img_low, img_high, remove_slice_num = 1, do_high_resolution = True, factor = 5, fill_apex = True)
                
            #     if self.relabel_myo == True:
            #         x = util.relabel(x,2,1)
            #         y = util.relabel(y,2,1)
                    
            #     x = util.one_hot(x, cg.num_classes)
            #     y = util.one_hot(y, cg.num_classes)
                # print(lv_slice, aug_slice_num, lv_slice_augment, x.shape, y.shape)