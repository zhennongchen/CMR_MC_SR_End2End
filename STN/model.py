import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv3D, MaxPool2D, MaxPool3D,UpSampling2D,UpSampling3D,
    Reshape,ZeroPadding2D,ZeroPadding3D, AveragePooling2D, AveragePooling3D, GlobalAveragePooling3D
)
from tensorflow.keras.layers import Concatenate,Multiply,PReLU, Reshape, Add, Flatten, Dense, LeakyReLU, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Layer
from voxelmorph.tf.layers import SpatialTransformer

import CMR_HFpEF_Analysis.Defaults as Defaults

conv_dict = {2: Conv2D, 3: Conv3D}
max_pooling_dict = {2:MaxPool2D, 3:MaxPool3D}
average_pooling_dict = {2: AveragePooling2D, 3: AveragePooling3D}
zero_sampling_dict = {2: ZeroPadding2D, 3: ZeroPadding3D}

cg = Defaults.Parameters()

def conv_bn_relu_1x(nb_filter, kernel_size, subsample = (1,), dimension = 3, batchnorm = False, activation = False):
    stride = subsample * dimension # stride
    Conv = conv_dict[dimension]

    def f(input_layer):
        x = Conv(
            filters=nb_filter,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="orthogonal",
            kernel_regularizer=l2(1e-4),
            bias_regularizer=l2(1e-4),
            )(input_layer)
        if batchnorm == True:
            x = BatchNormalization()(x)
        if activation == True:
            # x = PReLU()(x)
            x = LeakyReLU(alpha=0.1)(x)
        return x
    return f

def res_block(nb_filter,  subsample = (1,), dimension = 3):
    def f(input_layer):
        x = conv_bn_relu_1x(nb_filter, (3,3,3), subsample, dimension,batchnorm = True, activation = True)(input_layer)
        x = conv_bn_relu_1x(nb_filter, (3,3,3), subsample, dimension,batchnorm = True, activation = False)(x)
        input_x = conv_bn_relu_1x(nb_filter, (1,1,1), subsample, dimension,batchnorm = False, activation = False)(input_layer)
        final = Add()([input_x, x])
        return final
    return f

def fully_connect(output_num ,name, hidden_filter = 64):
    def f(input_layer):
        slice1 =  Dense(hidden_filter, activation = LeakyReLU(alpha=0.1), use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4))(input_layer)
        slice1 = Dropout(0.2)(slice1)
        
        slice1 =  Dense(output_num, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = name)(slice1)
        return slice1
    return f


def main_model(nb_filters, output_num = 2, subsample = (1,), dimension = 3):
    max_pool = max_pooling_dict[dimension]
    average_pool = average_pooling_dict[dimension]

    def f(input_layer):
        levels = []
        levels += [conv_bn_relu_1x(8,(2,2,2), subsample, dimension, batchnorm = True, activation = True)(input_layer)]

        for i in range(0,4):
            levels += [res_block(nb_filters[i])(levels[-1])]
            if i != 3:
                levels += [max_pool(pool_size = (2,2,1))(levels[-1])]
            else:
                levels += [conv_bn_relu_1x(16, (1,1,1), subsample, dimension,batchnorm = False, activation = False)(levels[-1])]
                # levels += [GlobalAveragePooling3D()(levels[-1])]


        flat =  Flatten()(levels[-1])
        flat = Dropout(0.2)(flat)

        slice1 = fully_connect(output_num, 'slice1')(flat); slice2 = fully_connect(output_num, 'slice2')(flat); slice3 = fully_connect(output_num, 'slice3')(flat)
        slice4 = fully_connect(output_num, 'slice4')(flat); slice5 = fully_connect(output_num, 'slice5')(flat); slice6 = fully_connect(output_num, 'slice6')(flat)
        slice7 = fully_connect(output_num, 'slice7')(flat); slice8 = fully_connect(output_num, 'slice8')(flat); slice9 = fully_connect(output_num, 'slice9')(flat)
        slice10 = fully_connect(output_num, 'slice10')(flat); slice11 = fully_connect(output_num, 'slice11')(flat); slice12 = fully_connect(output_num, 'slice12')(flat)

        combined_results = tf.keras.layers.concatenate([slice1, slice2, slice3, slice4, slice5, slice6, slice7, slice8, slice9, slice10, slice11, slice12], axis=1)
        combined_results = tf.reshape(combined_results, (-1,12, output_num))

        # apply STN
        s1 = apply_STN()(combined_results, 0, input_layer); s2 = apply_STN()(combined_results, 1, input_layer); s3 = apply_STN()(combined_results, 2, input_layer)
        s4 = apply_STN()(combined_results, 3, input_layer); s5 = apply_STN()(combined_results, 4, input_layer); s6 = apply_STN()(combined_results, 5, input_layer)
        s7 = apply_STN()(combined_results, 6, input_layer); s8 = apply_STN()(combined_results, 7, input_layer); s9 = apply_STN()(combined_results, 8, input_layer)
        s10 = apply_STN()(combined_results, 9, input_layer); s11 = apply_STN()(combined_results, 10, input_layer); s12 = apply_STN()(combined_results, 11, input_layer)

        final_img = tf.concat([s1,s2,s3,s4, s5, s6, s7, s8, s9, s10, s11, s12], axis = -1)
        print('final image shape: ', final_img.shape)
        return combined_results,  final_img
    
    return f



# model componenet:
    
class element_2D(Layer):
    def __init__(self,index1, index2):
        super(element_2D, self).__init__()
        self.index1 = index1
        self.index2 = index2

    def get_config(self):
        config = super().get_config().copy()
        config.update({'index1' : self.index1,  'index2' : self.index2,})
        return config
        
    
    def call(self, inputs):
        output = inputs[...,self.index1, self.index2]; output = tf.reshape(output,[-1,1,1])
        return output
    
    
class make_transform_matrix(Layer): 
    def __init__(self):
        super(make_transform_matrix, self).__init__()

    def translation(self,tx,ty):
        first_row = tf.reshape(tf.concat([tf.ones_like(tx) , tf.zeros_like(tx), tx], axis = -1),[-1,1,3])
        second_row = tf.reshape(tf.concat([tf.zeros_like(ty) , tf.ones_like(ty), ty], axis = -1), [-1,1,3])

        return tf.concat([first_row, second_row], axis = 1)


    def call(self, tx, ty):
        matrix = self.translation(tx, ty)
        return matrix


class apply_STN(Layer):
    def __init__(self):
        super(apply_STN, self).__init__()

    
    def call(self, combined_results, slice_index, input_layer):
        slice_tx = element_2D(slice_index,0)(combined_results)
        slice_ty = element_2D(slice_index,1)(combined_results)
        affine_matrix = make_transform_matrix()(slice_tx, slice_ty)
        slice_img = tf.reshape(input_layer[:,:,:,slice_index,:], [-1,128,128,1])
        s = SpatialTransformer(interp_method='nearest',indexing='ij',single_transform=False,fill_value=0,shift_center=True)((slice_img, affine_matrix))

        return s

      