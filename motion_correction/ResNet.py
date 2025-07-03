import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv3D, MaxPool2D, MaxPool3D,UpSampling2D,UpSampling3D,
    Reshape,ZeroPadding2D,ZeroPadding3D, AveragePooling2D, AveragePooling3D, GlobalAveragePooling3D
)
from tensorflow.keras.layers import Concatenate,Multiply,PReLU, Reshape, Add, Flatten, Dense, LeakyReLU, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as kb

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


def main_model(nb_filters, output_num = 6, subsample = (1,), dimension = 3):
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

        # print('before flatten shape: ', levels[-1].shape)

        fc =  Flatten()(levels[-1])
        fc = Dropout(0.2)(fc)

        # print('after flatten shape: ', fc.shape)

        fc = Dense(64, activation = LeakyReLU(alpha=0.1), use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'hidden')(fc)

        fc =  Dropout(0.2)(fc)

        center_x = Dense(output_num, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'center_x')(fc)
        center_y = Dense(output_num, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'center_y')(fc)

        return center_x, center_y
    return f