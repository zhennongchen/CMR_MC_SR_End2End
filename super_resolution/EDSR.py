import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv3D, MaxPool2D, MaxPool3D,UpSampling2D,UpSampling3D,
    Reshape,ZeroPadding2D,ZeroPadding3D,
)
from tensorflow.keras.layers import Concatenate,Multiply,PReLU, Reshape, Add, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as kb

import CMR_HFpEF_Analysis.Defaults as Defaults

conv_dict = {2: Conv2D, 3: Conv3D}
max_pooling_dict = {2: MaxPool2D, 3: MaxPool3D}
up_sampling_dict = {2: UpSampling2D, 3: UpSampling3D}
zero_sampling_dict = {2: ZeroPadding2D, 3: ZeroPadding3D}

cg = Defaults.Parameters()

def conv_bn_relu_1x(nb_filter, kernel_size, subsample = (1,), dimension = 3, batchnorm = False, activation = False):
    sub = subsample * dimension # stride
    Conv = conv_dict[dimension]

    def f(input_layer):
       
        x = Conv(
            filters=nb_filter,
            kernel_size=kernel_size,
            strides=sub,
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
            x = LeakyReLU()(x)
        return x

    return f

def res_block(nb_filter, kernel_size, subsample = (1,), dimension = 3):
    
    def f(input_layer):
        x = conv_bn_relu_1x(nb_filter, kernel_size, subsample, dimension,batchnorm = True, activation = True)(input_layer)
        x = conv_bn_relu_1x(nb_filter, kernel_size, subsample, dimension,batchnorm = True, activation = False)(x)

        final = Add()([input_layer, x])

        return final

    return f


def main_model(output_dim, nb_class, nb_filter, nb_resblock, layer_name = None, subsample = (1,), dimension = 3):
    UpSampling = up_sampling_dict[dimension]
    kernel_size = (3,) * dimension

    def f(input_layer):

        blocks = []

        # upsampling:
        up_layer = UpSampling(size = (1,1,5))(input_layer)
        # print('up_layer dimension: ',up_layer.shape)

        # conv
        blocks += [conv_bn_relu_1x(nb_filter, kernel_size, subsample = (1,), dimension = 3, batchnorm = False, activation = True)(up_layer)]
        # print('first conv layer dimension: ', blocks[0].shape)

        # resblocks
        for i in range(0,nb_resblock):
            blocks  += [res_block(nb_filter, kernel_size, subsample = (1,), dimension = 3)(blocks[-1])]
            
        
        # print('last resblock dimension: ',blocks[-1].shape)

        # last conv:
        blocks += [conv_bn_relu_1x(nb_filter, kernel_size, subsample = (1,), dimension = 3, batchnorm = True, activation = False)(blocks[-1])]
        # print('the second last conv before deep concatenate is: ',blocks[-1].shape)
        
        # deep add
        add = Add()([blocks[0], blocks[-1]])
        # print('deep add shape: ',add.shape)

        # last conv
        final_feature = conv_bn_relu_1x(nb_class, kernel_size, subsample, dimension,batchnorm = False, activation = False)(add)
        # print('final feature.shape: ',final_feature.shape)

        final_feature = Reshape((np.product(output_dim), nb_class))(final_feature)
        final_feature = Activation(activation="softmax")(final_feature)

        final_image = Reshape(output_dim + (nb_class,), name = layer_name)(final_feature)
        # print('final_image shape: ',final_image.shape)

        return final_image

    return f




def learning_rate_step_decay(epoch, lr, step=cg.lr_epochs, initial_power=cg.initial_power):
    """
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every `step` epochs.
    """
    ##
    num = epoch // step
    lrate = 10 ** (initial_power - num)
    print("Learning rate plan for epoch {} is {}.".format(epoch + 1, 1.0 * lrate))
    return np.float(lrate)


def learning_rate_step_decay2(epoch, lr, step=cg.lr_epochs, initial_power=cg.initial_power):
    """
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every `step` epochs.
    """
    ##
    num = epoch // step
    lrate = 10 ** (initial_power - num / 2)
    print("Learning rate plan for epoch {} is {}.".format(epoch + 1, 1.0 * lrate))
    return np.float(lrate)

def learning_rate_step_decay_classic(epoch, lr, decay = 0.01, initial_power=cg.initial_power, start_epoch = cg.start_epoch):
    """
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every `step` epochs.
    """
    ##

    lrate = (1/ (1 + decay * (epoch + start_epoch))) * (10 ** initial_power)

    print("Learning rate plan for epoch {} is {}.".format(epoch + 1, 1.0 * lrate))
    return np.float(lrate)
