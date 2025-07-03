
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras.layers import Layer



class element_1D(Layer):
    def __init__(self,index):
        super(element_1D, self).__init__()
        self.index = index

    def get_config(self):
        config = super().get_config().copy()
        config.update({'index' : self.index, })
        return config
    
    def call(self, inputs):
        output = inputs[...,self.index]; output = tf.reshape(output,[-1,1])
        return output



def learning_rate_step_decay(epoch, lr, step=25, initial_power=-4):
    """
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every `step` epochs.
    """
    ##
    num = epoch // step
    lrate = 10 ** (initial_power - num)
    print("Learning rate plan for epoch {} is {}.".format(epoch + 1, 1.0 * lrate))
    return np.float(lrate)

def learning_rate_step_decay2(epoch, lr, step=25, initial_power=-4):
    """
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every `step` epochs.
    """
    ##
    num = epoch // step
    lrate = 10 ** (initial_power - num / 2)
    print("Learning rate plan for epoch {} is {}.".format(epoch + 1, 1.0 * lrate))
    return np.float(lrate)


def learning_rate_step_decay_classic(epoch, lr, decay = 0.01, initial_power=-4, start_epoch = 0):
    """
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every `step` epochs.
    """
    ##

    lrate = (1/ (1 + decay * (epoch + start_epoch))) * (10 ** initial_power)

    print("Learning rate plan for epoch {} is {}.".format(epoch + 1, 1.0 * lrate))
    return np.float(lrate)

def custom_mae_loss(y_true, y_pred):
    '''if a row has [0,0], doesn't take it into consideration'''
    # Create a mask for rows with non-zero values in y_true
    mask = tf.math.reduce_any(y_true[:, :, 0:] != 0, axis=-1)
    
    # Apply the mask to y_true and y_pred
    masked_y_true = tf.boolean_mask(y_true, mask)
    masked_y_pred = tf.boolean_mask(y_pred, mask)
    
    # Calculate the absolute difference
    absolute_diff = tf.abs(masked_y_true - masked_y_pred)
    
    # Calculate the mean of the absolute difference
    loss = tf.reduce_mean(absolute_diff)
    
    return loss

def dice_coefficient_calculation(y_true, y_pred, class_value):
    class_mask_true = tf.cast(tf.equal(y_true, class_value), dtype=tf.float32)
    class_mask_pred = tf.cast(tf.equal(y_pred, class_value), dtype=tf.float32)

    intersection = tf.reduce_sum(class_mask_true * class_mask_pred)
    union = tf.reduce_sum(class_mask_true) + tf.reduce_sum(class_mask_pred)
    return (2.0 * intersection + 1e-5) / (union + 1e-5)


def dice_loss_two_classes(y_true, y_pred):
    dice_class_1  = dice_coefficient_calculation(y_true, y_pred, class_value = 1)
    dice_class_2 = dice_coefficient_calculation(y_true, y_pred, class_value = 2)

    return 1.0 - (dice_class_1 + dice_class_2) / 2.0


def dice_loss_one_class(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
    return 1.0 - dice
