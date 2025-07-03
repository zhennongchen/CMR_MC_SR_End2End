from fcntl import DN_MODIFY
import numpy as np
from numpy import random
import math
from .rotation_matrix_from_angle import rotation_matrix


def generate_transform_matrix(t,r,s,img_shape):

    assert type(img_shape) == tuple
   
    ## translation
    # t should be the translation in [x,y,z] directions
    assert len(t) == len(img_shape)

    translation = np.eye(len(img_shape) + 1)
    translation[:len(img_shape),len(img_shape)] = np.transpose(np.asarray(t))

    ## rotation
    # r should be the rotation angle in [x,y,z] if 3D, or a single scalar value if 2D
    if len(img_shape) == 2:
        assert type(r) == float
    if len(img_shape) == 3:
        assert len(r) == 3
    
    rotation = np.eye(len(img_shape) + 1)

    if len(img_shape) == 2:

        rotation[:2, :2] = rotation_matrix(r)

    elif len(img_shape) == 3:
        x = rotation_matrix(r[0],matrix_type="roll")
        y = rotation_matrix(r[1],matrix_type="pitch")
        z = rotation_matrix(r[2],matrix_type="yaw")
        rotation[:3, :3] = np.dot(z, np.dot(y, x))
    else:
        raise Exception("image_dimension must be either 2 or 3.")

    # scale
    scale = np.eye(len(img_shape) + 1)
    for ax in range(0,len(img_shape)):
        scale[ax, ax] = s[ax]

    return translation,rotation,scale,np.dot(scale, np.dot(rotation, translation))

def random_t(t_mu,t_sigma,img,bar):
    ndim = len(img.shape)

    while True:
        total_t = np.random.normal(t_mu,t_sigma)
        total_t = abs(total_t)
    
        if total_t <= bar: 
             break

    # randomly divide into x and y (and z)
    final_t = divide_t_into_directions(total_t, ndim)
    
    return final_t
        
        
def divide_t_into_directions(total_t, ndim):
    proportion = []
    for i in range(0,ndim):
        proportion.append(random.rand())
    # final t
    final_t = []
    for i in range(0,ndim):
        tt = math.sqrt(total_t**2 / sum(proportion) * proportion[i])
        a = random.rand()
        if a >=0.5:
            tt = -tt
        final_t.append(tt)
    
    return final_t


def random_r(amplitude):
    r = np.random.rand() * amplitude
    sign = np.random.rand()
    if sign > 0.5:
        r = -r
    return r
