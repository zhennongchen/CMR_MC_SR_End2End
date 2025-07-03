import numpy as np
import matplotlib.pyplot as plt
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.data_simulation.transformation as transform
import CMR_HFpEF_Analysis.Image_utils as util

# do random displacement
def displacement_generator(max = 5):
    tx_max = int(max)#int(img.shape[0] * percent)
    ty_max = int(max)#int(img.shape[1] * percent)
    # lv_slice = [i for i in range(0,img.shape[-1]) if np.where(img[:,:,i] == 1)[0].shape[0] != 0]
    tz_max = 0 #np.min(np.array([lv_slice[0], 11 - lv_slice[-1]]))

    tx_aug = np.round(np.random.rand() * tx_max * 2,0) - tx_max
    ty_aug = np.round(np.random.rand() * ty_max * 2,0) - ty_max
    tz_aug = 0 #np.round(np.random.rand() * tz_max * 2,0) - tz_max

    return [tx_aug, ty_aug, tz_aug], tx_max, ty_max, tz_max

def augmentation(img, t):

    center_mass = util.center_of_mass(img,0,large = True);center_mass = [int(center_mass[0]),int(center_mass[1]), int(center_mass[2])]
    translation,rotation,scale,M = transform.generate_transform_matrix(t,[0,0,0],[1,1,1],img.shape)
    M = transform.transform_full_matrix_offset_heart(M, center_mass)
    img_new = transform.apply_affine_transform(img, M, order = 0)

    return img_new,t