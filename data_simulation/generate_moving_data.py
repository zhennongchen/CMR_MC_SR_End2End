
import numpy as np
import math
import nibabel as nb
import CMR_HFpEF_Analysis.data_simulation.transformation as transform
import CMR_HFpEF_Analysis.functions_collection as ff
import CMR_HFpEF_Analysis.Image_utils as util

## generate movign data with preset translation range
def generate_moving_data(img_ds, t_mu, t_sigma, t_bar, r_amplitude,  spacing, order, extreme = True):
    img_new = np.zeros((img_ds.shape))
    record = []
    for i in range(0,img_ds.shape[-1]):
        I = img_ds[:,:,i]
        if np.sum(I) == 0: # no heart in this slice
            record.append([i, 0, [0,0] , [0,0], 0])
            continue

        if np.where(I == 1)[0].shape[0] == 0 and i > 3: # only do the translation if there is LV in the most basal slice
            record.append([i, 0, [0,0] , [0,0], 0])
            continue
        
        final_t_mm = transform.random_t(t_mu,t_sigma,I,t_bar)
        final_t = []
        for j in range(0,len(final_t_mm)):
            if final_t_mm[j] < 0:
                if extreme == True:
                    final_t.append(math.floor(final_t_mm[j] / spacing[j]))
                else:
                    final_t.append(final_t_mm[j] / spacing[j])
            else:
                if extreme == True:
                    final_t.append(math.ceil(final_t_mm[j] / spacing[j]))
                else:
                    final_t.append(final_t_mm[j] / spacing[j])

        total_t_mm = math.sqrt(np.sum(np.square(np.asarray(final_t_mm))))

        r = transform.random_r(r_amplitude)
        # print('slice ', i, final_t_mm, total_t_mm,final_t, r)

        center_mass = util.center_of_mass(I,1,large = True)
        center_mass = [int(center_mass[0]),int(center_mass[1])]

        translation,rotation,scale,M = transform.generate_transform_matrix(final_t,r/ 180 * np.pi,[1,1],I.shape)
        M = transform.transform_full_matrix_offset_heart(M, center_mass)
        II = transform.apply_affine_transform(I, M, order)
        img_new[:,:,i] = II
        record.append([i, total_t_mm, final_t_mm, final_t,abs(r)])

    return img_new, record


# generate nx moving data
def generate_moving_data_nx(img_ds, nx_for_t, nx_for_r, previous_record, spacing, order, extreme = True):
    img_nx = np.zeros((img_ds.shape))
    record_nx = []
    for i in range(0,img_ds.shape[-1]):
        I = img_ds[:,:,i]

        # final_t = [previous_record[i][3][0] * nx_for_t, previous_record[i][3][1] * nx_for_t]
        # final_t_mm = [final_t[j] * spacing[j] for j in range(0,len(final_t))]
        # total_t_mm = math.sqrt(np.sum(np.square(np.asarray(final_t_mm))))

        final_t_mm = [previous_record[i][2][0] * nx_for_t, previous_record[i][2][1] * nx_for_t]
        total_t_mm = math.sqrt(np.sum(np.square(np.asarray(final_t_mm))))
        final_t = []
        for j in range(0,len(final_t_mm)):
            if final_t_mm[j] < 0:
                if extreme == True:
                    final_t.append(math.floor(final_t_mm[j] / spacing[j]))
                else:
                    final_t.append(final_t_mm[j] / spacing[j])
            else:
                if extreme == True:
                    final_t.append(math.ceil(final_t_mm[j] / spacing[j]))
                else:
                    final_t.append(final_t_mm[j] / spacing[j])

        r = previous_record[i][4] * nx_for_r
        # print('slice ', i, final_t_mm, total_t_mm,final_t, r)

        if total_t_mm != 0:
            center_mass = util.center_of_mass(I,0,large = True);center_mass = [int(center_mass[0]),int(center_mass[1])]

            translation,rotation,scale,M = transform.generate_transform_matrix(final_t,r/ 180 * np.pi,[1,1],I.shape)
            M = transform.transform_full_matrix_offset_heart(M, center_mass)
            II = transform.apply_affine_transform(I, M, order)
            img_nx[:,:,i] = II
        record_nx.append([i, total_t_mm, final_t_mm,  final_t, abs(r)])

    return img_nx, record_nx