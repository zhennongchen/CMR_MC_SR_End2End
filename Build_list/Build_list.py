import numpy as np
import os
import pandas as pd


class Build():
    def __init__(self,file_list):
        
        self.file_list = file_list
        self.data = pd.read_excel(file_list)

    def __build__(self,batch_list):
        for b in range(len(batch_list)):
            cases = self.data.loc[self.data['batch'] == batch_list[b]]
            if b == 0:
                c = cases.copy()
            else:
                c = pd.concat([c,cases])

        
        patient_id_list = np.asarray(c['Patient_ID'])
        patient_tf_list = np.asarray(c['tf'])
        motion_name_list= np.asarray(c['motion_name'])
        batch_list = np.asarray(c['batch'])

        HRfile_list = np.asarray(c['GroundTruthHR'])
        LRfile_list = np.asarray(c['GroundTruthLR'])

        gt_center_list = np.asarray(c['GroundTruth_CenterPoints'])
        motion_center_list = np.asarray(c['Motion_CenterPoints'])


        motion_file_list = np.asarray(c['MotionFile'])

        return patient_id_list, patient_tf_list, motion_name_list, batch_list, HRfile_list,  LRfile_list,  gt_center_list,  motion_file_list,motion_center_list
