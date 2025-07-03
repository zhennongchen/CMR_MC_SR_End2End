import numpy as np
import os
import pandas as pd


class Build():
    def __init__(self,file_list):
        
        self.file_list = file_list
        self.data = pd.read_excel(file_list)

    def __build__(self):

        input_list = np.asarray(self.data['InputFile'])
        patient_id_list = np.asarray(self.data['Patient_ID'])
        patient_tf_list = np.asarray(self.data['tf'])

        return input_list, patient_id_list, patient_tf_list
