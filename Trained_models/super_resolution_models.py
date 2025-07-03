import os 
import numpy as np
import CMR_HFpEF_Analysis.Defaults as Defaults

cg = Defaults.Parameters()

class trained_models():
    def __init__(self):
        self.main = cg.model_dir
    
    def EDSR_super_resolution(self):
        
        model = [os.path.join(self.main, 'EDSR_LVmyo_ds_new/models/batch_0/model-021.hdf5'),
                 os.path.join(self.main, 'EDSR_LVmyo_ds_new/models/batch_1/model-193.hdf5'),]
        # lv_model = os.path.join(self.main, 'EDSR_LV_ds_dataversion2_b/models/batch_0/model-057.hdf5')  
        # lvmyo_model = os.path.join(self.main, 'EDSR_LVmyo_ds_dataversion2_b/models/batch_0/model-098.hdf5')
        # threeclass_model = os.path.join(self.main, 'EDSR_3class_ds_dataversion2_b/models/batch_0/model-066.hdf5')

        return model
    
    def EDSR_two_tasks(self):
        # motion correction + super resolution
        model = [os.path.join(self.main, 'EDSR_LVmyo_motion_new/models/batch_0/model-033.hdf5'),
                 os.path.join(self.main, 'EDSR_LVmyo_motion_new/models/batch_1/model-030.hdf5'),]
        return model


    # def EDSR_collection_ds_slice_aug(self):

    #     lv_model = os.path.join(self.main, 'EDSR_LV_ds_dataversion2/models/batch_0/model-072.hdf5')
                  
    #     lvmyo_model = os.path.join(self.main, 'EDSR_LVmyo_ds_dataversion2/models/batch_0/model-052.hdf5')

    #     threeclass_model = os.path.join(self.main, 'EDSR_3class_ds_dataversion2/models/batch_0/model-064.hdf5')

    #     return lv_model, lvmyo_model,threeclass_model

    # def Motion_ResNet_index(self,batch):
    #     if batch < 5:
    #         i = [batch]
    #     elif batch == 5:
    #         i = [0, 1, 2, 3, 4]
    #     else:
    #         ValueError('wrong batch num')
    #     return i