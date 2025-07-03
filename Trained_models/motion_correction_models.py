import os 
import numpy as np
import CMR_HFpEF_Analysis.Defaults as Defaults

cg = Defaults.Parameters()

class trained_models():
    def __init__(self):
        self.main = cg.model_dir

    def Motion_ResNet_collection(self):
        models = [os.path.join(self.main, 'Motion_ResNet_new/models/batch_0/model-017.hdf5'),
                  os.path.join(self.main, 'Motion_ResNet_new/models/batch_0/model-023.hdf5'),
                  os.path.join(self.main, 'Motion_ResNet_new/models/batch_0/model-071.hdf5'),
                  os.path.join(self.main, 'Motion_ResNet_new/models/batch_1/model-014.hdf5'),
                  os.path.join(self.main, 'Motion_ResNet_new/models/batch_1/model-048.hdf5'),
                  os.path.join(self.main, 'Motion_ResNet_new/models/batch_1/model-076.hdf5'),
                  os.path.join(self.main, 'Motion_ResNet_new/models/batch_2/model-015.hdf5'),
                  os.path.join(self.main, 'Motion_ResNet_new/models/batch_2/model-023.hdf5'),
                  os.path.join(self.main, 'Motion_ResNet_new/models/batch_2/model-080.hdf5'),]
        return models


    def Motion_ResNet_HR_collection(self):
        center_x_models = [os.path.join(self.main, 'Motion_HR_1/models/batch_0/model-056.hdf5'),
                  os.path.join(self.main, 'Motion_HR_1/models/batch_1/model-053.hdf5'),
                #   os.path.join(self.main, 'Motion_HR_2/models/batch_0/model-021.hdf5'),
                  os.path.join(self.main, 'Motion_HR_1/models/batch_2/model-041.hdf5'),
                  os.path.join(self.main, 'Motion_HR_1/models/batch_3/model-044.hdf5'),
                  os.path.join(self.main, 'Motion_HR_1/models/batch_4/model-043.hdf5'),]

        center_y_models = [os.path.join(self.main, 'Motion_HR_1/models/batch_0/model-045.hdf5'),
                  os.path.join(self.main, 'Motion_HR_1/models/batch_1/model-048.hdf5'),
                #   os.path.join(self.main, 'Motion_HR_2/models/batch_0/model-015.hdf5'),
                  os.path.join(self.main, 'Motion_HR_1/models/batch_2/model-042.hdf5'),
                  os.path.join(self.main, 'Motion_HR_1/models/batch_3/model-039.hdf5'),
                  os.path.join(self.main, 'Motion_HR_1/models/batch_4/model-026.hdf5'),]

        return center_x_models, center_y_models

    def Motion_ResNet_HR_index(self,batch):
        if batch < 5:
            i = [batch, batch + 5]
        elif batch == 5:
            i = [0, 1, 2 ]
        else:
            ValueError('wrong batch num')
        return i
    

        # def Motion_ResNet_collection(self):
    #     center_x_models = [os.path.join(self.main, 'Motion_ResNet/models/batch_0/model-027.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet/models/batch_1/model-041.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet/models/batch_2/model-069.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet/models/batch_3/model-033.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet/models/batch_4/model-018.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet_alternative/models/batch_0/model-036.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet_alternative/models/batch_1/model-040.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet_alternative/models/batch_2/model-044.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet_alternative/models/batch_3/model-034.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet_alternative/models/batch_4/model-049.hdf5')]

    #     center_y_models = [os.path.join(self.main, 'Motion_ResNet/models/batch_0/model-027.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet/models/batch_1/model-039.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet/models/batch_2/model-040.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet/models/batch_3/model-022.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet/models/batch_4/model-035.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet_alternative/models/batch_0/model-033.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet_alternative/models/batch_1/model-038.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet_alternative/models/batch_2/model-045.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet_alternative/models/batch_3/model-030.hdf5'),
    #               os.path.join(self.main, 'Motion_ResNet_alternative/models/batch_4/model-021.hdf5')]

    #     return center_x_models, center_y_models

    
    # def Motion_LR_5CP(self):
    #     center_x_models = [os.path.join(self.main, 'Motion_LR_5CP/models/batch_0/model-104.hdf5'),]
    #     center_y_models = [os.path.join(self.main, 'Motion_LR_5CP/models/batch_0/model-160.hdf5'),]
    #     return center_x_models, center_y_models

    # def Motion_LR_6CP(self):
    #     center_x_models = [os.path.join(self.main, 'Motion_LR_6CP/models/batch_0/model-054.hdf5'),]
    #     center_y_models = [os.path.join(self.main, 'Motion_LR_6CP/models/batch_0/model-059.hdf5'),]
    #     return center_x_models, center_y_models

    def Motion_LR_8CP(self):
        center_x_models = [os.path.join(self.main, 'Motion_LR_8CP/models/batch_0/model-045.hdf5'),]
        center_y_models = [os.path.join(self.main, 'Motion_LR_8CP/models/batch_0/model-033.hdf5'),]
        return center_x_models, center_y_models