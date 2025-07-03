import os 
import numpy as np
import CMR_HFpEF_Analysis.Defaults as Defaults

cg = Defaults.Parameters()

class trained_models():
    def __init__(self):
        self.main = cg.model_dir
    
    def STN_vector_img_models(self):
        models = [os.path.join(self.main, 'STN_vector_img/models/batch_0/model-056.hdf5'),
                  os.path.join(self.main, 'STN_vector_img/models/batch_1/model-077.hdf5'),
                  os.path.join(self.main, 'STN_vector_img/models/batch_2/model-068.hdf5')
                  ]
        return models
    
    def STN_vector_models(self):
        models = [os.path.join(self.main, 'STN_vector/models/batch_0/model-052.hdf5'),]
        return models