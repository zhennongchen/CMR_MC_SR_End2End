import os 
import numpy as np
import CMR_HFpEF_Analysis.Defaults as Defaults

cg = Defaults.Parameters()

class trained_models():
    def __init__(self):
        self.main = cg.model_dir

    def end2end_collection(self):
        models = [os.path.join(self.main, 'end2end/models/batch_0/model-022.hdf5'),
                  os.path.join(self.main, 'end2end/models/batch_1/model-022.hdf5'),
                  os.path.join(self.main, 'end2end/models/batch_2/model-058.hdf5'),]
        return models