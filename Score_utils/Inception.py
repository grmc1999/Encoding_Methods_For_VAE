
import torch
from torchvision.models.inception import inception_v3

# Load model
# Predict batch plus softmax # [batch channel width height] -> [batch classes=1000]
# take set of batches # [splits*batch classes=1000]
# take mean of splits # [splits*batch classes=1000]
# for each part take entropy between mean and ith part





class inception_score(object):
    def __init__(self,batch_size,split):
        # Load model
        self.batch_size=batch_size
        self.split=split

    def predict(self,x):

    def predict_batch(self,batch):