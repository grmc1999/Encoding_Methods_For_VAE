
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy

# Load model
# Predict batch plus softmax # [batch channel width height] -> [batch classes=1000]
# take set of batches # [splits*batch classes=1000]
# take mean of splits # [splits*batch classes=1000]
# for each part take entropy between mean and ith part





class inception_score(object):
    def __init__(self,batch_size,split):
        dtype=float
        # Load model
        self.model=inception_v3(pretrained=True,transform_input=False).type(dtype).eval()
        self.resize=nn.Upsample((299,299),mode='bilinear').type(dtype)
        self.batch_size=batch_size
        self.split=split

    def predict(self,x):
        x=self.up(x)
        return F.softmax(self.model(x)).data.cpu().numpy()

    def predict_batch(self,batch):
        b_x=self.predict(batch)
        return b_x

    def take_splits(self,batches,splits):
        scores=[]
        for split in range(batches//splits):
            batch_split=batches[split:split+1]
            batch_split_mean=np.mean(batch_split,axis=0)

            for batch in batches:
                scores.append(entropy(batch_split_mean,batch))
        
        return scores