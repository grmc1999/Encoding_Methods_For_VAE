from torchvision import transforms
import torch

class Tuple_to_dict(object):

    def __init__(self):
        self.new_sample={}
        self.flag=0

    def __call__(self, sample):
        
        if self.flag==0:
            self.new_sample["x"]=sample
            self.flag=self.flag+1
        else:
            self.new_sample["y"]=sample
            self.flag=0

        return self.new_sample

class No_target(object):

    def __init__(self):
        pass

    def __call__(self, sample):

        return None

class MultiInputToTensor(object):
  def __init__(self,images=["x"],metadata=["y"]):
    self.images=images
    self.metadata=metadata
    self.TT=transforms.ToTensor()
  def __call__(self,sample):
    for k in self.images:
      sample[k]=(self.TT(sample[k])).float()
    for k in self.metadata:
      sample[k]=torch.tensor(sample[k]).float()
    return sample

class Size_Normalization(object):
  def __init__(self,images=["x"],metadata=["y"],normalized_size=(512,512)):
    self.images=images
    self.metadata=metadata
    self.normalized_size=transforms.Resize(size=normalized_size,interpolation=transforms.InterpolationMode.BILINEAR)
  def __call__(self,sample):
    for k in self.images:
      sample[k]=(self.normalized_size(sample[k])).float()
    return sample