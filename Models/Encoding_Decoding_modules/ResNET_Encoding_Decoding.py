from DL_utils.utils import *
from DL_utils.ResNET import *

class ResNET_Convolutional_EDM(nn.Module):
  #def __init__(self,repr_sizes=[3,32,64,128,256],kernel_size=5,activators=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1):
  def __init__(self,repr_sizes=[[1,2,3],[3,4,5]],kernel_sizes=[[3,5],[5,5]],bridge_kernel_size=3,act=nn.ReLU(),bridge_act=nn.ReLU(),lay_act=nn.ReLU(),batch_norm=True,dropout=None,stride=[[1,1],[1,2]]):
    super(ResNET_Convolutional_EDM,self).__init__()
    #Encoding modules
    self.ENC=ResNET_ENC(
        repr_sizes=repr_sizes,
        kernel_sizes=kernel_sizes,
        bridge_kernel_size=bridge_kernel_size,
        act=act,
        bridge_act=bridge_act,
        lay_act=lay_act,
        batch_norm=batch_norm,
        dropout=dropout,
        stride=stride
    )
    #Encoding modules
    self.DEC=ResNET_DEC(
        repr_sizes=repr_sizes,
        kernel_sizes=kernel_sizes,
        bridge_kernel_size=bridge_kernel_size,
        act=act,
        bridge_act=bridge_act,
        lay_act=lay_act,
        batch_norm=batch_norm,
        dropout=dropout,
        stride=stride
    )
    #flatten
    self.fl=s_view()

  def sanity_check(self,x):
    ex=self.ENC(x)
    ex=self.fl(ex).shape
    return ex

  def Encoding(self,x):
    ex=self.ENC(x)
    ex=self.fl(ex)
    return ex
    
  def Decoding(self,z):
    dz=self.fl(z)
    dz=self.DEC(dz)
    return dz