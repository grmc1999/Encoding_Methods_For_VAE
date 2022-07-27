from DL_utils.utils import *
from DL_utils.PCNN import *

class MaxPooling_Convolutional_EDM(nn.Module):
  def __init__(self,repr_sizes=[3,32,64,128,256],kernel_size=5,activators=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1):
    super(MaxPooling_Convolutional_EDM,self).__init__()
    #Encoding modules
    self.ENC=Max_Pool_encoder_conv(
        repr_sizes=repr_sizes,
        kernel_size=kernel_size,
        activators=activators,
        pool=pooling,
        batch_norm=batch_norm,
        dropout=dropout,
        stride=stride
    )
    #Encoding modules
    self.DEC=Max_Unpool_decoder_conv(
        repr_sizes=repr_sizes,
        kernel_size=kernel_size,
        activators=activators,
        pool=pooling,
        batch_norm=batch_norm,
        dropout=dropout,
        stride=stride
    )
    #flatten
    self.fl=s_view()

#ci=MPEC(torch.rand((1,15,15)))
#print(ci.shape)
#i=MUEC(ci,MPEC.idx_list[::-1])
#print(di.shape)

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
    dz=self.DEC(dz,self.ENC.idx_list[::-1],self.ENC.Sidx_list[::-1])
    return dz
