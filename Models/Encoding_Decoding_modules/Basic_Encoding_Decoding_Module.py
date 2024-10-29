from DL_utils.utils import *
from DL_utils import flatten_utils
from DL_utils.flatten_utils import s_view,hyb_view#,enc_dim_DNN_type,dec_dim_CNN_type
from DL_utils.ResNET import *
from DL_utils.PCNN import *
from DL_utils.DNN import *
from DL_utils.DNN import b_encoder_NN as DNN_ENC
from DL_utils.DNN import b_decoder_NN as DNN_DEC
from DL_utils.CNN import b_encoder_conv as CNN_ENC
from DL_utils.CNN import b_decoder_conv as CNN_DEC

class EncodingDecodingModule(nn.Module):
  def __init__(self):
    super(EncodingDecodingModule,self).__init__()
    #Encoding modules
    self.el1=set_conv(1, 2, kernel_size=5, act=nn.ReLU(), pooling=False, batch_norm=False, dropout=None, stride=1)
    self.el2=set_conv(2, 4, kernel_size=5, act=nn.ReLU(), pooling=False, batch_norm=False, dropout=None, stride=1)
    #Decoding modules
    self.dl1=set_deconv(4, 2, kernel_size=5, act=nn.ReLU(), pooling=False, batch_norm=False, dropout=None, stride=1)
    self.dl2=set_deconv(2, 1, kernel_size=5, act=nn.ReLU(), pooling=False, batch_norm=False, dropout=None, stride=1)
    #flatten
    self.fl=s_view()

  def Encoding(self,x):
    ex=self.el1(x)
    ex=self.el2(ex)
    ex=self.fl(ex)
    return ex
    
  def Decoding(self,z):
    dz=self.fl(z)
    dz=self.dl1(dz)
    dz=self.dl2(dz)
    return dz


class Asymmetrical_Dense_Neural_Net_EDM(nn.Module):
  def __init__(self,encoder_parameters,decoder_parameters,flat=True):
    super(Asymmetrical_Dense_Neural_Net_EDM,self).__init__()
    #Encoding modules
    self.flat=flat
    self.ENC=b_encoder_NN(**encoder_parameters
                   )
    #Decoding modules
    self.DEC=b_decoder_NN(**decoder_parameters
                   )
    #flatten
    self.fl=s_view()

  def sanity_check(self,x):
    ex=self.ENC(x)
    ex=self.fl(ex).shape
    return ex

  def Encoding(self,x):
    ex=self.ENC(x)
    if self.flat:
      ex=self.fl(ex)
    return ex
    
  def Decoding(self,z):
    dz=z
    if self.flat:
      dz=self.fl(z)
    dz=self.DEC(dz)
    return dz

class Basic_Convolutional_EDM(nn.Module):
  def __init__(self,repr_sizes=[3,32,64,128,256],kernel_size=5,activators=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1,flat=True):
    super(Basic_Convolutional_EDM,self).__init__()
    #Encoding modules
    self.flat=flat
    self.ENC=b_encoder_conv(
                   repr_sizes=repr_sizes,
                   kernel_size=kernel_size,
                   activators=activators,
                   batch_norm=batch_norm,
                   dropout=dropout,
                   stride=stride,
                   pooling=pooling
                   )
    #Decoding modules
    self.DEC=b_decoder_conv(
                   repr_sizes=repr_sizes,
                   kernel_size=kernel_size,
                   activators=activators,
                   batch_norm=batch_norm,
                   dropout=dropout,
                   stride=stride,
                   pooling=pooling
                   )
    #flatten
    self.fl=s_view()

  def sanity_check(self,x):
    ex=self.ENC(x)
    ex=self.fl(ex).shape
    return ex

  def Encoding(self,x):
    ex=self.ENC(x)
    if self.flat:
      ex=self.fl(ex)
    return ex
    
  def Decoding(self,z):
    if self.flat:
      dz=self.fl(z)
    dz=self.DEC(dz)
    return dz
  
class Asymmetrical_Convolutional_EDM(nn.Module):
  def __init__(self,encoder_parameters,decoder_parameters,flat=True):
    super(Asymmetrical_Convolutional_EDM,self).__init__()
    #Encoding modules
    self.flat=flat
    self.ENC=b_encoder_conv(**encoder_parameters
                   )
    #Decoding modules
    self.DEC=b_decoder_conv(**decoder_parameters
                   )
    #flatten
    self.fl=s_view()

  def sanity_check(self,x):
    ex=self.ENC(x)
    ex=self.fl(ex).shape
    return ex

  def Encoding(self,x):
    ex=self.ENC(x)
    if self.flat:
      ex=self.fl(ex)
    return ex
    
  def Decoding(self,z):
    if self.flat:
      dz=self.fl(z)
    dz=self.DEC(dz)
    return dz
  

class Asymmetrical_CNN_DNN_EDM(nn.Module):
  def __init__(self,encoder_parameters,decoder_parameters,Enc_type="DNN",Dec_type="CNN",compression_factor=1,flat=True,deflat=True,i_shape=[28,28]):
    super(Asymmetrical_CNN_DNN_EDM,self).__init__()
    #Encoding modules
    self.flat=flat
    self.deflat=deflat
    self.ENC=globals()[Enc_type](**encoder_parameters)
    self.DEC=globals()[Dec_type](**decoder_parameters)
    #flatten
    self.compression_factor=compression_factor
    self.fl=hyb_view(i_shape,
                     compression_factor,
                     enc=self.ENC,
                     dec=self.DEC,
                     enc_dim_func=getattr(flatten_utils,"dim_{}_type".format(Enc_type)),
                     dec_dim_func=getattr(flatten_utils,"dim_{}_type".format(Dec_type))
                     )

  def sanity_check(self,x):
    ex=self.ENC(x)
    ex=self.fl(ex).shape
    return ex

  def Encoding(self,x):
    #self.fl=hyb_view(list(x.shape),cf=self.compression_factor)
    ex=self.ENC(x)
    if self.flat:
      ex=self.fl(ex)
    return ex
    
  def Decoding(self,z):
    if self.deflat:
      z=self.fl(z)
    z=self.DEC(z)
    return z  


from DL_utils.PCNN import *

class MaxPooling_Convolutional_EDM(nn.Module):
  def __init__(self,repr_sizes=[3,32,64,128,256],kernel_size=5,activators=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1,flat=True):
    super(MaxPooling_Convolutional_EDM,self).__init__()
    #Encoding modules
    self.flat=flat
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

  def sanity_check(self,x):
    ex=self.ENC(x)
    ex=self.fl(ex).shape
    return ex

  def Encoding(self,x):
    ex=self.ENC(x)
    if self.flat:
      ex=self.fl(ex)
    return ex
    
  def Decoding(self,z):
    dz=z
    if self.flat:
      dz=self.fl(z)
    dz=self.DEC(dz,self.ENC.idx_list[::-1],self.ENC.Sidx_list[::-1])
    return dz


class ResNET_Convolutional_EDM(nn.Module):
  #def __init__(self,repr_sizes=[3,32,64,128,256],kernel_size=5,activators=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1):
  def __init__(self,repr_sizes=[[1,2,3],[3,4,5]],kernel_sizes=[[3,5],[5,5]],bridge_kernel_size=3,act=nn.ReLU(),bridge_act=nn.ReLU(),lay_act=nn.ReLU(),batch_norm=True,dropout=None,stride=[[1,1],[1,2]],flat=True):
    super(ResNET_Convolutional_EDM,self).__init__()
    #Encoding modules
    self.flat=flat
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
    if self.flat:
      ex=self.fl(ex)
    return ex
    
  def Decoding(self,z):
    if self.flat:
      dz=self.fl(z)
    dz=self.DEC(dz)
    return dz