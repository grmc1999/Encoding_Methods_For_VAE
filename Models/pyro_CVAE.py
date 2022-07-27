import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import pyro
import pyro.distributions as dist
#from pyro.infer import SVI,Trace_ELBO
#from pyro.optim import Adam

from .DL_utils.utils import NeuralNet
from .DL_utils.Generative_autoencoder_utils import Base_Generative_AutoEncoder

class Relaxed_CVAE(NeuralNet):
  """
  prior relaxed without baseline
  to generate normal distributions
  """
  def __init__(self,Encoder_Decoder_Module,y_layer_size,y_enc_activators,x_layer_size,x_enc_activators,save_output=False,aux_dir=None,module_name=None):
    super(Relaxed_CVAE,self).__init__(layer_size=[],activators=[],save_output=save_output,aux_dir=aux_dir,module_name=module_name)
    self.EncoderDecoder=Encoder_Decoder_Module

    self.y_xz_mu=NeuralNet(y_layer_size,y_enc_activators,save_output,aux_dir,module_name)
    self.y_xz_sig=NeuralNet(y_layer_size,y_enc_activators,save_output,aux_dir,module_name)

    self.z_xy_mu=NeuralNet(x_layer_size,x_enc_activators,save_output,aux_dir,module_name)
    self.z_xy_sig=NeuralNet(x_layer_size,x_enc_activators,save_output,aux_dir,module_name)
    self.z_dim=x_layer_size[-1]

  def model(self,x_obs,y_obs,In_ID=None):
    pyro.module("y_xz_mu_NET",self.y_xz_mu)
    pyro.module("y_xz_sig_NET",self.y_xz_sig)
    with pyro.plate("Data",x_obs.shape[0]):
      x=self.pre_net(x_obs,In_ID)
      #z allocation
      z_mu=x.new_zeros(torch.Size((x.shape[0],self.z_dim)))
      z_sig=x.new_ones(torch.Size((x.shape[0],self.z_dim)))
      z=pyro.sample("z",dist.Normal(z_mu,z_sig).to_event(1))
      #post z representation z_dec=self.post_z()
      #xz=torch.cat((x,z_dec),dim=1)
      xz=torch.cat((x,z),dim=1)
      y_mu=self.y_xz_mu(xz,In_ID)
      y_sig=torch.exp(self.y_xz_sig(xz,In_ID))
      #y=self.post_net(x,In_ID)
      pyro.sample("y",dist.Normal(y_mu,y_sig).to_event(1),obs=y_obs) #reduce with bernoulli dist
  
  def guide(self,x_obs,y_obs,In_ID=None):
    pyro.module("z_xy_mu_NET",self.z_xy_mu)
    pyro.module("z_xy_sig_NET",self.z_xy_sig)
    with pyro.plate("Data",x_obs.shape[0]):
      x=self.pre_net(x_obs,In_ID)
      xy=torch.cat((x,y_obs.unsqueeze(1)),dim=1)
      z_mu=self.z_xy_mu(xy,In_ID)
      z_sig=torch.exp(self.z_xy_sig(xy,In_ID))
      z=pyro.sample("z",dist.Normal(z_mu,z_sig).to_event(1))
  
  def forward_pass(self,x,In_ID=None):
    xe=self.pre_net(x,In_ID)

    z_mu=x.new_zeros(torch.Size((x.shape[0],self.z_dim)))
    z_sig=x.new_ones(torch.Size((x.shape[0],self.z_dim)))
    z=pyro.sample("z",dist.Normal(z_mu,z_sig).to_event(1))

    #z_mu=self.z_x_mu(xe,In_ID)
    #z_sig=torch.exp(self.z_x_sig(xe,In_ID))
    #z=dist.Normal(z_mu,z_sig).sample()

    #post z representation z_dec=self.post_z()
    #xz=torch.cat((x,z_dec),dim=1)

    xz=torch.cat((xe,z),dim=1)

    #y_r=self.y_xz(xz,In_ID)
    #y_r=self.post_net(y_r,In_ID)
    y_mu=self.y_xz_mu(xz,In_ID)
    y_sig=torch.exp(self.y_xz_sig(xz,In_ID))
    y=dist.Normal(y_mu,y_sig).sample()

    return z_mu,z_sig,y_mu,y_sig,y

class Relaxed_CVAE(NeuralNet):
  """
  prior relaxed without baseline
  """
  def __init__(self,y_layer_size,y_enc_activators,x_layer_size,x_enc_activators,save_output=False,aux_dir=None,module_name=None):
    super(Relaxed_CVAE,self).__init__(layer_size=[],activators=[],save_output=save_output,aux_dir=aux_dir,module_name=module_name)
    self.EncoderDecoder=Encoder_Decoder_Module
    #self.pre_net=NeuralNet(pre_layer_size,[nn.LeakyReLU() for i in range(len(pre_layer_size)-1)],save_output,aux_dir,module_name)
    #self.post_net=NeuralNet(pre_layer_size[::-1],[nn.LeakyReLU() for i in range(len(pre_layer_size)-1)],save_output,aux_dir,module_name)

    self.y_xz_mu=NeuralNet(y_layer_size,y_enc_activators,save_output,aux_dir,module_name)
    self.y_xz_sig=NeuralNet(y_layer_size,y_enc_activators,save_output,aux_dir,module_name)

    self.z_xy_mu=NeuralNet(x_layer_size,x_enc_activators,save_output,aux_dir,module_name)
    self.z_xy_sig=NeuralNet(x_layer_size,x_enc_activators,save_output,aux_dir,module_name)
    self.z_dim=x_layer_size[-1]
    #self.x_z=NeuralNet(layer_size[::-1],dec_activators,save_output,aux_dir,module_name)

  def model(self,x_obs,y_obs,In_ID=None):
    pyro.module("y_xz_mu_NET",self.y_xz_mu)
    pyro.module("y_xz_sig_NET",self.y_xz_sig)
    with pyro.plate("Data",x_obs.shape[0]):
      x=self.pre_net(x_obs,In_ID)
      #z allocation
      z_mu=x.new_zeros(torch.Size((x.shape[0],self.z_dim)))
      z_sig=x.new_ones(torch.Size((x.shape[0],self.z_dim)))
      z=pyro.sample("z",dist.Normal(z_mu,z_sig).to_event(1))
      #post z representation z_dec=self.post_z()
      #xz=torch.cat((x,z_dec),dim=1)
      xz=torch.cat((x,z),dim=1)
      y_mu=self.y_xz_mu(xz,In_ID)
      y_sig=torch.exp(self.y_xz_sig(xz,In_ID))
      #y=self.post_net(x,In_ID)
      pyro.sample("y",dist.Normal(y_mu,y_sig).to_event(1),obs=y_obs) #reduce with bernoulli dist
  
  def guide(self,x_obs,y_obs,In_ID=None):
    pyro.module("z_xy_mu_NET",self.z_xy_mu)
    pyro.module("z_xy_sig_NET",self.z_xy_sig)
    with pyro.plate("Data",x_obs.shape[0]):
      x=self.pre_net(x_obs,In_ID)
      xy=torch.cat((x,y_obs.unsqueeze(1)),dim=1)
      z_mu=self.z_xy_mu(xy,In_ID)
      z_sig=torch.exp(self.z_xy_sig(xy,In_ID))
      z=pyro.sample("z",dist.Normal(z_mu,z_sig).to_event(1))
  
  def forward_pass(self,x,In_ID=None):
    xe=self.pre_net(x,In_ID)

    z_mu=x.new_zeros(torch.Size((x.shape[0],self.z_dim)))
    z_sig=x.new_ones(torch.Size((x.shape[0],self.z_dim)))
    z=pyro.sample("z",dist.Normal(z_mu,z_sig).to_event(1))

    #z_mu=self.z_x_mu(xe,In_ID)
    #z_sig=torch.exp(self.z_x_sig(xe,In_ID))
    #z=dist.Normal(z_mu,z_sig).sample()

    #post z representation z_dec=self.post_z()
    #xz=torch.cat((x,z_dec),dim=1)

    xz=torch.cat((xe,z),dim=1)

    #y_r=self.y_xz(xz,In_ID)
    #y_r=self.post_net(y_r,In_ID)
    y_mu=self.y_xz_mu(xz,In_ID)
    y_sig=torch.exp(self.y_xz_sig(xz,In_ID))
    y=dist.Normal(y_mu,y_sig).sample()

    return z_mu,z_sig,y_mu,y_sig,y

class CVAE(Base_Generative_AutoEncoder):
  """
  without baseline
  """
  def __init__(self,X_Encoder_Decoder_Module,Y_Encoder_Decoder_Module,P_NET,Q_NET,entities=10,losses_weigths={},save_output=False,aux_dir=None,module_name=None):
    super(CVAE,self).__init__(Encoder_Decoder_Module=None,P_NET=P_NET,Q_NET=Q_NET,losses_weigths=losses_weigths)
    self.losses={}
    self.gen_loss_fun=pyro.infer.Trace_ELBO().differentiable_loss

    self.X_EncoderDecoder=X_Encoder_Decoder_Module
    self.Y_EncoderDecoder=Y_Encoder_Decoder_Module
    
    self.entities=entities

  def model(self,x_obs,y_obs):
    pyro.module("z_x_mu_NET",self.P.z_x_mu)
    pyro.module("z_x_sig_NET",self.P.z_x_sig)
    pyro.module("y_xz_NET",self.P.y_xz)
    with pyro.plate("Data",x_obs.shape[0]):
      x=self.X_EncoderDecoder.Encoding(x_obs)
      #z allocation
      z_mu=self.P.z_x_mu(x)
      z_sig=torch.exp(self.P.z_x_sig(x))
      z=pyro.sample("z",dist.Normal(z_mu,z_sig).to_event(1))  #Change z generation in dependent of x
      xz=torch.cat((x,z),dim=1)

      ye=self.P.y_xz(xz)
      y=self.Y_EncoderDecoder.Decoding(ye)
      #pyro.sample("y",dist.Normal(y_mu,y_sig).to_event(1),obs=y_obs) #reduce with bernoulli dist
      pyro.sample("y",dist.Binomial(total_count=self.entities,probs=y).to_event(3),obs=y_obs)
  
  def guide(self,x_obs,y_obs):
    pyro.module("z_xy_mu_NET",self.Q.z_xy_mu)
    pyro.module("z_xy_sig_NET",self.Q.z_xy_sig)
    with pyro.plate("Data",x_obs.shape[0]):
      #ENCODE Y
      #Encode with same x encoder
      #with torch.no_grad():
      #  xe=self.Y_EncoderDecoder.Encoding(x_obs)
      #Parallely train a x-encoder decoder

      #Encode with it own y encoder
      xe=self.X_EncoderDecoder.Encoding(x_obs)
      ye=self.Y_EncoderDecoder.Encoding(y_obs)

      #xy=torch.cat((xe,ye.unsqueeze(1)),dim=1)
      xy=torch.cat((xe,ye),dim=1)
      z_mu=self.Q.z_xy_mu(xy)
      z_sig=torch.exp(self.Q.z_xy_sig(xy))
      z=pyro.sample("z",dist.Normal(z_mu,z_sig).to_event(1))

  def compute_losses(self,x_obs,y_obs):
    self.losses["total_loss"]=0
    self.losses["generative_loss"]=self.gen_loss_fun(self.model,self.guide,*(x_obs,y_obs))

    for loss in self.losses_weigths.keys():
      self.losses["total_loss"]=self.losses["total_loss"]+self.losses[loss]*self.losses_weigths[loss]

    return self.losses
  
  def forward_pass(self,x,batch_size=1):

    tshape=np.array(self.Y_EncoderDecoder.fl.i_shape)
    tshape[0]=batch_size
    self.Y_EncoderDecoder.fl.i_shape=torch.Size(tshape)

    xe=self.X_EncoderDecoder.Encoding(x)
    #z allocation

    z_mu=self.P.z_x_mu(xe)
    z_sig=torch.exp(self.P.z_x_sig(xe))
    z=dist.Normal(z_mu,z_sig).sample()  #Change z generation in dependent of x

    xz=torch.cat((xe,z),dim=1)
    ye=self.P.y_xz(xz)
    y=self.Y_EncoderDecoder.Decoding(ye)

    y_c=dist.Binomial(total_count=self.entities,probs=y).sample()

    return z_mu,z_sig,y,y_c

class Baseline_CVAE(NeuralNet):
  """
  baseline should be a function that maps x to y
  """
  def __init__(self,baseline,xz_layer_size,xz_enc_activators,x_layer_size,x_enc_activators,xy_layer_size,xy_enc_activators,save_output=False,aux_dir=None,module_name=None):
    self.baseline=baseline

    self.EncoderDecoder=Encoder_Decoder_Module
    #self.pre_net=NeuralNet(pre_layer_size,[nn.LeakyReLU() for i in range(len(pre_layer_size-1))],save_output,aux_dir,module_name)
    #self.post_net=NeuralNet(pre_layer_size[::-1],[nn.LeakyReLU() for i in range(len(pre_layer_size-1))],save_output,aux_dir,module_name)

    self.y_xz_mu=NeuralNet(xz_layer_size,xz_enc_activators,save_output,aux_dir,module_name)
    self.y_xz_sig=NeuralNet(xz_layer_size,xz_enc_activators,save_output,aux_dir,module_name)

    self.z_x_mu=NeuralNet(x_layer_size,x_enc_activators,save_output,aux_dir,module_name)
    self.z_x_sig=NeuralNet(x_layer_size,x_enc_activators,save_output,aux_dir,module_name)

    self.z_xy_mu=NeuralNet(xy_layer_size,xy_enc_activators,save_output,aux_dir,module_name)
    self.z_xy_sig=NeuralNet(xy_layer_size,xy_enc_activators,save_output,aux_dir,module_name)

    self.z_dim=x_layer_size[-1]
    self.y_dim=xz_layer_size[-1]
    #self.x_z=NeuralN(layer_size[::-1],dec_activators,save_output,aux_dir,module_name)

  def model(self,x_obs,y_obs,In_ID=None):
    pyro.module("y_xz_mu_NET",self.y_xz_mu)
    pyro.module("y_xz_sig_NET",self.y_xz_sig)
    with pyro.plate("Data",x_obs.shape[0]):
      x=self.pre_net(x_obs,In_ID)

      #compute y estimation
      y_est=self.baseline(x_obs)
      if not isinstance(y_est,torch.tensor):
        y_est=torch.tensor(y_est)
      y_est=y_est.reshape(y_obs.shape).detach() #Check torch.no_grad()

      xy_est=torch.cat((x,y_est),dim=1)

      z_mu=self.z_x_mu(xy_est,In_ID)
      z_sig=self.z_x_sig(xy_est,In_ID)
      z=pyro.sample("z",dist.Normal(z_mu,z_sig).to_event(1))  #Change z generation in dependent of x
      xz=torch.cat((x,z),dim=1)
      y_mu=self.y_xz_mu(xz,In_ID)
      y_sig=torch.exp(self.y_xz_sig(xz,In_ID))
      #pyro.sample("y",dist.Bernoulli(y).to_event(1),obs=y_obs) #reduce with bernoulli dist
      pyro.sample("y",dist.Normal(y_mu,y_sig).to_event(1),obs=y_obs) #reduce with bernoulli dist
  
  def guide(self,x_obs,y_obs,In_ID=None):
    pyro.module("z_x_mu_NET",self.z_x_mu)
    pyro.module("z_x_sig_NET",self.z_x_sig)
    with pyro.plate("Data",x_obs.shape[0]):
      x_obs=self.pre_net(x_obs,In_ID)
      z_mu=self.z_x_mu(x_obs,In_ID)
      z_sig=self.z_x_sig(x_obs,In_ID)
      z=pyro.sample("z",dist.Normal(z_mu,z_sig).to_event(1))
  
  def forward_pass(self,x,In_ID):
    xe=self.pre_net(x,In_ID)

    y_est=self.baseline(x)
    if not isinstance(y_est,torch.tensor):
      y_est=torch.tensor(y_est)
    y_est=y_est.reshape((1,self.y_dim)).detach() #Check torch.no_grad()

    xy_est=torch.cat((xe,y_est),dim=1)

    z_mu=self.z_x_mu(xy_est,In_ID)
    z_sig=self.z_x_sig(xy_est,In_ID)

    z=dist.Normal(z_mu,z_sig).sample()

    xz=torch.cat((x,z),dim=1)

    y_mu=self.y_xz_mu(xz,In_ID)
    y_sig=torch.exp(self.y_xz_sig(xz,In_ID))

    y=dist.Normal(y_mu,y_sig).sample()

    return z_mu,z_sig,y_mu,y_sig,y