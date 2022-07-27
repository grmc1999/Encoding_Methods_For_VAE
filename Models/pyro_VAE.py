import torch
from torch import nn
import copy
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
#from pyro.infer import SVI,Trace_ELBO
#from pyro.optim import Adam

from DL_utils.utils import NeuralNet
from DL_utils.Generative_autoencoder_utils import Base_Generative_AutoEncoder


class VAE(NeuralNet):
  def __init__(self,pre_layer_size,layer_size,enc_activators,dec_activators,save_output=False,aux_dir=None,module_name=None):
    super(VAE,self).__init__(layer_size=[],activators=[],save_output=save_output,aux_dir=aux_dir,module_name=module_name)

    self.pre_net=NeuralNet(pre_layer_size,[nn.GELU() for i in range(len(pre_layer_size)-1)],save_output,aux_dir,module_name)
    self.post_net=NeuralNet(pre_layer_size[::-1],[nn.GELU() for i in range(len(pre_layer_size)-2)]+[nn.Sigmoid()],save_output,aux_dir,module_name)
    self.z_x_mu=NeuralNet(layer_size,enc_activators,save_output,aux_dir,module_name)
    self.z_x_sig=NeuralNet(layer_size,enc_activators,save_output,aux_dir,module_name)
    self.x_z=NeuralNet(layer_size[::-1],dec_activators,save_output,aux_dir,module_name)
    self.z_dim=layer_size[-1]

  def model(self,x,In_ID=None):
    #Sampling pass
    pyro.module("post_NET",self.post_net)
    pyro.module("x_z_NET",self.x_z)
    with pyro.plate("data",x.shape[0]):
      z_mu=x.new_zeros(torch.Size((x.shape[0],self.z_dim)))
      z_sig=x.new_ones(torch.Size((x.shape[0],self.z_dim)))

      #Allocate memory for sampling
      #define "latent" probabilistic variable with its prior
      #dimension 1 is independent
      z=pyro.sample("latent",dist.Normal(z_mu,z_sig).to_event(1))

      x_re=self.x_z(z,In_ID)
      x_r=self.post_net(x_re,In_ID)

      #Define prior relation to obs
      pyro.sample("obs",dist.Bernoulli(x_r).to_event(1),obs=((x>0.5)).float())
      #pyro.sample("obs",dist.Bernoulli(x_r).to_event(1),obs=((x)).float())
  
  def guide(self,x,In_ID=None):
    #Inference pass
    pyro.module("pre_NET",self.pre_net)
    pyro.module("z_x_mu_NET",self.z_x_mu)
    pyro.module("z_x_sig_NET",self.z_x_sig)

    with pyro.plate("data",x.shape[0]):

      xe=self.pre_net(x,In_ID)
      z_mu=self.z_x_mu(xe,In_ID)
      z_sig=torch.exp(self.z_x_sig(xe,In_ID))
      #define prior in inference
      z=pyro.sample("latent",dist.Normal(z_mu,z_sig).to_event(1))



  def forward_pass(self,xi,In_ID=None):
    x=self.pre_net(xi,In_ID)
    z_mu=self.z_x_mu(x,In_ID)
    z_sig=torch.exp(self.z_x_sig(x,In_ID))

    z=dist.Normal(z_mu,z_sig).sample()

    x_re=self.x_z(z,In_ID)
    x_r=self.post_net(x_re,In_ID)

    return z_mu.detach(),z_sig.detach(),x_r.detach()

class Flexible_Encoding_Decoding_VAE(Base_Generative_AutoEncoder):
  def __init__(self,encoding_decoding_module,P_NET,Q_NET,losses_weigths={"generative_loss":1},subsample=None,sig_scale=1,resize=None,save_output=False,aux_dir=None,module_name=None):
    super(Flexible_Encoding_Decoding_VAE,self).__init__(Encoder_Decoder_Module=encoding_decoding_module,P_NET=P_NET,Q_NET=Q_NET,losses_weigths=losses_weigths)
    """
    encoding_module: Trainable function that maps X to y where y is a vector
    decoding_module: Trainable function that maps y to y where X is a vector
    """
    self.subsample=subsample
    self.scale=sig_scale
    self.losses={}
    self.gen_loss_fun=pyro.infer.Trace_ELBO().differentiable_loss
    self.resize=resize

    self.Encoding_Decoding=encoding_decoding_module
    #self.z_x_mu=NeuralNet(layer_size,enc_activators)
    #self.z_x_sig=NeuralNet(layer_size,enc_activators)
    #self.x_z=NeuralNet(layer_size[::-1],dec_activators)
    self.z_dim=self.Q.z_x_mu.layer_sizes[-1]

  def model(self,x):
    #Sampling pass
    #pyro.module("post_NET",self.Encoding_Decoding.Decoding)
    pyro.module("x_z_NET",self.P.x_z)
    with pyro.plate("data",x.shape[0]):
      z_mu=x.new_zeros(torch.Size((x.shape[0],self.z_dim)))
      z_sig=x.new_ones(torch.Size((x.shape[0],self.z_dim)))

      #Allocate memory for sampling
      #define "latent" probabilistic variable with its prior
      #dimension 1 is independent
      z=pyro.sample("latent",dist.Normal(z_mu,z_sig).to_event(1))

      x_re=self.P.x_z(z)
      x_r=self.Encoding_Decoding.Decoding(x_re)

      #Add losses

      #Define prior relation to obs
      if self.resize!=None:
        x=F.interpolate(x,self.resize)

      pyro.sample("obs",dist.Bernoulli(x_r).to_event(3),obs=((x>0.5)).float())
      #pyro.sample("obs",dist.Bernoulli(x_r).to_event(1),obs=((x)).float())
  
  def guide(self,x):
    #Inference pass
    #pyro.module("pre_NET",self.Encoding_Decoding.Encoding)
    pyro.module("z_x_mu_NET",self.Q.z_x_mu)
    pyro.module("z_x_sig_NET",self.Q.z_x_sig)

    with pyro.plate("data",x.shape[0],subsample=self.subsample):

      xe=self.Encoding_Decoding.Encoding(x)
      z_mu=self.Q.z_x_mu(xe)
      z_sig=torch.exp(self.Q.z_x_sig(xe)*self.scale)
      #define prior in inference
      z=pyro.sample("latent",dist.Normal(z_mu,z_sig).to_event(1))

  def compute_losses(self,x_obs):
    self.losses["total_loss"]=0
    self.losses["generative_loss"]=self.gen_loss_fun(self.model,self.guide,x_obs)

    for loss in self.losses_weigths.keys():
      self.losses["total_loss"]=self.losses["total_loss"]+self.losses[loss]*self.losses_weigths[loss]

    return self.losses

  def gen_forward_pass(self,xi):
    x=self.Encoding_Decoding.Encoding(xi)
    z_mu=self.Q.z_x_mu(x)
    z_sig=torch.exp(self.Q.z_x_sig(x))

    z=dist.Normal(z_mu,z_sig).sample()

    x_re=self.P.x_z(z)
    x_r=self.Encoding_Decoding.Decoding(x_re)

    return {"z_mu":z_mu.detach().cpu(),"z_sig":z_sig.detach().cpu(),"x_r":x_r.detach().cpu()}

  def forward_pass(self,xi):
    x=self.Encoding_Decoding.Encoding(xi)
    z_mu=self.Q.z_x_mu(x)
    z_sig=torch.exp(self.Q.z_x_sig(x))

    z=dist.Normal(z_mu,z_sig).sample()

    x_re=self.P.x_z(z)
    x_r=self.Encoding_Decoding.Decoding(x_re)

    return z_mu.detach(),z_sig.detach(),x_r.detach()

  def forward_latent_pass(self,xi):
    x=self.Encoding_Decoding.Encoding(xi)
    z_mu=self.Q.z_x_mu(x)
    z_sig=torch.exp(self.Q.z_x_sig(x))

    return z_mu.detach(),z_sig.detach()

class Decoupled_Loss_Flexible_Encoding_Decoding_VAE(Base_Generative_AutoEncoder):
  def __init__(self,encoding_decoding_module,P_NET,Q_NET,losses_weigths={"generative_loss":1},subsample=None,sig_scale=1,save_output=False,aux_dir=None,module_name=None,resize=None):
    super(Flexible_Encoding_Decoding_VAE,self).__init__(Encoder_Decoder_Module=encoding_decoding_module,P_NET=P_NET,Q_NET=Q_NET,losses_weigths=losses_weigths)
    """
    encoding_module: Trainable function that maps X to y where y is a vector
    decoding_module: Trainable function that maps y to y where X is a vector
    """
    self.resize=resize
    self.subsample=subsample
    self.scale=sig_scale
    self.losses={}
    self.gen_loss_fun=pyro.infer.Trace_ELBO().differentiable_loss

    self.Encoding_Decoding=encoding_decoding_module
    #self.z_x_mu=NeuralNet(layer_size,enc_activators)
    #self.z_x_sig=NeuralNet(layer_size,enc_activators)
    #self.x_z=NeuralNet(layer_size[::-1],dec_activators)
    self.z_dim=self.Q.z_x_mu.layer_sizes[-1]

  def model(self,x):
    #Sampling pass
    #pyro.module("post_NET",self.Encoding_Decoding.Decoding)
    pyro.module("x_z_NET",self.P.x_z)
    with pyro.plate("data",x.shape[0]):
      z_mu=x.new_zeros(torch.Size((x.shape[0],self.z_dim)))
      z_sig=x.new_ones(torch.Size((x.shape[0],self.z_dim)))

      #Allocate memory for sampling
      #define "latent" probabilistic variable with its prior
      #dimension 1 is independent
      z=pyro.sample("latent",dist.Normal(z_mu,z_sig).to_event(1))

      x_re=self.P.x_z(z)
      x_r=self.Encoding_Decoding.Decoding(x_re)

      #Add losses

      #Define prior relation to obs
      if self.resize!=None:
        x=F.interpolate(x,self.resize)
      self.x_r=pyro.sample("obs",dist.Bernoulli(x_r).to_event(3),obs=((x>0.5)).float())
  
  def guide(self,x):
    #Inference pass
    #pyro.module("pre_NET",self.Encoding_Decoding.Encoding)
    pyro.module("z_x_mu_NET",self.Q.z_x_mu)
    pyro.module("z_x_sig_NET",self.Q.z_x_sig)

    with pyro.plate("data",x.shape[0],subsample=self.subsample):

      xe=self.Encoding_Decoding.Encoding(x)
      z_mu=self.Q.z_x_mu(xe)
      z_sig=torch.exp(self.Q.z_x_sig(xe)*self.scale)
      #define prior in inference
      z=pyro.sample("latent",dist.Normal(z_mu,z_sig).to_event(1))

  def compute_losses(self,x_obs):
    self.losses["total_loss"]=0
    self.losses["generative_loss"]=self.gen_loss_fun(self.model,self.guide,x_obs)
    self.losses["reconstruction"]=F.binary_cross_entropy(self.x_r,x_obs,size_average=False)

    for loss in self.losses_weigths.keys():
      self.losses["total_loss"]=self.losses["total_loss"]+self.losses[loss]*self.losses_weigths[loss]

    return self.losses

  def gen_forward_pass(self,xi):
    x=self.Encoding_Decoding.Encoding(xi)
    z_mu=self.Q.z_x_mu(x)
    z_sig=torch.exp(self.Q.z_x_sig(x))

    z=dist.Normal(z_mu,z_sig).sample()

    x_re=self.P.x_z(z)
    x_r=self.Encoding_Decoding.Decoding(x_re)

    return {"z_mu":z_mu.detach().cpu(),"z_sig":z_sig.detach().cpu(),"x_r":x_r.detach().cpu()}

  def forward_pass(self,xi):
    x=self.Encoding_Decoding.Encoding(xi)
    z_mu=self.Q.z_x_mu(x)
    z_sig=torch.exp(self.Q.z_x_sig(x))

    z=dist.Normal(z_mu,z_sig).sample()

    x_re=self.P.x_z(z)
    x_r=self.Encoding_Decoding.Decoding(x_re)

    return z_mu.detach(),z_sig.detach(),x_r.detach()

class Decoupled_Loss_Normal_Prior_Flexible_Encoding_Decoding_VAE(Base_Generative_AutoEncoder):
  def __init__(self,encoding_decoding_module,P_NET,Q_NET,losses_weigths={"generative_loss":0.2,"deviation_loss":0.8},subsample=None,sig_scale=1,save_output=False,aux_dir=None,module_name=None,resize=None):
    super(Decoupled_Loss_Normal_Prior_Flexible_Encoding_Decoding_VAE,self).__init__(Encoder_Decoder_Module=encoding_decoding_module,P_NET=P_NET,Q_NET=Q_NET,losses_weigths=losses_weigths)
    """
    encoding_module: Trainable function that maps X to y where y is a vector
    decoding_module: Trainable function that maps y to y where X is a vector
    """
    self.resize=resize
    self.subsample=subsample
    self.scale=sig_scale
    self.losses={}
    self.gen_loss_fun=pyro.infer.Trace_ELBO().differentiable_loss

    #Define clones
    self.Encoding_Decoding=encoding_decoding_module
    self.Encoding_Decoding_mu=copy.deepcopy(self.Encoding_Decoding)
    self.Encoding_Decoding_sig=copy.deepcopy(self.Encoding_Decoding)

    del self.Encoding_Decoding.DEC
    del self.Encoding_Decoding_mu.ENC.im_layers
    del self.Encoding_Decoding_sig.ENC.im_layers
    self.P.x_z_mu=copy.deepcopy(self.P.x_z)
    self.P.x_z_sig=copy.deepcopy(self.P.x_z)

    del self.P.x_z

    self.z_dim=self.Q.z_x_mu.layer_sizes[-1]

    if hasattr(self.Encoding_Decoding_mu.ENC,"idx_list"):
      self.Encoding_Decoding_sig.ENC.idx_list=self.Encoding_Decoding.ENC.idx_list
      self.Encoding_Decoding_mu.ENC.idx_list=self.Encoding_Decoding.ENC.idx_list
      self.Encoding_Decoding_sig.ENC.Sidx_list=self.Encoding_Decoding.ENC.Sidx_list
      self.Encoding_Decoding_mu.ENC.Sidx_list=self.Encoding_Decoding.ENC.Sidx_list

  def model(self,x):
    pyro.module("x_z_mu_NET",self.P.x_z_mu)
    pyro.module("x_z_sig_NET",self.P.x_z_sig)
    with pyro.plate("data",x.shape[0]):
      z_mu=x.new_zeros(torch.Size((x.shape[0],self.z_dim)))
      z_sig=x.new_ones(torch.Size((x.shape[0],self.z_dim)))

      z=pyro.sample("latent",dist.Normal(z_mu,z_sig).to_event(1))

      #Mean
      x_re=self.P.x_z_mu(z)
      self.x_r_mu=self.Encoding_Decoding_mu.Decoding(x_re)
      #Std
      x_re=self.P.x_z_sig(z)
      self.x_r_sig=self.Encoding_Decoding_sig.Decoding(x_re)
      self.x_r_sig=torch.exp(self.x_r_sig*0.01)

      #Add losses

      #Define prior relation to obs
      if self.resize!=None:
        x=F.interpolate(x,self.resize)

      self.x_r=pyro.sample("obs",dist.Normal(self.x_r_mu,self.x_r_sig).to_event(3),obs=x.float())
  
  def guide(self,x):
    #Inference pass
    #pyro.module("pre_NET",self.Encoding_Decoding.Encoding)
    pyro.module("z_x_mu_NET",self.Q.z_x_mu)
    pyro.module("z_x_sig_NET",self.Q.z_x_sig)

    with pyro.plate("data",x.shape[0],subsample=self.subsample):

      xe=self.Encoding_Decoding.Encoding(x)
      z_mu=self.Q.z_x_mu(xe)
      z_sig=torch.exp(self.Q.z_x_sig(xe)*self.scale)
      #define prior in inference
      z=pyro.sample("latent",dist.Normal(z_mu,z_sig).to_event(1))

      #ATTRIBUTE SHARING
      self.Encoding_Decoding_sig.fl.i_shape=self.Encoding_Decoding.fl.i_shape
      self.Encoding_Decoding_mu.fl.i_shape=self.Encoding_Decoding.fl.i_shape
      if hasattr(self.Encoding_Decoding_mu.ENC,"idx_list"):
        self.Encoding_Decoding_sig.ENC.idx_list=self.Encoding_Decoding.ENC.idx_list
        self.Encoding_Decoding_mu.ENC.idx_list=self.Encoding_Decoding.ENC.idx_list
        self.Encoding_Decoding_sig.ENC.Sidx_list=self.Encoding_Decoding.ENC.Sidx_list
        self.Encoding_Decoding_mu.ENC.Sidx_list=self.Encoding_Decoding.ENC.Sidx_list


  def compute_losses(self,x_obs):
    self.losses["total_loss"]=0
    self.losses["generative_loss"]=self.gen_loss_fun(self.model,self.guide,x_obs)
    self.losses["deviation_loss"]=torch.sum(self.x_r_sig)
    #self.losses["reconstruction"]=F.binary_cross_entropy(self.x_r,x_obs,size_average=False)

    for loss in self.losses_weigths.keys():
      self.losses["total_loss"]=self.losses["total_loss"]+self.losses[loss]*self.losses_weigths[loss]

    return self.losses

  def gen_forward_pass(self,xi,resize=None):
    x=self.Encoding_Decoding.Encoding(xi)

    #ATTRIBUTE SHARING
    self.Encoding_Decoding_sig.fl.i_shape=self.Encoding_Decoding.fl.i_shape
    self.Encoding_Decoding_mu.fl.i_shape=self.Encoding_Decoding.fl.i_shape
    if hasattr(self.Encoding_Decoding_mu.ENC,"idx_list"):
      self.Encoding_Decoding_sig.ENC.idx_list=self.Encoding_Decoding.ENC.idx_list
      self.Encoding_Decoding_mu.ENC.idx_list=self.Encoding_Decoding.ENC.idx_list
      self.Encoding_Decoding_sig.ENC.Sidx_list=self.Encoding_Decoding.ENC.Sidx_list
      self.Encoding_Decoding_mu.ENC.Sidx_list=self.Encoding_Decoding.ENC.Sidx_list

    z_mu=self.Q.z_x_mu(x)
    z_sig=torch.exp(self.Q.z_x_sig(x))

    z=dist.Normal(z_mu,z_sig).sample()

    #DECODING
    x_re=self.P.x_z_mu(z)
    x_r_mu=self.Encoding_Decoding_mu.Decoding(x_re)
    #Std
    x_re=self.P.x_z_sig(z)
    x_r_sig=self.Encoding_Decoding_sig.Decoding(x_re)
    x_r_sig=torch.exp(x_r_sig*0.01)
    #Add losses
    #Define prior relation to obs
    x_r=dist.Normal(x_r_mu,x_r_sig).sample()

    return {"z_mu":z_mu.detach().cpu(),"z_sig":z_sig.detach().cpu(),"x_r":x_r.detach().cpu()}

  def forward_pass(self,xi,resize=None):
    x=self.Encoding_Decoding.Encoding(xi)

    #ATTRIBUTE SHARING
    self.Encoding_Decoding_sig.fl.i_shape=self.Encoding_Decoding.fl.i_shape
    self.Encoding_Decoding_mu.fl.i_shape=self.Encoding_Decoding.fl.i_shape
    if hasattr(self.Encoding_Decoding_mu.ENC,"idx_list"):
      self.Encoding_Decoding_sig.ENC.idx_list=self.Encoding_Decoding.ENC.idx_list
      self.Encoding_Decoding_mu.ENC.idx_list=self.Encoding_Decoding.ENC.idx_list
      self.Encoding_Decoding_sig.ENC.Sidx_list=self.Encoding_Decoding.ENC.Sidx_list
      self.Encoding_Decoding_mu.ENC.Sidx_list=self.Encoding_Decoding.ENC.Sidx_list

    z_mu=self.Q.z_x_mu(x)
    z_sig=torch.exp(self.Q.z_x_sig(x))

    z=dist.Normal(z_mu,z_sig).sample()

    #DECODING
    x_re=self.P.x_z_mu(z)
    x_r_mu=self.Encoding_Decoding_mu.Decoding(x_re)
    #Std
    x_re=self.P.x_z_sig(z)
    x_r_sig=self.Encoding_Decoding_sig.Decoding(x_re)
    x_r_sig=torch.exp(x_r_sig*self.scale)
    
    #Add losses
    #Define prior relation to obs
    x_r=dist.Normal(x_r_mu,x_r_sig).sample()

    return z_mu.detach(),z_sig.detach(),x_r.detach()