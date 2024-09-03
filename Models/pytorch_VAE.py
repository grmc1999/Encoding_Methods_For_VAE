
import torch
from torch import nn
import copy
import torch.nn.functional as F
from DL_utils.utils import NeuralNet
from DL_utils.Generative_autoencoder_utils import Base_Generative_AutoEncoder
from einops import rearrange

class Unpaired_Flexible_Encoding_Decoding_VAE(Base_Generative_AutoEncoder):
  def __init__(self,encoding_decoding_module,P_NET,Q_NET,losses_weigths={"generative_loss":1},subsample=None,sig_scale=1,resize=None,save_output=False,aux_dir=None,module_name=None):
    super(Unpaired_Flexible_Encoding_Decoding_VAE,self).__init__(Encoder_Decoder_Module=None,P_NET=P_NET,Q_NET=Q_NET,losses_weigths=losses_weigths)
    """
    encoding_module: Trainable function that maps X to y where y is a vector
    decoding_module: Trainable function that maps y to y where X is a vector
    """
    self.subsample=subsample
    self.scale=sig_scale
    self.losses={}
    self.reconstruction_fun=torch.nn.BCELoss(reduction="mean")
    self.resize=resize

    self.Encoding=encoding_decoding_module.Encoding
    self.Decoding=encoding_decoding_module.Decoding
    self.z_dim=self.Q.z_x_mu.layer_sizes[-1]

  def model(self,z):
    
    x_re=self.P.x_z(z)
    x_r=self.Decoding(x_re)

    if self.resize!=None:
      x_r=F.interpolate(x_r,self.resize)
    
    return x_r
  
  def guide(self,x):

    xe_q=self.Encoding(x) #COMMENT: Q image encoding will output 2N entities [B,W,H,R]

    z_mu_q=self.Q.z_x_mu(xe_q)
    z_sig_q=torch.exp(self.Q.z_x_sig(xe_q)*self.scale)

    return (z_mu_q,z_sig_q)
  
  def reconstruction_loss(self,r_x,x):
    BCE=self.reconstruction_fun(r_x,x) #TODO: implement BCE
    return BCE
  
  def KLD_loss(self,z_mean,z_sig):
    KLD=-0.5*torch.mean(1+torch.log(z_sig.pow(2))-z_mean.pow(2)-z_sig.pow(2))
    return KLD
  
  def reparametrize(self,mu,sig):
    std=sig.pow(2)
    eps=torch.randn_like(std)
    return eps*std + mu
  
  def ELBO(self,x):
    x_r,z_mean,z_logvar=self.forward(x)  
    reconstruction=self.reconstruction_loss(x_r,x)
    KLD=self.KLD_loss(z_mean,z_logvar)

    loss=reconstruction+KLD

        #BUILD LOSSES DICT
    self.losses['KLD']=KLD
    self.losses['reconstruction']=reconstruction
    self.losses["total_loss"]=loss
      
    return self.losses

  def compute_losses(self,x):
    z_q=self.guide(x)
    z_q_=self.reparametrize(*(z_q))

    x_r=self.model(z_q_)

    self.losses["total_loss"]=0
    self.losses["reconstructive"]=self.reconstruction_loss(x_r,x)
    self.losses["generative"]=self.KLD_loss(*(z_q))
      
    for loss in self.losses_weigths.keys():
      self.losses["total_loss"]=self.losses["total_loss"]+self.losses[loss]*self.losses_weigths[loss]

    return self.losses
  
  def forward_pass(self,xi):
    z_d=self.guide(xi)
    z_q_=self.reparametrize(*(z_d))

    x_r=self.model(z_q_)

    return z_d[0].detach(),z_d[1].detach(),x_r.detach()


class Unpaired_Flexible_Encoding_Decoding_VAE_Decoupler(Base_Generative_AutoEncoder):
  def __init__(self,p_encoding_module,q_encoding_module,decoding_module,P_NET,Q_NET,P_NET_ENC,losses_weigths={"generative_loss":1},subsample=None,sig_scale=1,resize=None,save_output=False,aux_dir=None,module_name=None):
    super(Unpaired_Flexible_Encoding_Decoding_VAE_Decoupler,self).__init__(Encoder_Decoder_Module=None,P_NET=P_NET,Q_NET=Q_NET,losses_weigths=losses_weigths)
    """
    encoding_module: Trainable function that maps X to y where y is a vector
    decoding_module: Trainable function that maps y to y where X is a vector
    """
    self.subsample=subsample
    self.scale=sig_scale
    self.losses={}
    self.reconstruction_fun=torch.nn.BCELoss(reduction="mean")
    self.resize=resize

    self.Q_Encoding=q_encoding_module
    self.P_Encoding=p_encoding_module
    self.Decoding=decoding_module
    self.P_ENC=P_NET_ENC
    self.z_dim=self.Q.z_x_mu.layer_sizes[-1]

  def model(self,z_1,z_2):
    
    x_re_1=self.P.x_z(z_1)
    x_re_2=self.P.x_z(z_2)
    x_r_1=self.Decoding.Decoding(x_re_1)
    x_r_2=self.Decoding.Decoding(x_re_2)

    if self.resize!=None:
      x_r_1=F.interpolate(x_r_1,self.resize)
      x_r_2=F.interpolate(x_r_2,self.resize)
    
    return x_r_1,x_r_2
  
  def guide(self,x_12,x_1,x_2):

    xe_1=self.P_Encoding.Encoding(x_1)
    z_mu_1_p=self.P_ENC.z_x_mu(xe_1)
    z_sig_1_p=torch.exp(self.P_ENC.z_x_sig(xe_1)*self.scale)

    xe_2=self.P_Encoding.Encoding(x_2)
    z_mu_2_p=self.P_ENC.z_x_mu(xe_2)
    z_sig_2_p=torch.exp(self.P_ENC.z_x_sig(xe_2)*self.scale)


    xe_12_q=self.Q_Encoding.Encoding(x_12) #COMMENT: Q image encoding will output 2N entities [B,W,H,R]
    xe_1_q,xe_2_q=rearrange(xe_12_q,' b (n_ent r) w h -> n_ent b (r w h)',n_ent=2)

    z_mu_1_q=self.Q.z_x_mu(xe_1_q)
    z_sig_1_q=torch.exp(self.Q.z_x_sig(xe_1_q)*self.scale)

    z_mu_2_q=self.Q.z_x_mu(xe_2_q)
    z_sig_2_q=torch.exp(self.Q.z_x_sig(xe_2_q)*self.scale)
    
    return (z_mu_1_p,z_sig_1_p),(z_mu_2_p,z_sig_2_p),(z_mu_1_q,z_sig_1_q),(z_mu_2_q,z_sig_2_q)
  
  def reconstruction_loss(self,r_x,x):
    BCE=self.reconstruction_fun(r_x,x) #TODO: implement BCE
    return BCE
  
  def KLD_loss(self,z_mean,z_sig):
    KLD=-0.5*torch.mean(1+torch.log(z_sig.pow(2))-z_mean.pow(2)-z_sig.pow(2))
    return KLD
  
  def reparametrize(self,mu,sig):
    std=sig.pow(2)
    eps=torch.randn_like(std)
    return eps*std + mu
  
  def ELBO(self,x):
    x_r,z_mean,z_logvar=self.forward(x)  
    reconstruction=self.reconstruction_loss(x_r,x)
    KLD=self.KLD_loss(z_mean,z_logvar)

    loss=reconstruction+KLD

        #BUILD LOSSES DICT
    self.losses['KLD']=KLD
    self.losses['reconstruction']=reconstruction
    self.losses["total_loss"]=loss
      
    return self.losses

  def compute_losses(self,x_12,x_1,x_2):
    z_1_p,z_2_p,z_1_q,z_2_q=self.guide(x_12,x_1,x_2)
    #TODO: distance between z_p and z_q

    #z_1_p=self.reparametrize(*(z_1_p))
    #z_2_p=self.reparametrize(*(z_2_p))
    z_1_q_=self.reparametrize(*(z_1_q))
    z_2_q_=self.reparametrize(*(z_2_q))

    x_r_1,x_r_2=self.model(z_1_q_,z_2_q_)

    self.losses["total_loss"]=0
    self.losses["reconstructive_1"]=self.reconstruction_loss(x_r_1,x_1)
    self.losses["reconstructive_2"]=self.reconstruction_loss(x_r_2,x_2)
    self.losses["generative_p_1"]=self.KLD_loss(*(z_1_p))
    self.losses["generative_p_2"]=self.KLD_loss(*(z_2_p))
    self.losses["generative_q_1"]=self.KLD_loss(*(z_2_q))
    self.losses["generative_q_2"]=self.KLD_loss(*(z_1_q))
      
    for loss in self.losses_weigths.keys():
      self.losses["total_loss"]=self.losses["total_loss"]+self.losses[loss]*self.losses_weigths[loss]

    return self.losses

#  def gen_forward_pass(self,xi):
#    x=self.Encoding.Encoding(xi)
#    z_mu=self.Q.z_x_mu(x)
#    z_sig=torch.exp(self.Q.z_x_sig(x))
#
#    z=dist.Normal(z_mu,z_sig).sample()
#
#    x_re=self.P.x_z(z)
#    x_r=self.Decoding.Decoding(x_re)
#
#    return {"z_mu":z_mu.detach().cpu(),"z_sig":z_sig.detach().cpu(),"x_r":x_r.detach().cpu()}
#
#  def forward_pass(self,xi):
#    x=self.Encoding.Encoding(xi)
#    z_mu=self.Q.z_x_mu(x)
#    z_sig=torch.exp(self.Q.z_x_sig(x))
#
#    z=dist.Normal(z_mu,z_sig).sample()
#
#    x_re=self.P.x_z(z)
#    x_r=self.Decoding.Decoding(x_re)
#
#    return z_mu.detach(),z_sig.detach(),x_r.detach()
#
#  def forward_latent_pass(self,xi):
#    x=self.Encoding.Encoding(xi)
#    z_mu=self.Q.z_x_mu(x)
#    z_sig=torch.exp(self.Q.z_x_sig(x))
#
#    return z_mu.detach(),z_sig.detach()