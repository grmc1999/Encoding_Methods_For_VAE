import torch
from torch import nn
from DL_utils.Graph_layers import *

class ResNET_WL(nn.Module):
    def __init__(self,image_encoding_mode,graph_encoding_mode,Estimation_NeuralNet,NN=10,losses_weigths={"regression_loss":1}):
      super(ResNET_WL, self).__init__()
      self.losses_weigths=losses_weigths
      self.losses={}
      self.MSEloss=nn.MSELoss()
      self.img_enc=image_encoding_mode
      self.grph_enc=graph_encoding_mode
      self.E_NN=Estimation_NeuralNet
      self.K=NN
    
    def compute_losses(self,x_obs,y_obs):
      #xie,gep,out=self.forward_pass(x_obs)
      _,_,out=self.forward_pass(x_obs)

      self.losses["total_loss"]=0
      self.losses["regression_loss"]=self.MSEloss(out,y_obs)
      for loss in self.losses_weigths.keys():
        self.losses["total_loss"]=self.losses["total_loss"]+self.losses[loss]*self.losses_weigths[loss]

      return self.losses

    def forward_pass(self,xi):
      xie=self.img_enc(xi)

      ge,edges=KNN_graph(xie,k=self.K)
      #image batch to graph batch
      batched_idx_shift,batch_edges=image_batch_to_graph_batch(edges,ge.shape[0])
      #graph processing
      gep=self.grph_enc(x=torch.concat(tuple(ge),dim=0), edge_index=batch_edges.reshape(2,-1))
      #graph batch split
      gep=split_batch_graph(batched_idx_shift,gep)
      #prediction per batch
      out=self.E_NN(gep.flatten(1))

      return xie,gep,out

    def gen_forward_pass(self,xi):
      xie=self.img_enc(xi)

      ge,edges=KNN_graph(xie,k=self.K)
      #image batch to graph batch
      batched_idx_shift,batch_edges=image_batch_to_graph_batch(edges,ge.shape[0])
      #graph processing
      gep=self.grph_enc(x=torch.concat(tuple(ge),dim=0), edge_index=batch_edges.reshape(2,-1))
      #graph batch split
      gep=split_batch_graph(batched_idx_shift,gep)
      #prediction per batch
      out=self.E_NN(gep.flatten(1))

      return {"xie":xie.detach().cpu().item(),"gep":gep.detach().cpu().item(),"out":out.detach().cpu().item()}