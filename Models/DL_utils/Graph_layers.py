import torch
from torch_geometric.nn import MessagePassing
from torch import nn

def KNN_graph(encodings,k=3):
  encodings=encodings.reshape(encodings.shape[0],encodings.shape[1],-1)
  edges=[]
  for i in range(encodings.shape[1]):
    dist=(torch.sum((encodings-encodings[:,i].unsqueeze(1))**2,dim=2))**0.5
    n_ind=torch.topk(dist,k=k+1,largest=False,dim=1).indices[:].unsqueeze(1)
    edge=torch.cat(
        (
            n_ind,
            torch.ones_like(n_ind)*(n_ind[:,:,0].unsqueeze(1))
        ),
        dim=1
    )
    edges.append(edge)
  return encodings.float(),torch.cat(edges,dim=2).long()

def image_batch_to_graph_batch(edges,batch_size):
  padd=torch.arange(batch_size).to("cuda" if edges.is_cuda else "cpu")
  batched_idx_shift=torch.max(edges.reshape(batch_size,-1),dim=1)[0]*(padd)+padd
  batch_edges=edges+(batched_idx_shift).unsqueeze(1).unsqueeze(1)
  return batched_idx_shift,batch_edges

def split_batch_graph(batched_idx_shift,x):
  batch_samples=[]
  idxs=torch.cat((batched_idx_shift,torch.tensor([x.shape[0]]).to("cuda" if batched_idx_shift.is_cuda else "cpu")))
  idxs=torch.cat(
      (
        idxs[:-1].unsqueeze(0),
        idxs[1:].unsqueeze(0)
      ),dim=0
    ).T
  for idx in idxs:
    batch_samples.append(x[torch.arange(*idx).to("cuda" if batched_idx_shift.is_cuda else "cpu"),:].unsqueeze(0))
  return torch.cat(batch_samples,dim=0)

class Conv_WL_GNN(MessagePassing):
  def __init__(self,conv_in,conv_out):
    super(Conv_WL_GNN,self).__init__(aggr="add")
    self.N_conv_linear=nn.Linear(conv_in,conv_out)
    self.n_conv_linear=nn.Linear(conv_in,conv_out)

  def message(self,x_j):
    #Linear
    x_j=self.N_conv_linear(x_j)

    return x_j
  def update(self,aggr_out,x):
    #Linear to x
    #return sum to aggr_out
    return aggr_out+self.n_conv_linear(x)
    
  def forward(self,edge_index,x):
    return self.propagate(edge_index,x=x)