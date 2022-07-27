import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange

class Dot_product_attention(nn.Module):
    """
    Forward: [batch,(h w),(heads head_size)]
    """
    def __init__(self,head_size):
        super(Dot_product_attention,self).__init__()
        self.softmax=nn.Softmax(dim=-1)
        self.scale=head_size**-0.5
    def forward(self,q,k,v):
        dot=torch.matmul(q,k.transpose(-1,-2))*self.scale
        out=torch.matmul(self.softmax(dot),v)
        return dot,out

class Additive_attention(nn.Module):
    def __init__(self,h_dim,w_dim,heads,head_size):
        super(Additive_attention,self).__init__()
        t_size=2*h_dim*w_dim*heads*head_size
        self.Tanh=nn.Tanh()
        self.softmax=nn.Softmax(dim=-1)
        self.W=nn.linear(t_size,t_size,bias=False)
        self.V=nn.linear(t_size,t_size,bias=False)
    def forward(self,q,k,v): #[batch,(h w),(heads head_size)]
        dot=self.V(
                self.Tanh(
                    self.W(
                        torch.cat(
                            rearrange(q,"batch,(h w),(heads head_size)->batch,(h w heads head_size)"),
                            rearrange(k,"batch,(h w),(heads head_size)->batch,(h w heads head_size)"),
                            dim=-1
                            )
                    )
                )
            )
        out=v*rearrange(self.softmax(dot),"batch,(h w heads head_size)->batch,(h w),(heads head_size)")
        return dot,out

#class Additive_attention(nn.Module):
#    def __init__(self,h_dim,w_dim,heads,head_size):
#        t_size=h_dim*w_dim*heads*head_size
#        self.Tanh=nn.Tanh()
#        self.softmax=nn.Softmax(dim=-1)
#        self.W=nn.linear(t_size,t_size,bias=False)
#        self.V=nn.linear(t_size,t_size,bias=False)
#    def forward(self,q,k,v): #[batch,(h w),(heads head_size)]
#        dot=self.V(
#                self.Tanh(
#                    self.W(
#                        torch.cat(
#                            rearrange(q,"batch,(h w),(heads head_size)->batch,(h w heads head_size)"),
#                            rearrange(k,"batch,(h w),(heads head_size)->batch,(h w heads head_size)"),
#                            dim=-1
#                            )
#                    )
#                )
#            )
#        out=v*rearrange(self.softmax(dot),"batch,(h w heads head_size)->batch,(h w),(heads head_size)")
#        return dot,out

    