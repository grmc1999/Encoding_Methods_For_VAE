import torch
from torch import nn
from einops import rearrange

class set_enc_NN(nn.Module):
    def __init__(self,inp,out,act=nn.ReLU(),batch_norm=True,dropout=None):
        super(set_enc_NN, self).__init__()

        self.comp_layer=nn.ModuleList(
            [nn.Linear(inp,out)]+\
            ([nn.BatchNorm1d(out)] if batch_norm else [])+\
            [act]+\
            ([nn.Dropout(dropout)] if dropout!=None else [])
        )

    def forward(self,x):
        for l in self.comp_layer:
            x=l(x)
        return x

class set_dec_NN(nn.Module):
    def __init__(self,inp,out,act=nn.ReLU(),batch_norm=True,dropout=None):
        super(set_dec_NN, self).__init__()

        self.comp_layer=nn.ModuleList(
            [nn.Linear(inp,out)]+\
            ([nn.BatchNorm1d(out)] if batch_norm else [])+\
            [act]+\
            ([nn.Dropout(dropout)] if dropout!=None else [])
        )
    def forward(self,x):
        for l in self.comp_layer:
            x=l(x)
        return x


class b_encoder_NN(nn.Module):
    def __init__(self,inp_sizes=[5],activators=nn.ReLU(),batch_norm=True,dropout=None):
        super(b_encoder_NN, self).__init__()
        self.inp_sizes=inp_sizes
        #self.stride=[stride for i in range(len(repr_sizes)-1)]
        
        #kernels
        if isinstance(inp_sizes,int):
            self.inp_sizes=[inp_sizes for i in range(len(inp_sizes)-1)]
        else:
            self.inp_sizes=inp_sizes

        #activators
        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(inp_sizes)-1)]
        else:
            self.activators=activators
        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(inp_sizes)-1)]
        else:
            self.batch_norm=batch_norm
        
        if isinstance(dropout,float) or dropout==None:
            self.dropout=[dropout for i in range(len(inp_sizes)-1)]
        else:
            self.dropout=dropout
        
        self.im_layers=nn.ModuleList(
            [
                set_enc_NN(inp,
                out,
                act,
                batch_norm,
                dropout
                )
                for inp,out,act,batch_norm,dropout in zip(
                    self.inp_sizes[:-1],
                    self.inp_sizes[1:],
                    self.activators,
                    self.batch_norm,
                    self.dropout
                )
            ]
        )
    def forward(self,x):
        if len(x.shape)>3:
            x=rearrange(x,"b h w c -> b (h w c)")
        else:
            x=rearrange(x,"b h w -> b (h w)")
        for l in self.im_layers:
            x=l(x)
        return x
    
class b_decoder_NN(nn.Module):
    def __init__(self,inp_sizes=[5],activators=nn.ReLU(),batch_norm=True,dropout=None,input_size=[28,28]):
        super(b_decoder_NN,self).__init__()
        self.inp_sizes=inp_sizes
        self.inp_sizes=self.inp_sizes[::-1]
        self.input_size=input_size
        #self.stride=[stride for i in range(len(repr_sizes)-1)][::-1]
        #activators
        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(inp_sizes)-1)]
        else:
            self.activators=activators[::-1]
        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(inp_sizes)-1)]
        else:
            self.batch_norm=batch_norm[::-1]

        if isinstance(dropout,float) or dropout==None:
            self.dropout=[dropout for i in range(len(inp_sizes)-1)]
        else:
            self.dropout=dropout[::-1]
        
        self.im_layers=nn.ModuleList(
            [
                set_dec_NN(repr_in,
                repr_out,
                act,
                batch_norm,
                dropout,
                )
                for repr_in,repr_out,act,batch_norm,dropout in zip(
                    self.inp_sizes[:-1],
                    self.inp_sizes[1:],
                    self.activators,
                    self.batch_norm,
                    self.dropout
                )
            ]
        )
    def forward(self,x):
        for l in self.im_layers:
            x=l(x)
        
        x=rearrange(x,"b (h w c) -> b h w c",w=self.input_size[0],h=self.inp_sizes[1],c=self.inp_sizes[1])
        return x