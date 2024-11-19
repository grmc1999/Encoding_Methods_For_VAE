from torch import dropout, nn
import torch.nn.functional as F
import torch
from einops.layers.torch import Rearrange
from einops import rearrange


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(pad) is not tuple:
        pad = (pad, pad)
    
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w

def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1,out_pad=0):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(pad) is not tuple:
        pad = (pad, pad)
        
    if type(dilation) is not tuple:
        dilation = (dilation, dilation)
        
    if type(out_pad) is not tuple:
        out_pad = (out_pad, out_pad)
        
    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + dilation[0]*(kernel_size[0]-1) + out_pad[0]+1
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + dilation[0]*(kernel_size[1]-1) + out_pad[1]+1
    
    return h, w

class NN_layer(nn.Module):
    def __init__(self,inp,out,act=nn.ReLU(),batch_norm=True,dropout=None):
        super(NN_layer,self).__init__()
        self.batch_norm=batch_norm
        self.layer=nn.ModuleList(
            [nn.Linear(inp,out)]+([nn.BatchNorm1d(out)] if self.batch_norm else [])+[act]+([nn.Dropout(dropout)] if dropout!=None else [])
            )
    def forward(self,x):
        for sl in self.layer:
            x=sl(x)
        return x


class NeuralNet(nn.Module):
    def __init__(self,layer_sizes=[300,150,50],activators=nn.LeakyReLU(),batch_norm=True,dropout=None):
        super(NeuralNet,self).__init__()
        self.layer_sizes=layer_sizes
        #self.activators=activators

        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(layer_sizes)-1)]

        if isinstance(dropout,float):
            self.dropout=[dropout for i in range(len(layer_sizes)-1)]
        else:
            self.dropout=[None for i in range(len(layer_sizes)-1)]

        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(layer_sizes)-1)]
        else:
            self.activators=activators

        self.layers=nn.ModuleList(
            [
                nn.Sequential(NN_layer(in_size,out_size,act,bat_norm,dropout))
                for in_size,out_size,act,bat_norm,dropout in zip(
                    self.layer_sizes[:-1],
                    self.layer_sizes[1:],
                    self.activators,
                    self.batch_norm,
                    self.dropout
                )
            ]
        )
    def forward(self,x):
        for l in self.layers:
            x=l(x)
        return x


class multi_head_Attenttion(nn.Module):
    def __init__(self,input_size,heads,head_size,Attention_mechanism,Dropout=0.5,return_dot=False):
        super(multi_head_Attenttion,self).__init__()
        # input size (p1*p2*c)
        self.head_size=head_size
        
        self.multi_qkv_heads=nn.Linear(input_size,heads*head_size*3)
        self.Attention_mechanism=Attention_mechanism
        self.last_layer=nn.Sequential(
            nn.Linear(heads*head_size,input_size),
            nn.Dropout(Dropout)
        )
    
    def forward(self,x): #[batch,(h w),(p1,p2,c)]
        qkv=self.multi_qkv_heads(x).chunk(3,dim=-1)  #([batch,(h w),(heads head_size)],[batch,(h w),(heads head_size)],[batch,(h w),(heads head_size)])
        q,k,v=map(lambda t: rearrange(t,'batch num_patches (heads head_size) -> batch heads num_patches head_size',head_size=self.head_size),qkv)
        dot,out=self.Attention_mechanism(q,k,v) #out={"attention_vector":Tensor,"dot":Tensor,...} 
        #out["attention_vector"] [batch,(h w),(heads head_size)]
        out=rearrange(out,"batch heads num_patches heads_size -> batch num_patches (heads heads_size)")
        return self.last_layer(out)

class Transformer(nn.Module):
    def __init__(self,Transformer_layers_sizes,Attention_mechanisms,layers_heads,layer_head_sizes,dropout=0.,batch_norm=False,mode="Encoding",interpolation_mode='nearest'):
        """
        Transformer_layers_sizes: list of representations of size [n+1]
        Attention_mechanisms: List of attention module of size [n]
        layers_heads: List of int of size [n]
        layer_head_sizes: List of int of size [n]
        dropout: List of float[0,1] of size [n]
        mode: "Encoding" or "Decoding"
        """
        super(Transformer,self).__init__()
        self.interpolation_mode=interpolation_mode
        

        self.inputs=Transformer_layers_sizes[:-1]
        self.outputs=Transformer_layers_sizes[1:]
        if isinstance(dropout,float) or dropout==None:
            self.dropout=[dropout for i in range(len(self.inputs))]
        else:
            self.dropout=dropout

        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(self.inputs))]
        else:
            self.batch_norm=batch_norm

        if mode=="Encoding":
            self.Transformer_layers_sizes=Transformer_layers_sizes
            
        elif mode=="Decoding":
            self.Transformer_layers_sizes=Transformer_layers_sizes[::-1]
            self.batch_norm[::-1]
            self.dropout[::-1]

        self.Attention_transformer_layers=nn.ModuleList([])
        self.MLP_transformer_layers=nn.ModuleList([])

        for input,output,Attention_mechanism,layers_head,layer_head_size,dropout,batch_norm in zip(self.inputs,self.outputs,Attention_mechanisms,layers_heads,layer_head_sizes,self.dropout,self.batch_norm):
            self.Attention_transformer_layers.append(
                #Normalization
                nn.Sequential(
                nn.LayerNorm(input),
                #Attention
                multi_head_Attenttion(
                    input_size=input,
                    heads=layers_head,
                    head_size=layer_head_size,
                    Attention_mechanism=Attention_mechanism,
                    Dropout=dropout
                )
                )
            )
            self.MLP_transformer_layers.append(
                #Normalization
                nn.Sequential(
                nn.LayerNorm(input),
                #MLP
                NeuralNet(
                    layer_sizes=[input,output],
                    activators=[nn.GELU(),nn.Identity()], #POSIBLE EXPERIMENT: varible middle activator
                    batch_norm=False,
                    dropout=dropout
                    )
                )
            )

    def forward(self,x):
        for attention_module,mlp_module,input,output in zip(self.Attention_transformer_layers,self.MLP_transformer_layers,self.inputs,self.outputs):
            x=attention_module(x)+x # batch num_patches (heads heads_size)
            if input!=output:
                x=mlp_module(x)+nn.functional.interpolate(x,output,mode=self.interpolation_mode)
            else:
                x=mlp_module(x)+x
        return x
            

