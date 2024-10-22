from torch import nn
from einops import rearrange

class s_view(nn.Module):
    def forward(self,x):
        if len(x.shape)==4:
            self.i_shape=x.shape
            out=x.view(x.shape[0],-1)
        elif len(x.shape)==2:
            out=x.view(self.i_shape)
        return out

dim_DNN_type=(lambda module: module.comp_layer[0].outfeatures)
dim_CNN_type=(lambda module: module.comp_layer[0].out_channels)


class hyb_view(object):
    def __init__(self,original_shape,cf,model,enc_dim_func,dec_dim_func):
        self.enc_dims=enc_dim_func(model.EncoderDecoder.ENC.im_layers[-1])
        self.dec_dims=dec_dim_func(model.EncoderDecoder.DEC.im_layers[0])
        original_shape[-1]=int(original_shape[-1]*cf)
        original_shape[-2]=int(original_shape[-2]*cf)
        self.i_shape=original_shape
    def __call__(self,x):
        print(x.shape)
        if len(x.shape)==4: # batch c w h
            self.i_shape=x.shape
            #out=x.view(x.shape[0],-1)
            out = rearrange(x,"b c h w -> b (c h w)")
        elif len(x.shape)==2: # batch layer_dim
            out=rearrange(x,"b (c h w) -> b c h w",c=self.dec_dims,h=self.i_shape[-2],w=self.i_shape[-1])
            #out=x.view([x.shape[0]]+self.i_shape)
        return out