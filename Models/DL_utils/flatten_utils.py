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

dim_DNN_type=(lambda module: module.comp_layer[0].out_features)
dim_CNN_type=(lambda module: module.comp_layer[0].out_channels)


class hyb_view(object):
    def __init__(self,original_shape,cf,enc,dec,enc_dim_func,dec_dim_func):
        self.enc_dims=enc_dim_func(enc.im_layers[-1])
        self.dec_dims=dec_dim_func(dec.im_layers[0])
        if len(original_shape)==2:
            self.channel_shape=1
        else:
            self.channel_shape=original_shape[-3]
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
            out=rearrange(x,"b (c h w) -> b c h w",
                          c=self.dec_dims,
                          h=self.i_shape[-2]*(self.channel_shape//self.dec_dims),
                          w=self.i_shape[-1]*(self.channel_shape//self.dec_dims)
                          )
            #out=x.view([x.shape[0]]+self.i_shape)
        return out