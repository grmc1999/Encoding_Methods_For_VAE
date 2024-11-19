from torch import nn
import torch
from einops.layers.torch import Rearrange
from einops import repeat
from .utils import Transformer
import Attention_methods


#if deepViT
#class ViT_enc_layrer(nn.Module):
#class ViT_dec_layrer(nn.Module):

class ViT_ENC(nn.Module):
    def __init__(self,image_shape,patch_shape,Transformer_layers_sizes,Attention_mechanisms,layers_heads,layer_head_sizes,dropout,batch_norm,pre_batch_norm,pre_activator=nn.Identity(),pre_dropout=0.5):
        """
        image_shape: withd heitgh and channels
        """
        super(ViT_ENC,self).__init__()

        inputs=Transformer_layers_sizes
        
        assert not(image_shape[0]%patch_shape[0]==0 and image_shape[1]%patch_shape[1]==0), "image shape is not divisible by patch shape"

        self.num_patches=(image_shape[0]//patch_shape[0]) * (image_shape[1]//patch_shape[1])

        self.image_patch_embedding=nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)',ph=patch_shape[0],pw=patch_shape[1]),
            nn.Linear(patch_shape[0]*patch_shape[1]*image_shape[2],inputs[0])
        )
        self.Positional_embedding=nn.Parameter(torch.randn(1,self.num_patches+1,inputs[0]))
        self.class_tokens=nn.Parameter(torch.randn(1,1,inputs[0]))
        self.Pre_Transformer=nn.ModuleList(
            #Batchnorm
            ([nn.BatchNorm1d(self.num_patches+1)] if pre_batch_norm else [])+\
            #Pre_Activator
            ([pre_activator])+\
            #Dropout
            ([nn.Dropout(pre_dropout)])
        )

        self.Transformer=Transformer(
            Transformer_layers_sizes,
            #Attention_mechanisms=getattr(Attention_methods,Attention_mechanisms)(),
            getattr(Attention_methods,Attention_mechanisms)(),
            layers_heads,
            layer_head_sizes,
            dropout=dropout,
            batch_norm=batch_norm,
            mode="Encoding"
        )


    def forward(self,x):
        #patch embedding
        x=self.image_patch_embedding(x)
        #Tokenization
        x=torch.cat((x,repeat(self.class_tokens,"1 1 d->b 1 d",b=x.shape[0])),dim=1)
        x=x+self.Positional_embedding[:,:]
        #PreLayer
        for layer in self.Pre_Transformer:
            x=layer(x)
        #Transformer
        x=self.Transformer(x)
        return x



class ViT_DEC(nn.Module):
    def __init__(self,image_shape,patch_shape,Transformer_layers_sizes,Attention_mechanisms,layers_heads,layer_head_sizes,dropout,batch_norm,pre_batch_norm,pre_activator=nn.Identity(),pre_dropout=0.5):
        """
        image_shape: withd heitgh and channels
        """
        super(ViT_DEC,self).__init__()

        inputs=Transformer_layers_sizes[:-1]
        
        assert image_shape[0]%patch_shape[0]==0 and image_shape[1]%patch_shape[1]==0, "image shape is not divisible by patch shape"

        self.num_patches=(image_shape[0]//patch_shape[0]) * (image_shape[1]//patch_shape[1])

        self.image_patch_embedding=nn.Sequential(
            nn.Linear(inputs[0],patch_shape[0]*patch_shape[1]*image_shape[2]),
            Rearrange('b (h w) (ph pw c) -> b c (h ph) (w pw)',ph=patch_shape[0],pw=patch_shape[1],h=image_shape[0]//patch_shape[0],w=image_shape[1]//patch_shape[1])
            
        )
        self.Positional_embedding=nn.Parameter(torch.randn(1,self.num_patches+1,inputs[0]))
        self.class_tokens=nn.Parameter(torch.randn(1,1,inputs[0]))
        self.Pre_Transformer=nn.ModuleList(
            #Batchnorm
            ([nn.BatchNorm1d(self.num_patches+1)] if pre_batch_norm else [])+\
            #Pre_Activator
            ([pre_activator])+\
            #Dropout
            ([nn.Dropout(pre_dropout)])
        )

        self.Transformer=Transformer(
            Transformer_layers_sizes[::-1],
            Attention_mechanisms[::-1],
            layers_heads[::-1],
            layer_head_sizes[::-1],
            dropout=dropout,
            batch_norm=batch_norm,
            mode="Decoding"
        )


    def forward(self,x):
        x=self.Transformer(x) #[Batch, num_patches+1, head]

        for layer in self.Pre_Transformer:
            x=layer(x) #[Batch, num_patches+1, head]
        #patch embedding
        
        x=x-self.Positional_embedding[:,:]
        #Detokenization
        x,est_class_token=torch.split(x,self.num_patches,dim=1)

        self.est_class_token=est_class_token

        x=self.image_patch_embedding(x)

        
        
        return x