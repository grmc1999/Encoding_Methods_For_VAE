import torch
from torch import nn

class set_conv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out,kernel_size=5,act=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1):
        super(set_conv, self).__init__()
        self.stride=stride
        if stride==1:
            self.padding=0
        elif stride==2:
            self.padding=int((kernel_size-1)/2)

        self.comp_layer=nn.ModuleList(
            [nn.Conv2d(repr_size_in,repr_size_out,kernel_size=kernel_size,stride=self.stride,padding=self.padding)]+\
            ([nn.BatchNorm2d(repr_size_out)] if batch_norm else [])+\
            [act]+\
            ([nn.Dropout(dropout)] if dropout!=None else [])+\
            ([nn.MaxPool2d(kernel_size=kernel_size,stride=self.stride,padding=self.padding)] if pooling else [])
        )

    def forward(self,x):
        for l in self.comp_layer:
            x=l(x)
        return x

class set_deconv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out,kernel_size=5,act=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1):
        super(set_deconv, self).__init__()
        self.stride=stride
        if stride==1:
            self.padding=0
            self.out_pad=0
        elif stride==2:
            self.padding=int((kernel_size-1)/2)
            self.out_pad=1

        self.comp_layer=nn.ModuleList(
            [nn.ConvTranspose2d(repr_size_in,repr_size_out,kernel_size=kernel_size,stride=self.stride,padding=self.padding,output_padding=self.out_pad)]+\
            ([nn.BatchNorm2d(repr_size_out)] if batch_norm else [])+\
            [act]+\
            ([nn.Dropout(dropout)] if dropout!=None else [])+\
            ([nn.MaxUnpool2d(kernel_size=kernel_size,stride=self.stride,padding=self.padding)] if pooling else [])
        )
    def forward(self,x):
        for l in self.comp_layer:
            x=l(x)
        return x


class b_encoder_conv(nn.Module):
    def __init__(self,repr_sizes=[3,32,64,128,256],
                kernel_size=5,activators=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1):
        super(b_encoder_conv, self).__init__()
        self.repr_sizes=repr_sizes
        self.stride=[stride for i in range(len(repr_sizes)-1)]
        
        #kernels
        if isinstance(kernel_size,int):
            self.kernels=[kernel_size for i in range(len(repr_sizes)-1)]
        else:
            self.kernels=kernel_size
        #activators
        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(repr_sizes)-1)]
        else:
            self.activators=activators
        #pooling
        if isinstance(pooling,bool):
            self.pooling=[pooling for i in range(len(repr_sizes)-1)]
        else:
            self.pooling=pooling
        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(repr_sizes)-1)]
        else:
            self.batch_norm=batch_norm
        
        if isinstance(dropout,float) or dropout==None:
            self.dropout=[dropout for i in range(len(repr_sizes)-1)]
        else:
            self.dropout=dropout
        
        self.im_layers=nn.ModuleList(
            [
                set_conv(repr_in,
                repr_out,
                kernel_size,
                act,
                pooling,
                batch_norm,
                dropout,
                stride
                )
                for repr_in,repr_out,kernel_size,act,pooling,batch_norm,dropout,stride in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:],
                    self.kernels,
                    self.activators,
                    self.pooling,
                    self.dropout,
                    self.batch_norm,
                    self.stride
                )
            ]
        )
    def forward(self,x):
        for l in self.im_layers:
            x=l(x)
        return x
    
class b_decoder_conv(nn.Module):
    def __init__(self,repr_sizes=[32,64,128,256],
                kernel_size=5,activators=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1):
        super(b_decoder_conv,self).__init__()
        self.repr_sizes=repr_sizes
        self.repr_sizes=self.repr_sizes[::-1]
        self.stride=[stride for i in range(len(repr_sizes)-1)][::-1]
        
        #kernels
        if isinstance(kernel_size,int):
            self.kernels=[kernel_size for i in range(len(repr_sizes)-1)]
        else:
            self.kernels=kernel_size[::-1]
        #activators
        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(repr_sizes)-1)]
        else:
            self.activators=activators[::-1]
        #pooling
        if isinstance(pooling,bool):
            self.pooling=[pooling for i in range(len(repr_sizes)-1)]
        else:
            self.pooling=pooling[::-1]
        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(repr_sizes)-1)]
        else:
            self.batch_norm=batch_norm[::-1]

        if isinstance(dropout,float) or dropout==None:
            self.dropout=[dropout for i in range(len(repr_sizes)-1)]
        else:
            self.dropout=dropout[::-1]
        
        self.im_layers=nn.ModuleList(
            [
                set_deconv(repr_in,
                repr_out,
                kernel_size,
                act,
                pooling,
                batch_norm,
                dropout,
                stride
                )
                for repr_in,repr_out,kernel_size,act,pooling,batch_norm,dropout,stride in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:],
                    self.kernels,
                    self.activators,
                    self.pooling,
                    self.batch_norm,
                    self.dropout,
                    self.stride
                )
            ]
        )
    def forward(self,x):
        for l in self.im_layers:
            x=l(x)
        return x