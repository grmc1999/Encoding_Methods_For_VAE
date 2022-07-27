from .CNN import *
from torch import nn
import numpy as np


class set_ResNET_conv(nn.Module):
    def __init__(self,repr_sizes=1,kernel_sizes=5,bridge_kernel_size=1,act=nn.ReLU(),bridge_act=nn.ReLU(),lay_act=nn.ReLU(),batch_norm=True,dropout=None,stride=1):
        super(set_ResNET_conv, self).__init__()

        if not isinstance(kernel_sizes,list):
            kernel_sizes=[kernel_sizes,kernel_sizes]
        
        if not isinstance(repr_sizes,list):
            repr_sizes=[repr_sizes,repr_sizes,repr_sizes]

        if not isinstance(stride,list):
            stride=[stride,stride]

        #Define padding according kernel
        padding=((np.array(kernel_sizes)-1)/2).astype(int)

        if repr_sizes[0]==repr_sizes[2]: # alternative representations
            #set bridge
            self.shortcut=nn.ModuleList(
                [nn.Identity()]
            )

        elif repr_sizes[0]!=repr_sizes[2]:
            #if stride[0]!=1: # increase representations with low resolution
            bridge_stride=np.sum(np.array(stride)[np.array(stride)%2==0])
            if bridge_stride==0:
                bridge_stride=1

            bridge_padding=int((bridge_kernel_size-1)/2)
            self.shortcut=nn.ModuleList(
                [nn.Conv2d(repr_sizes[0],repr_sizes[2],kernel_size=bridge_kernel_size,stride=bridge_stride,padding=bridge_padding)]+\
                ([nn.BatchNorm2d(num_features=repr_sizes[2])] if batch_norm else [])+\
                [bridge_act]+\
                ([nn.Dropout(dropout)] if dropout!=None else [])
            )

        self.pre_layer=nn.ModuleList(
            [nn.Conv2d(repr_sizes[0],repr_sizes[1],kernel_size=kernel_sizes[0],stride=stride[0],padding=padding[0])]+\
            ([nn.BatchNorm2d(num_features=repr_sizes[1])] if batch_norm else [])+\
            [act]+\
            [nn.Conv2d(repr_sizes[1],repr_sizes[2],kernel_size=kernel_sizes[1],stride=stride[1],padding=padding[1])]+\
            ([nn.BatchNorm2d(num_features=repr_sizes[2])] if batch_norm else [])
        )

        self.lay_act=lay_act

    def forward(self,x):
        Fx=x.clone()
        for l in self.pre_layer:
            Fx=l(Fx)
        for l in self.shortcut:
            x=l(x)
        x=self.lay_act(Fx+x)
        return x


class set_ResNET_deconv(nn.Module):
    def __init__(self,repr_sizes=1,kernel_sizes=5,bridge_kernel_size=1,act=nn.ReLU(),bridge_act=nn.ReLU(),lay_act=nn.ReLU(),batch_norm=True,dropout=None,stride=1):
        super(set_ResNET_deconv,self).__init__()

        if not isinstance(kernel_sizes,list):
            kernel_sizes=[kernel_sizes,kernel_sizes]
        else:
            kernel_sizes=kernel_sizes[::-1]
        
        if not isinstance(repr_sizes,list):
            repr_sizes=[repr_sizes,repr_sizes,repr_sizes]
        else:
            repr_sizes=repr_sizes[::-1]

        if not isinstance(stride,list):
            stride=[stride,stride]
        else:
            stride=stride[::-1]

        #Define padding according kernel
        padding=((np.array(kernel_sizes)-1)/2).astype(int)
        
        if repr_sizes[0]==repr_sizes[2]: # alternative representations
            #set bridge
            self.shortcut=nn.ModuleList(
                [nn.Identity()]
            )

        elif repr_sizes[0]!=repr_sizes[2]:
            #if stride[0]!=1: # increase representations with low resolution
            bridge_stride=np.sum(np.array(stride)[np.array(stride)%2==0])
            if bridge_stride==0:
                bridge_stride=1

            bridge_padding=int((bridge_kernel_size-1)/2)
            self.shortcut=nn.ModuleList(
                [nn.ConvTranspose2d(repr_sizes[0],repr_sizes[2],kernel_size=bridge_kernel_size,stride=bridge_stride,padding=bridge_padding)]+\
                ([nn.BatchNorm2d(repr_sizes[2])] if batch_norm else [])+\
                [bridge_act]+\
                ([nn.Dropout(dropout)] if dropout!=None else [])
            )

        self.pre_layer=nn.ModuleList(
            [nn.ConvTranspose2d(repr_sizes[0],repr_sizes[1],kernel_size=kernel_sizes[0],stride=stride[0],padding=padding[0])]+\
            ([nn.BatchNorm2d(repr_sizes[1])] if batch_norm else [])+\
            [act]+\
            [nn.ConvTranspose2d(repr_sizes[1],repr_sizes[2],kernel_size=kernel_sizes[1],stride=stride[1],padding=padding[1])]+\
            ([nn.BatchNorm2d(repr_sizes[2])] if batch_norm else [])
        )

        self.lay_act=lay_act

    def forward(self,x):
        Fx=x.clone()
        for l in self.pre_layer:
            Fx=l(Fx)
        for l in self.shortcut:
            x=l(x)
        x=self.lay_act(Fx+x)
        return x


class ResNET_ENC(nn.Module):
    def __init__(self,repr_sizes=[[1,2,3],[3,4,5]],kernel_sizes=[[3,5],[5,5]],bridge_kernel_size=3,act=nn.ReLU(),bridge_act=nn.ReLU(),lay_act=nn.ReLU(),batch_norm=True,dropout=None,stride=[[1,1],[1,2]]):
        super(ResNET_ENC, self).__init__()
        #repr_size should be 
        self.repr_sizes=repr_sizes
        self.kernel_sizes=kernel_sizes
        self.stride=stride
        
        #main activators
        if isinstance(act,nn.Module):
            self.act=[act for i in range(len(repr_sizes))]
        else:
            self.act=act

        if isinstance(bridge_kernel_size,int):
            self.bridge_kernel_size=[bridge_kernel_size for i in range(len(repr_sizes))]
        else:
            self.bridge_kernel_size=bridge_kernel_size

        #bridge activators
        if isinstance(bridge_act,nn.Module):
            self.bridge_act=[bridge_act for i in range(len(repr_sizes))]
        else:
            self.bridge_act=bridge_act

        #layer activators
        if isinstance(lay_act,nn.Module):
            self.lay_act=[lay_act for i in range(len(repr_sizes))]
        else:
            self.lay_act=lay_act

        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(repr_sizes))]
        else:
            self.batch_norm=batch_norm
        
        if isinstance(dropout,float) or dropout==None:
            self.dropout=[dropout for i in range(len(repr_sizes))]
        else:
            self.dropout=dropout
        
        self.im_layers=nn.ModuleList(
            [
                set_ResNET_conv(
                    repr_sizes=repr_sizes,
                    kernel_sizes=kernel_sizes,
                    bridge_kernel_size=bridge_kernel_size,
                    act=act,
                    bridge_act=bridge_act,
                    lay_act=lay_act,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    stride=stride
                )
                for repr_sizes,kernel_sizes,bridge_kernel_size,act,bridge_act,lay_act,batch_norm,dropout,stride in zip(
                    self.repr_sizes,
                    self.kernel_sizes,
                    self.bridge_kernel_size,
                    self.act,
                    self.bridge_act,
                    self.lay_act,
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

class ResNET_DEC(nn.Module):
    def __init__(self,repr_sizes=[[1,2,3],[3,4,5]],kernel_sizes=[[3,5],[5,5]],bridge_kernel_size=3,act=nn.ReLU(),bridge_act=nn.ReLU(),lay_act=nn.ReLU(),batch_norm=True,dropout=None,stride=[[1,1],[1,2]]):
    #def __init__(self,repr_sizes=[32,64,128,256],kernel_size=5,activators=nn.ReLU(),pooling=True,batch_norm=True,dropout=None,stride=1):
        super(ResNET_DEC,self).__init__()
        self.repr_sizes=repr_sizes[::-1]
        self.kernel_sizes=kernel_sizes[::-1]
        self.stride=stride[::-1]
        
        #main activators
        if isinstance(act,nn.Module):
            self.act=[act for i in range(len(repr_sizes))]
        else:
            self.act=act[::-1]

        if isinstance(bridge_kernel_size,int):
            self.bridge_kernel_size=[bridge_kernel_size for i in range(len(repr_sizes))]
        else:
            self.bridge_kernel_size=bridge_kernel_size[::-1]

        #bridge activators
        if isinstance(bridge_act,nn.Module):
            self.bridge_act=[bridge_act for i in range(len(repr_sizes))]
        else:
            self.bridge_act=bridge_act[::-1]

        #layer activators
        if isinstance(lay_act,nn.Module):
            self.lay_act=[lay_act for i in range(len(repr_sizes))]
        else:
            self.lay_act=lay_act[::-1]

        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(repr_sizes))]
        else:
            self.batch_norm=batch_norm[::-1]
        
        if isinstance(dropout,float) or dropout==None:
            self.dropout=[dropout for i in range(len(repr_sizes))]
        else:
            self.dropout=dropout[::-1]
        
        self.im_layers=nn.ModuleList(
            [
                set_ResNET_deconv(
                    repr_sizes=repr_sizes,
                    kernel_sizes=kernel_sizes,
                    bridge_kernel_size=bridge_kernel_size,
                    act=act,
                    bridge_act=bridge_act,
                    lay_act=lay_act,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    stride=stride
                )
                for repr_sizes,kernel_sizes,bridge_kernel_size,act,bridge_act,lay_act,batch_norm,dropout,stride in zip(
                    self.repr_sizes,
                    self.kernel_sizes,
                    self.bridge_kernel_size,
                    self.act,
                    self.bridge_act,
                    self.lay_act,
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