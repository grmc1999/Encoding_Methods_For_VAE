import torch
from torch import nn

class set_Max_Pool_conv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out,kernel_size=5,act=nn.ReLU(),pool=True,batch_norm=True,dropout=None,stride=1):
        super(set_Max_Pool_conv, self).__init__()
        self.stride=stride
        if stride==1:
            self.padding=0
        elif stride==2:
            self.padding=int((kernel_size-1)/2)
        self.pool=pool

        if pool:
            #pool_layer=[nn.MaxPool2d(kernel_size=kernel_size,stride=self.stride,padding=self.padding,return_indices=True)]
            pool_layer=[nn.MaxPool2d(kernel_size=kernel_size,stride=kernel_size,padding=self.padding,return_indices=True)]
        else:
            pool_layer=[]
        self.comp_layer=nn.ModuleList(
            [nn.Conv2d(repr_size_in,repr_size_out,kernel_size=kernel_size,stride=self.stride,padding=self.padding)]+\
            ([nn.BatchNorm2d(repr_size_out)] if batch_norm else [])+\
            [act]+\
            ([nn.Dropout(dropout)] if dropout!=None else [])+\
            (pool_layer)
        )

    def forward(self,x):
        for l in self.comp_layer[:-2]:
            x=l(x)
        if self.pool:
            self.Sidx=x.shape
            x,idx=self.comp_layer[-1](x)
            self.idx=idx
        else:
            self.Sidx=None
            x=self.comp_layer[-1](x)
            self.idx=None
        return x

class set_Max_Unpool_conv(nn.Module):
    def __init__(self,repr_size_in,repr_size_out,kernel_size=5,act=nn.ReLU(),pool=True,batch_norm=True,dropout=None,stride=1):
        super(set_Max_Unpool_conv, self).__init__()
        self.stride=stride
        if stride==1:
            self.padding=0
            self.out_pad=0
        elif stride==2:
            self.padding=int((kernel_size-1)/2)
            self.out_pad=1
        self.pool=pool

        if self.pool:
            #pool_layer=[nn.MaxUnpool2d(kernel_size=kernel_size,stride=self.stride,padding=self.padding)]
            pool_layer=[nn.MaxUnpool2d(kernel_size=kernel_size,stride=kernel_size,padding=self.padding)]
        else:
            pool_layer=[]

        self.comp_layer=nn.ModuleList(
            (pool_layer)+\
            [nn.ConvTranspose2d(repr_size_in,repr_size_out,kernel_size=kernel_size,stride=self.stride,padding=self.padding,output_padding=self.out_pad)]+\
            ([nn.BatchNorm2d(repr_size_out)] if batch_norm else [])+\
            [act]+\
            ([nn.Dropout(dropout)] if dropout!=None else [])
        )

    def forward(self,x,idx,Sidx):
        if self.pool:
            x=self.comp_layer[0](x,idx,output_size=Sidx)
            for l in self.comp_layer[1:]:
                x=l(x)
        else:
            for l in self.comp_layer:
                x=l(x)
        
        return x

class Max_Pool_encoder_conv(nn.Module):
    def __init__(self,repr_sizes=[3,32,64,128,256],
                kernel_size=5,activators=nn.ReLU(),pool=True,batch_norm=True,dropout=None,stride=1):
        super(Max_Pool_encoder_conv, self).__init__()
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
        if isinstance(pool,bool):
            self.pool=[pool for i in range(len(repr_sizes)-1)]
        else:
            self.pool=pool
        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(repr_sizes)-1)]
        else:
            self.batch_norm=batch_norm
        
        if isinstance(dropout,float) or dropout==None:
            self.dropout=[dropout for i in range(len(repr_sizes)-1)]
        else:
            self.dropout=dropout
        self.idx_list=[]
        self.Sidx_list=[]
        
        self.im_layers=nn.ModuleList(
            [
                set_Max_Pool_conv(
                    repr_size_in,
                    repr_size_out,
                    kernel_size,
                    act,
                    pool,
                    batch_norm,
                    dropout,
                    stride,
                )
                for repr_size_in,repr_size_out,kernel_size,act,pool,batch_norm,dropout,stride in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:],
                    self.kernels,
                    self.activators,
                    self.pool,
                    self.dropout,
                    self.batch_norm,
                    self.stride
                )
            ]
        )
    def forward(self,x):
        self.idx_list=[]
        self.Sidx_list=[]
        for l in self.im_layers:
            x=l(x)
            self.idx_list.append(l.idx)
            self.Sidx_list.append(l.Sidx)
        return x
    
class Max_Unpool_decoder_conv(nn.Module):
    def __init__(self,repr_sizes=[32,64,128,256],
                kernel_size=5,activators=nn.ReLU(),pool=True,batch_norm=True,dropout=None,stride=1):
        super(Max_Unpool_decoder_conv,self).__init__()
        self.repr_sizes=repr_sizes
        self.repr_sizes=self.repr_sizes[::-1]
        self.stride=[stride for i in range(len(repr_sizes)-1)]
        
        #kernels
        if isinstance(kernel_size,int):
            self.kernels=[kernel_size for i in range(len(repr_sizes)-1)][::-1]
        else:
            self.kernels=kernel_size[::-1]
        #activators
        if isinstance(activators,nn.Module):
            self.activators=[activators for i in range(len(repr_sizes)-1)][::-1]
        else:
            self.activators=activators[::-1]
        #pooling
        if isinstance(pool,bool):
            self.pool=[pool for i in range(len(repr_sizes)-1)][::-1]
        else:
            self.pool=pool[::-1]
        #batch_norm
        if isinstance(batch_norm,bool):
            self.batch_norm=[batch_norm for i in range(len(repr_sizes)-1)][::-1]
        else:
            self.batch_norm=batch_norm[::-1]

        if isinstance(dropout,float) or dropout==None:
            self.dropout=[dropout for i in range(len(repr_sizes)-1)][::-1]
        else:
            self.dropout=dropout[::-1]
        
        self.im_layers=nn.ModuleList(
            [
                set_Max_Unpool_conv(
                    repr_size_in,
                    repr_size_out,
                    kernel_size,
                    act,
                    pool,
                    batch_norm,
                    dropout,
                    stride
                )
                for repr_size_in,repr_size_out,kernel_size,act,pool,batch_norm,dropout,stride in zip(
                    self.repr_sizes[:-1],
                    self.repr_sizes[1:],
                    self.kernels,
                    self.activators,
                    self.pool,
                    self.batch_norm,
                    self.dropout,
                    self.stride
                )
            ]
        )
    def forward(self,x,idx_list,Sidx_list):
        for l,idx,Sidx in zip(self.im_layers,idx_list,Sidx_list):
            x=l(x,idx,Sidx)
        return x
