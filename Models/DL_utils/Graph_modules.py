from torch import nn
from .Graph_layers import *

class WL_GNN_ENC(nn.Module):
    def __init__(self,attribute_dims=[5,2,1]):
        super(WL_GNN_ENC, self).__init__()
        #repr_size should be 
        self.attribute_dims=attribute_dims
        
        self.graph_layers=nn.ModuleList(
            [
                Conv_WL_GNN(
                    conv_in=inp,
                    conv_out=out
                )
                for inp,out in zip(
                    self.attribute_dims[:-1],
                    self.attribute_dims[1:]
                )
            ]
        )

    def forward(self,x,edge_index):
        for l in self.graph_layers:
            x=l(x=x,edge_index=edge_index)
        return x