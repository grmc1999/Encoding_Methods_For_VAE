

from DL_utils.utils import NeuralNet


class P_NET(NeuralNet):
    def __init__(self,layer_size,dec_activators,batch_norm=False,dropout=None,save_output=False,aux_dir=None,module_name=None):
        super(P_NET,self).__init__(layer_sizes=[],activators=[])
        self.x_z=NeuralNet(layer_size,dec_activators,batch_norm,dropout)
    
    def x_gener(self,z,In_ID=None):
        x_re=self.x_z(z,In_ID)
        return x_re