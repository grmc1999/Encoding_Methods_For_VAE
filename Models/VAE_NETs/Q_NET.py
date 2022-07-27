import torch

from DL_utils.utils import NeuralNet


class Q_NET(NeuralNet):
    def __init__(self,layer_size,enc_activators,batch_norm=False,dropout=None,save_output=False,aux_dir=None,module_name=None):
        super(Q_NET,self).__init__(layer_sizes=[],activators=[])
        self.z_x_mu=NeuralNet(layer_size,enc_activators,batch_norm,dropout)
        self.z_x_sig=NeuralNet(layer_size,enc_activators,batch_norm,dropout)
    
    def z_infer(self,x,In_ID=None):
        z_mu=self.z_x_mu(x,In_ID)
        z_sig=torch.exp(self.z_x_sig(x,In_ID))
        return z_mu,z_sig