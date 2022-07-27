import torch

from DL_utils.utils import NeuralNet


class Q_NET(NeuralNet):
    def __init__(self,CVAE_mode,xy_layer_size,xy_enc_activators,batch_norm=True,dropout=None,save_output=False,aux_dir=None,module_name=None):
        super(Q_NET,self).__init__(layer_sizes=[],activators=[])

        if CVAE_mode=="no_baseline":
            self.z_xy_mu=NeuralNet(xy_layer_size,xy_enc_activators,batch_norm=batch_norm,dropout=dropout)
            self.z_xy_sig=NeuralNet(xy_layer_size,xy_enc_activators,batch_norm=batch_norm,dropout=dropout)
    
    #def z_infer(self,x,In_ID=None):
    #    z_mu=self.z_x_mu(x,In_ID)
    #    z_sig=torch.exp(self.z_x_sig(x,In_ID))
    #    return z_mu,z_sig