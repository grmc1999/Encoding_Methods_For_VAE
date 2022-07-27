import torch
from DL_utils.utils import NeuralNet

class P_NET(NeuralNet):
    """
    if binomial estimation in y --> y_xz las activation function should be sigmoid
    """
    def __init__(self,CVAE_mode,y_layer_size,y_enc_activators,x_layer_size,x_enc_activators,batch_norm=True,dropout=None,save_output=False,aux_dir=None,module_name=None):
        super(P_NET,self).__init__(layer_sizes=[],activators=[])

        if CVAE_mode=="no_baseline":
            #self.y_xz_mu=NeuralNet(y_layer_size,y_enc_activators,save_output,aux_dir,module_name)
            self.y_xz=NeuralNet(y_layer_size,y_enc_activators,batch_norm=batch_norm,dropout=dropout)

            self.z_x_mu=NeuralNet(x_layer_size,x_enc_activators,batch_norm=batch_norm,dropout=dropout)
            self.z_x_sig=NeuralNet(x_layer_size,x_enc_activators,batch_norm=batch_norm,dropout=dropout)
    
    #def x_gener(self,z,In_ID=None):
    #    x_re=self.x_z(z,In_ID)
    #    return x_re