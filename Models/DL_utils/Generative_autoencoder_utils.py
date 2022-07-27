from .utils import NeuralNet

class Base_Generative_AutoEncoder(NeuralNet):
  """
  prior relaxed without baseline
  to generate normal distributions
  """
  def __init__(self,Encoder_Decoder_Module,P_NET,Q_NET,losses_weigths={},save_output=False,aux_dir=None,module_name=None):
    super(Base_Generative_AutoEncoder,self).__init__(layer_sizes=[],activators=[],batch_norm=False,dropout=None)
    self.losses={}
    self.losses_weigths=losses_weigths
    
    #Encoding decoding method
    #CONSIDER ERASING ATTRIBUTE SINCE IS ALWAYS VARIABLE
    self.EncoderDecoder=Encoder_Decoder_Module

    #Generative networks
    self.P=P_NET

    #Inference networks
    self.Q=Q_NET