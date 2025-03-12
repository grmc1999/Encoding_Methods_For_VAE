#from DL_utils.utils import *
#from DL_utils.Attention_methods import *
#from DL_utils.ViT import *
#
#
#class ViT_EDM(nn.Module):
#  def __init__(self,image_shape,patch_shape,Transformer_layers_sizes,Attention_mechanisms,layers_heads,layer_head_sizes,dropout,batch_norm,pre_batch_norm,pre_activator,pre_dropout):
#    super(ViT_EDM,self).__init__()
#    #Encoding modules
#    self.ENC=ViT_ENC(
#        image_shape=image_shape,
#        patch_shape=patch_shape,
#        Transformer_layers_sizes=Transformer_layers_sizes,
#        Attention_mechanisms=Attention_mechanisms,
#        layers_heads=layers_heads,
#        layer_head_sizes=layer_head_sizes,
#        dropout=dropout,
#        batch_norm=batch_norm,
#        pre_batch_norm=pre_batch_norm,
#        pre_activator=pre_activator,
#        pre_dropout=pre_dropout
#    )
#    #Encoding modules
#    self.DEC=ViT_DEC(
#        image_shape=image_shape,
#        patch_shape=patch_shape,
#        Transformer_layers_sizes=Transformer_layers_sizes,
#        Attention_mechanisms=Attention_mechanisms,
#        layers_heads=layers_heads,
#        layer_head_sizes=layer_head_sizes,
#        dropout=dropout,
#        batch_norm=batch_norm,
#        pre_batch_norm=pre_batch_norm,
#        pre_activator=pre_activator,
#        pre_dropout=pre_dropout
#    )
#    #flatten
#    self.fl=s_view()
#    del self.DEC.Positional_embedding
#    del self.DEC.class_tokens
#    self.DEC.Positional_embedding=self.ENC.Positional_embedding
#    self.DEC.class_tokens=self.ENC.class_tokens
#
#  def sanity_check(self,x):
#    ex=self.ENC(x)
#    ex=self.fl(ex).shape
#    return ex
#
#  def Encoding(self,x):
#    ex=self.ENC(x)
#    ex=self.fl(ex)
#    return ex
#    
#  def Decoding(self,z):
#    dz=self.fl(z)
#    dz=self.DEC(dz)
#    return dz