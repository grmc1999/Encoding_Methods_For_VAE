import glob
import json
import argparse
import os
import numpy as np


image_size_x=64
image_size_y=64
def get_jsons(args):
    jsons_list=glob.glob(os.path.join(args.path,args.model,"*","*.json"))
    return jsons_list

def load_change_json(json_dir,args):
    
    config_json=json.load(open(json_dir))
    if args.model=="VAE_DNN":
        config_json["experiment_state"]="waiting"
        config_json["trainer"]["batch_size"]=64
        config_json["trainer"]["use_cuda"]=True
        config_json["trainer"]["in_device"]='cuda:0'
        config_json["trainer"]["num_workers"]=10

        config_json["transforms"]={
            "Tuple_to_dict": {},
            "MultiInputToTensor": {
                  "metadata": []
            },
            "Size_Normalization":{
                  "normalized_size":[image_size_x,image_size_y]
            }
        }

        mn_rs=config_json["model"]["sub_modules"]["P_NET"]["parameters"]["layer_size"]
        config_json["model"]["sub_modules"]["P_NET"]["parameters"]["layer_size"]=[int(nr*(image_size_x*image_size_y*3/784)) for nr in mn_rs]
        mn_rs=config_json["model"]["sub_modules"]["Q_NET"]["parameters"]["layer_size"]
        config_json["model"]["sub_modules"]["Q_NET"]["parameters"]["layer_size"]=[int(nr*(image_size_x*image_size_y*3/784)) for nr in mn_rs]

        mn_rs=config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["inp_sizes"]
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["inp_sizes"]=[int(nr*(image_size_x*image_size_y*3/784)) for nr in mn_rs]
        mn_rs=config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["inp_sizes"]
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["inp_sizes"]=[int(nr*(image_size_x*image_size_y*3/784)) for nr in mn_rs]

        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["input_size"]=[3,image_size_x,image_size_y]
    elif args.model=="VAE_CNN":
        config_json["experiment_state"]="waiting"
        config_json["trainer"]["batch_size"]=64
        config_json["trainer"]["use_cuda"]=True
        config_json["trainer"]["in_device"]='cuda:0'
        #config_json["trainer"]["epochs"]=2
        config_json["trainer"]["num_workers"]=10

        config_json["transforms"]={
            "Tuple_to_dict": {},
            "MultiInputToTensor": {
                  "metadata": []
            },
            "Size_Normalization":{
                  "normalized_size":[image_size_x,image_size_y]
            }
        }

        #nl=len(config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["repr_sizes"])
        #config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["activators"]["name"]=["LeakyReLU" for i in range(nl-2)]+["Sigmoid"]
        #config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["activators"]["params"]=[{} for i in range(nl-1)]

        mn_rs=config_json["model"]["sub_modules"]["P_NET"]["parameters"]["layer_size"]
        config_json["model"]["sub_modules"]["P_NET"]["parameters"]["layer_size"]=[int(nr*(image_size_x*image_size_y*3/784)) for nr in mn_rs]
        mn_rs=config_json["model"]["sub_modules"]["Q_NET"]["parameters"]["layer_size"]
        config_json["model"]["sub_modules"]["Q_NET"]["parameters"]["layer_size"]=[int(nr*(image_size_x*image_size_y*3/784)) for nr in mn_rs]
        
        # Size correction
        rep_dims = config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["repr_sizes"]
        base_stride=config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["stride"]
        base_stride=(base_stride[0] if isinstance(base_stride,list) else base_stride)
        min_rep=np.sum(min(image_size_x,image_size_y)%(base_stride**np.arange(25))==0)-1
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["repr_sizes"]=[r if i<min_rep else rep_dims[min_rep] for i,r in enumerate(rep_dims)]
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["stride"]=[base_stride if i<min_rep else 1 for i,_ in enumerate(rep_dims[:-1])]

        config_json["model"]["model_params"]["resize"]=[image_size_x, image_size_y]

    elif args.model=="VAE_DNN_CNN":
        config_json["experiment_state"]="waiting"
        config_json["trainer"]["batch_size"]=64
        config_json["trainer"]["use_cuda"]=True
        config_json["trainer"]["in_device"]='cuda:0'
        #config_json["trainer"]["epochs"]=2
        config_json["trainer"]["num_workers"]=10

        config_json["transforms"]={
            "Tuple_to_dict": {},
            "MultiInputToTensor": {
                  "metadata": []
            },
            "Size_Normalization":{
                  "normalized_size":[image_size_x,image_size_y]
            }
        }

        # Size correction
        rep_dims = config_json["model"]["sub_modules"]["encoding_decoding_module"]["module_type"]="Asymmetrical_CNN_DNN_EDM"


        mn_rs=config_json["model"]["sub_modules"]["P_NET"]["parameters"]["layer_size"]
        config_json["model"]["sub_modules"]["P_NET"]["parameters"]["layer_size"]=[int(nr*(image_size_x*image_size_y*3/784)) for nr in mn_rs]
        mn_rs=config_json["model"]["sub_modules"]["Q_NET"]["parameters"]["layer_size"]
        config_json["model"]["sub_modules"]["Q_NET"]["parameters"]["layer_size"]=[int(nr*(image_size_x*image_size_y*3/784)) for nr in mn_rs]

        mn_rs=config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["inp_sizes"]
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["inp_sizes"]=[int(nr*(image_size_x*image_size_y*3/784)) for nr in mn_rs]

        rep_dims = config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["flat"]=False
        rep_dims = config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["deflat"]=True


        oid = config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["repr_sizes"][0]
        #config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["repr_sizes"][0]=round(oid**0.5)**2

        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["i_shape"]=[3,image_size_x,image_size_y]
        
        c_shape=config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["i_shape"]
        cf=config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["compression_factor"]
        new_channel=int((round(oid**0.5)**2))
        config_json["model"]["sub_modules"]["P_NET"]["parameters"]["layer_size"][-1]=int(new_channel*int(c_shape[-2]*cf/(new_channel**0.5))*int(c_shape[-1]*cf/(new_channel**0.5)))

        config_json["model"]["model_params"]["resize"]=[image_size_x, image_size_y]
    elif args.model=="VAE_CNN_DNN":
        config_json["experiment_state"]="waiting"
        config_json["trainer"]["batch_size"]=64
        config_json["trainer"]["use_cuda"]=True
        config_json["trainer"]["in_device"]='cuda:0'
        #config_json["trainer"]["epochs"]=2
        config_json["trainer"]["num_workers"]=10

        config_json["transforms"]={
            "Tuple_to_dict": {},
            "MultiInputToTensor": {
                  "metadata": []
            },
            "Size_Normalization":{
                  "normalized_size":[image_size_x,image_size_y]
            }
        }

        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["input_size"]=[3,image_size_x,image_size_y]

        mn_rs=config_json["model"]["sub_modules"]["P_NET"]["parameters"]["layer_size"]
        config_json["model"]["sub_modules"]["P_NET"]["parameters"]["layer_size"]=[int(nr*(image_size_x*image_size_y*3/784)) for nr in mn_rs]
        mn_rs=config_json["model"]["sub_modules"]["Q_NET"]["parameters"]["layer_size"]
        config_json["model"]["sub_modules"]["Q_NET"]["parameters"]["layer_size"]=[int(nr*(image_size_x*image_size_y*3/784)) for nr in mn_rs]

        mn_rs=config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["inp_sizes"]
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["inp_sizes"]=[int(nr*(image_size_x*image_size_y*3/784)) for nr in mn_rs]

        # Size correction
        rep_dims = config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["flat"]=True
        rep_dims = config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["deflat"]=False

        config_json["model"]["model_params"]["resize"]=[image_size_x, image_size_y]
    else:
        print("type not found")
    
    f=open(json_dir,"w")
    json.dump(config_json,f,indent=6)
    f.close()
def change_json(args):
    jsons_dirs=get_jsons(args)
    for json_dir in jsons_dirs[:]:
        print("modifiying: "+json_dir)
        load_change_json(json_dir,args)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Send API')
    parser.add_argument('--model', '-m', help='model_type_name',type=str, required=True)
    parser.add_argument('--path', '-p', help='environment_specific_path',type=str, required=True)
    args = parser.parse_args()
    change_json(args)