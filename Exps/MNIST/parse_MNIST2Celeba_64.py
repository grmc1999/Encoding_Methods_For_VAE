import glob
import json
import argparse
import os
import numpy as np

def get_jsons(args):
    jsons_list=glob.glob(os.path.join(args.path,args.model,"*","*.json"))
    return jsons_list

def load_change_json(json_dir,args):
    
    config_json=json.load(open(json_dir))
    if args.model=="VAE_DNN":
        config_json["experiment_state"]="waiting"
        config_json["trainer"]["batch_size"]=2048
        config_json["trainer"]["use_cuda"]=True
        config_json["trainer"]["use_cuda"]='cuda:0'
        config_json["trainer"]["num_workers"]=10
        config_json["model"]["sub_modules"]["P_NET"]["parameters"]["batch_norm"]=False
        config_json["model"]["sub_modules"]["Q_NET"]["parameters"]["batch_norm"]=False
        config_json["model"]["sub_modules"]["P_NET"]["parameters"]["dec_activators"]["name"]="LeakyReLU"
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["batch_norm"]=False
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["batch_norm"]=False
        nl=len(config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["inp_sizes"])
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["activators"]["name"]=["LeakyReLU" for i in range(nl-2)]+["Sigmoid"]
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["activators"]["params"]=[{} for i in range(nl-1)]
    elif args.model=="VAE_CNN":
        config_json["experiment_state"]="waiting"
        config_json["trainer"]["batch_size"]=2048
        config_json["trainer"]["use_cuda"]=True
        config_json["trainer"]["use_cuda"]='cuda:0'
        config_json["trainer"]["num_workers"]=10
        nl=len(config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["repr_sizes"])
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["activators"]["name"]=["LeakyReLU" for i in range(nl-2)]+["Sigmoid"]
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["activators"]["params"]=[{} for i in range(nl-1)]

        # Size correction
        rep_dims = config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["repr_sizes"]
        base_stride=config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["stride"]
        base_stride=(base_stride[0] if isinstance(base_stride,list) else base_stride)
        min_rep=np.sum(28%(base_stride**np.arange(15))==0)-1
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["repr_sizes"]=[r if i<min_rep else rep_dims[min_rep] for i,r in enumerate(rep_dims)]
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["stride"]=[base_stride if i<min_rep else 1 for i,_ in enumerate(rep_dims[:-1])]

    elif args.model=="VAE_DNN_CNN":
        config_json["experiment_state"]="waiting"
        config_json["trainer"]["batch_size"]=2048
        config_json["trainer"]["use_cuda"]=True
        config_json["trainer"]["use_cuda"]='cuda:0'
        config_json["trainer"]["num_workers"]=10

        # Size correction
        rep_dims = config_json["model"]["sub_modules"]["encoding_decoding_module"]["module_type"]="Asymmetrical_CNN_DNN_EDM"

        rep_dims = config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["flat"]=False
        rep_dims = config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["deflat"]=True


        oid = config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["repr_sizes"][0]
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["repr_sizes"][0]=round(oid**0.5)**2

    elif args.model=="VAE_CNN_DNN":
        config_json["experiment_state"]="waiting"
        config_json["trainer"]["batch_size"]=2048
        config_json["trainer"]["use_cuda"]=True
        config_json["trainer"]["use_cuda"]='cuda:0'
        config_json["trainer"]["num_workers"]=10

        # Size correction
        rep_dims = config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["flat"]=True
        rep_dims = config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["deflat"]=False

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