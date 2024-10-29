import glob
import json
import argparse
import os

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

        #mn_rs=config_json["model"]["sub_modules"]["P_NET"]["parameters"]["layer_size"]
        #config_json["model"]["sub_modules"]["P_NET"]["parameters"]["layer_size"]=[int(nr*(218*178*3/784)) for nr in mn_rs]
        #mn_rs=config_json["model"]["sub_modules"]["Q_NET"]["parameters"]["layer_size"]
        #config_json["model"]["sub_modules"]["Q_NET"]["parameters"]["layer_size"]=[int(nr*(218*178*3/784)) for nr in mn_rs]

        #mn_rs=config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["inp_sizes"]
        #config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["inp_sizes"]=[int(nr*(218*178*3/784)) for nr in mn_rs]
        #mn_rs=config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["inp_sizes"]
        #config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["inp_sizes"]=[int(nr*(218*178*3/784)) for nr in mn_rs]
    if args.model=="VAE_CNN":
        config_json["experiment_state"]="waiting"
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