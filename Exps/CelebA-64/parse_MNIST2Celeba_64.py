import glob
import json
import argparse
import os

def get_jsons(args):
    jsons_list=glob.glob(os.path.join(".",args.model,"*","*.json"))
    return jsons_list

def load_change_json(json_dir,args):
    
    config_json=json.load(open(json_dir))
    if args.model=="VAE_DNN":
        config_json["trainer"]["batch_size"]=1024
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["input_size"]=[3,218,178]
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["encoder_parameters"]["inp_sizes"][0]=116412
        config_json["model"]["sub_modules"]["encoding_decoding_module"]["parameters"]["decoder_parameters"]["inp_sizes"][-1]=116412
    else:
        print("type not found")
    
    f=open(json_dir,"w")
    json.dump(config_json,f,indent=6)
    f.close()
def change_json(args):
    jsons_dirs=get_jsons(args)
    for json_dir in jsons_dirs[:]:
        print("modifiying: "+json_dir)
        load_change_json(json_dir)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Send API')
    parser.add_argument('--model', '-m', help='model_type_name',type=str, required=True)
    args = parser.parse_args()
    change_json(args)