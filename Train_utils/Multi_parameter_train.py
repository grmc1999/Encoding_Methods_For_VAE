import os
import sys
import copy
import numpy as np
import json
from tqdm import tqdm
from tqdm.notebook import tqdm
from torchvision import transforms
import torch
from torch import nn
import traceback

sys.path.append(os.path.join("..","Dataset_utils"))
#from Dataset_utils import Custom_Transforms
import Custom_Transforms
#from Dataset_utils import poses_parser
#from Dataset_utils.DataLoader import UAV_GPS_Dataset

sys.path.append(os.path.join("..","Models"))
#from Models import DL_utils
#from Models import Encoding_Decoding_modules
import Models
import DL_utils
import Encoding_Decoding_modules


from TT_class import trainer

class multi_parameter_training(trainer):
    def __init__(self,results_directory,dataset_root_directory,Dataset_type,train=True,test=True,K_fold_training=None,visualization=False,split_frac=0.8):

        self.Dataset_type=Dataset_type
        self.datasets={}
        self.Compose_trans=None
        self.split_frac=split_frac
        self.results_directory=results_directory
        self.dataset_root_directory=dataset_root_directory
        self.train=train
        self.test=test
        self.K_fold_training=K_fold_training

        self.test_dirs=np.array(os.listdir(self.results_directory))
        self.test_dirs=np.vectorize(lambda td:os.path.join(self.results_directory,td))(self.test_dirs)
        #self.test_dirs=np.vectorize(lambda td:os.path.join(td,"config.json"))(self.test_dirs)
    
    
    def prepare_transforms(self,transform_args):
        trans_seq=list(transform_args.keys())
        if len(trans_seq)>0:
            instantiated_trans_seq=[]
            for t in trans_seq:
                instantiated_trans_seq.append(getattr(Custom_Transforms,t)(**(transform_args[t])))
            self.Compose_trans=transforms.Compose(instantiated_trans_seq)
        else:
            self.Compose_trans=None
 
    def split_dataset(self,dataset,test_dir):
        train_s= int((len(dataset))*self.split_frac)
        test_s=int((len(dataset))-train_s)

        if "data_split.pkl" in os.listdir(self.results_directory):
            print("data split found")
            dict=self.load_dict(os.path.join(self.results_directory,"data_split.pkl"))
            train_index=dict["train_index"]
            test_index=dict["test_index"]
        else:
            #NON_DEBUG
            train_index,test_index=torch.utils.data.random_split(range(len(dataset)),[train_s, test_s])
            #DEBUG
            #train_index,test_index=torch.utils.data.random_split(range(30),[train_s, test_s])
            dataset_split_index={
                "train_index":train_index,
                "test_index":test_index
            }
            self.save_dict(dataset_split_index,os.path.join(test_dir,"data_split.pkl"))

        self.datasets["train_set"] = torch.utils.data.Subset(dataset, train_index)
        self.datasets["test_set"] = torch.utils.data.Subset(dataset, test_index)

    def set_datasets(self,test_dir,split=True):

        dataset=self.Dataset_type(
            self.dataset_root_directory,
            transform=self.Compose_trans,
            target_transform=self.Compose_trans.transforms[0]
            )
        if split:
            self.split_dataset(dataset,test_dir)
        else:
            self.dataset=dataset

            
    def parse_activators(self,raw_params):
        for param in list(raw_params.keys()):
            if "activators" in param or "act" in param or "bridge_act" in param or "lay_act" in param:
                if isinstance(raw_params[param]["name"],list):
                    instantiated_act=[]
                    for act_id in range(len(raw_params[param]["name"])):
                        instantiated_act.append(
                            getattr(nn,raw_params[param]["name"][act_id])(**(raw_params[param]["params"][act_id]))
                        )
                else:
                    instantiated_act=getattr(nn,raw_params[param]["name"])(**(raw_params[param]["params"]))
                raw_params[param]=instantiated_act
        return raw_params

    def set_model(self,model_args):
        submodels_data=model_args["sub_modules"]
        #Load modules
        for module in list(submodels_data.keys()):
            if module=="P_NET" or module=="Q_NET":
                #Load variational modules
                inst_module=getattr(
                    getattr(
                        getattr(
                            Models,
                            submodels_data[module]["variational_generation_type"]
                            ),
                        module
                    ),
                    module
                    )(**(self.parse_activators(submodels_data[module]["parameters"])))
                submodels_data[module]=inst_module
            elif "Asymmetrical" in submodels_data[module]["module_type"].split("_"):
                submodels_data[module]["parameters"]["encoder_parameters"]=self.parse_activators(submodels_data[module]["parameters"]["encoder_parameters"])
                submodels_data[module]["parameters"]["decoder_parameters"]=self.parse_activators(submodels_data[module]["parameters"]["decoder_parameters"])
                #Load variational modules
                inst_module=getattr(
                    getattr(
                        Encoding_Decoding_modules,
                        "Basic_Encoding_Decoding_Module"
                    ),
                    submodels_data[module]["module_type"]
                    )(
                        **(submodels_data[module]["parameters"])
                        )
                submodels_data[module]=inst_module
            else:
                #Load variational modules
                inst_module=getattr(
                    getattr(
                        Encoding_Decoding_modules,
                        "Basic_Encoding_Decoding_Module"
                    ),
                    submodels_data[module]["module_type"]
                    )(
                        **(self.parse_activators(submodels_data[module]["parameters"]))
                        )
                submodels_data[module]=inst_module
            
            #Instance model
        model_args["model_params"].update(model_args["sub_modules"])
        self.instantiated_model=getattr(
            getattr(
                Models,
                model_args["model_class"]
                ),
                model_args["model_name"]
                )(**(model_args["model_params"]))

        
    def Train(self):
        for test_id in tqdm(range(len(self.test_dirs)),desc="Model test"):
            test=self.test_dirs[test_id]
            test_json=json.load(open(os.path.join(test,"config.json")))
            test_json_save=copy.deepcopy(test_json)
            trainer_args=test_json["trainer"]
            model_args=test_json["model"]
            transforms_args=test_json["transforms"]

            self.prepare_transforms(transforms_args)
            self.set_datasets(test)
            self.set_model(model_args)
            
            if test_json["trainer"]["use_cuda"]:
                #trainer_args["model"]=self.instantiated_model.cuda()
                trainer_args["model"]=self.instantiated_model.to("cuda:0")
            else:
                trainer_args["model"]=self.instantiated_model
            trainer_args["data_dir"]=test
            trainer_args["dataset"]=self.datasets

            #model load debug

            self.trainer=trainer(**(trainer_args))
            self.trainer.optimizer=torch.optim.Adam(self.trainer.model.parameters(),**(test_json["optimizer"]))
            
            if test_json["experiment_state"]=="waiting":
                try:
                    if self.train and self.test:
                        self.trainer.train_test(**(self.datasets))
                        test_json_save["experiment_state"]="done"
                    elif self.train and not(self.test):
                        #set train rutine
                        test_json_save["experiment_state"]="done"
                    elif not(self.train) and self.test:
                        self.trainer.train(self.datasets["train"])
                        self.trainer.test(self.datasets["test"])
                        test_json_save["experiment_state"]="done"
                #elif self.K_fold_training!=None:

                except Exception as e:
                    #FOR DEBUGGING
                    #traceback.print_exc()
                    tqdm.write("tranning failed")
                    tqdm.write(str(e))
                    test_json_save["experiment_state"]="fail"
                    test_json_save["error"]=str(e)
                    #TODO show error
                #save config.json
                f=open(os.path.join(test,"config.json"),"w")
                json.dump(test_json_save,f,indent=6)
                #f.write(json.dump(test_json))
                f.close()
                tqdm.write("Training model "+str(test_id))

class image_graph_multi_parameter_training(multi_parameter_training):
    def set_model(self,model_args):
        submodels_data=model_args["sub_modules"]
        #Load modules
        for module in list(submodels_data.keys()):
            if module=="graph_encoding_mode":
                #Load graph modules
                inst_module=getattr(
                    getattr(
                        getattr(
                            Models,
                            "DL_utils"
                            ),
                        "Graph_modules"
                    ),
                    submodels_data[module]["module_type"]
                    )(**(self.parse_activators(submodels_data[module]["parameters"])))
                submodels_data[module]=inst_module
            elif module=="image_encoding_mode":
                #Load image modules
                inst_module=getattr(
                    getattr(
                        getattr(
                            Models,
                            "DL_utils"
                            ),
                        "ResNET"
                    ),
                    submodels_data[module]["module_type"]
                    )(**(self.parse_activators(submodels_data[module]["parameters"])))
                submodels_data[module]=inst_module
            elif module=="Estimation_NeuralNet":
                inst_module=getattr(
                    getattr(
                        getattr(
                            Models,
                            "DL_utils"
                            ),
                        "utils"
                    ),
                    submodels_data[module]["module_type"]
                    )(**(self.parse_activators(submodels_data[module]["parameters"])))
                submodels_data[module]=inst_module
            
            #Instance model
        model_args["model_params"].update(model_args["sub_modules"])
        self.instantiated_model=getattr(
            getattr(
                Models,
                model_args["model_class"]
                ),
                model_args["model_name"]
                )(**(model_args["model_params"]))