import fire
import os
import sys
sys.path.append("Models")
import argparse
import glob
from torchvision import datasets

from Train_utils.Multi_parameter_train import multi_parameter_training





class launch_train(object):

    def set_dataset(self,dataset_dir,dataset_name):
        getattr(datasets,dataset_name)(dataset_dir, download=True)
        
        #DS=datasets.MNIST(dataset_dir, download=True)

    def train(self,data_exp_path,module_iteration_path,dataset_dir,dataset_name):
        """
        data_exp_path: exp directory for dataset - ex:  /kaggle/working/Result_dir/ELE2346_exps/MNIST
        module_iteration_path: Iteration type - ex: VAE_DNN
        dataset_dir: dataset locally saved - ex: "/content/sample_data/MNIST"
        dataset_name: pytorch defined name of dataset - ex: MNIST
        """

        self.set_dataset(dataset_dir,dataset_name)
        

        #print(EncDec)
        mpt=multi_parameter_training(
            results_directory=os.path.join(data_exp_path,module_iteration_path), #should be combination iteracion /content/drive/MyDrive/ELE2346_exps/MNIST/VAE_DNN
            dataset_root_directory=dataset_dir,
            Dataset_type=getattr(datasets,dataset_name),
            train=True,
            test=True,
            K_fold_training=None,
            visualization=True
        )
        mpt.Train()


if __name__=="__main__":
    #launch command python Multi_train.py train --data_exp_path args --module_iteration_path args --dataset_dir args --dataset_name args
    fire.Fire(launch_train)