import fire
import os
import sys
sys.path.append("Models")

from Train_utils.Multi_parameter_train import image_graph_multi_parameter_training




def main():

    for model in os.listdir(os.path.join("Results")):
        for EncDec in os.listdir(os.path.join("Results",model)):
            print(model)
            print(EncDec)
            mpt=image_graph_multi_parameter_training(
                results_directory=os.path.join("Results",model,EncDec),
                dataset_root_directory=os.path.join("..","Datasets","GPS","train_sorted"),
                train=True,
                test=True,
                K_fold_training=None,
                visualization=False
            )
            mpt.Train()


if __name__=="__main__":
    fire.Fire(main)