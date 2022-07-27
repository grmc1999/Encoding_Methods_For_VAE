from tqdm import tqdm
import numpy as np
import torch
import os
import pickle
import copy
import matplotlib.pyplot as plt
import gc

from .plot_utils import plot_training_sample

class trainer(object):
    def __init__(self,model,dataset,epochs,folds,batch_size,use_cuda,data_dir,in_device=None,num_workers=0,args=["PhantomRGB"],uniform=True,view_out_state=True,MNIST_debug=False):
        self.epochs=epochs
        self.dataset=dataset
        self.folds=folds
        self.template_model=model
        self.model=model
        self.batch_size=batch_size
        self.use_cuda=use_cuda
        self.in_device=in_device
        self.data_dir=data_dir
        self.args=args
        self.uniform=uniform
        self.losses={}
        self.loss_DATA={}
        self.loss_DATA_FOLD={}
        self.MNIST_debug=MNIST_debug

        self.current_epoch=0
        self.current_fold=0
        self.num_workers=num_workers
        #TODO: check automatic list retrieval
        self.loss_list=list(self.model.losses_weigths.keys())+["total_loss"]
        self.loss_epoch={}
        self.loss_epoch=self.create_loss_data(self.loss_epoch)
        self.view_out_state=view_out_state

        self.loss_DATA=self.create_loss_data(self.loss_DATA)


    def create_loss_data(self,dict):
        teld={}
        trld={}
        for l in self.loss_list:
            trld[l]=[]
            teld[l]=[]
        dict={
            "train":trld,
            "test":teld
        }
        return dict

    def save_dict(self,dict,path):
        file=open(path,'wb')
        pickle.dump(dict,file)
        file.close()

    def load_dict(self,path):
        file=open(path,'rb')
        dict=pickle.load(file)
        file.close()
        return dict
    
    def prep_log(self,losses_dict):
        msg="data:"
        for l in self.loss_list:
            msg=msg+"\t"+l+(" {loss:.4f}").format(loss=losses_dict[l].item())
        return msg

    def security_checkpoint(self,current_epoch,loss):
        torch.save({
            'current_epoch':current_epoch,
            'total_epoch':self.epochs,
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss':loss
        },os.path.join(self.data_dir,"checkpoint.pt"))

    def train(self,dataloader):
        device="cpu"
        if self.use_cuda:
            device="cuda"
        if self.in_device!=None:
            device=self.in_device
        
        for l in self.loss_list:
            self.loss_epoch["train"][l]=[]

        for idx, batch in tqdm(enumerate(dataloader),desc="instances"):
            #print(batch["x"].size())
            #ONLY FOR DEBUG ------------------------------------------------------------------------
            if self.MNIST_debug:
                batch=batch[0]
            #----------------------------------------------------------------------------------------
            args=(batch[arg].to(device) for arg in self.args)
            losses=self.model.compute_losses(*(args))
            
            #Compute loss
            loss=losses["total_loss"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            msg=self.prep_log(losses)

            tqdm.write(
                msg+" \tbatch {shape:.4f}".format(
                    shape=batch[self.args[0]].shape[0]
            )
            )
            #STACK MINIBATCH LOSS
            for l in self.loss_list:
                self.loss_epoch["train"][l].append(losses[l].item())

        #EPOCH MEAN LOSSES
        for l in self.loss_list:
            self.loss_DATA["train"][l].append(
                np.mean(
                    np.array(self.loss_epoch["train"][l])
                    )
                )
        return self.loss_epoch


    def test(self,dataloader):
        device="cpu"
        if self.use_cuda:
            device="cuda"
        if self.in_device!=None:
            device=self.in_device

        for l in self.loss_list:
            self.loss_epoch["test"][l]=[]
        
        for idx, batch in tqdm(enumerate(dataloader),desc="Test"):
            #if self.uniform:
            #    losses=self.model.ELBO(batch[self.args[0]].to(device))
            #else:
            #    args=(batch[arg].to(device) for arg in self.args)
            #    losses=self.model.ELBO(*(args))
            if self.MNIST_debug:
                batch=batch[0]
            args=(batch[arg].to(device) for arg in self.args)
            losses=self.model.compute_losses(*(args))

            msg=self.prep_log(losses)

            tqdm.write(
                msg+" \tbatch {shape:.4f}".format(
                    shape=batch[self.args[0]].shape[0]
            )
            )

            #STACK MINIBATCH LOSS
            for l in self.loss_list:
                self.loss_epoch["test"][l].append(losses[l].item())

        #EPOCH MEAN LOSSES
        for l in self.loss_list:
            self.loss_DATA["test"][l].append(
                np.mean(
                    np.array(self.loss_epoch["test"][l])
                    )
                )
        return self.loss_epoch

    def train_test(self,train_set,test_set):

        #Is file already exists charge ------------------------------------------------------------------------------------------------------------------------
        if b"loss_results.npy" in os.listdir(self.data_dir) or "loss_results.npy" in os.listdir(self.data_dir):
            print("result found")
            self.loss_DATA=np.load(os.path.join(self.data_dir,'loss_results.npy'),allow_pickle=True).tolist()

            #LOAD OPTIMIZER, MODEL, CURRENT EPOCH AND NUMBER OF EPOCHS FROM CHECKPOINT ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if b"checkpoint.pt" in os.listdir(self.data_dir) or "checkpoint.pt" in os.listdir(self.data_dir):
            print("checkpoint found")

            checkpoint=torch.load(os.path.join(self.data_dir,"checkpoint.pt"))
            self.current_epoch=checkpoint["current_epoch"]
            #self.epochs=checkpoint["total_epoch"]
            if self.epochs-1!=self.current_epoch:

                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print("was the last")

        else:
            self.current_epoch=0


        best_result=0

        if self.view_out_state:
            #z_mu,z_sig,x_r,x
            plot_sample=plot_training_sample(images_idxs=[2,3],titles=["reconstruction","input"])

        for epoch in tqdm(range(self.current_epoch,self.epochs),desc="Epoch"):

            drop_train=False
            drop_test=False
            if len(train_set)%self.batch_size==1:
                drop_train=True

            if len(test_set)%self.batch_size==1:
                drop_test=True

            dataloader_train=torch.utils.data.DataLoader(train_set,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,drop_last=drop_train)
            dataloader_test=torch.utils.data.DataLoader(test_set,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers,drop_last=drop_test)

            loss_epoch=self.train(dataloader_train)
            losses_te=self.test(dataloader_test)
            torch.cuda.empty_cache()
            
            gc.collect()

            #CHECK OUTPUT PER EPOCH
            if self.view_out_state:
                idx=np.random.randint(len(test_set))
                if self.MNIST_debug:
                    x=test_set[idx][0][self.args[0]]
                else:
                    x=test_set[idx][self.args[0]]
                if self.use_cuda:
                    x=x.cuda()
                self.model.eval()
                out=self.model.forward_pass(x.unsqueeze(0))
                out=(list(out))
                out.append(x.unsqueeze(0))
                #z_mu,z_sig,x_r,x
                self.model.train()
                plot_sample.plot(out)

            if (np.mean(np.array(losses_te["test"]["total_loss"])))>best_result:
                best_result=(np.mean(np.array(losses_te["test"]["total_loss"])))
                best_model=self.model.state_dict()
                torch.save(best_model,"{fname}.pt".format(fname=os.path.join(self.data_dir,"best"+str(self.current_fold))))

            #TODO: add epoch data
            tqdm.write("epoch {epoch:.2f}%".format(
                        epoch=epoch
                        ))

            del dataloader_train
            del dataloader_test
            
            np.save(os.path.join(self.data_dir,"loss_results"+'.npy'),self.loss_DATA)

            #SAVE CHECKPOINT ------------------------------------------------------------------------------------------------------------------------------------------------
            self.security_checkpoint(current_epoch=epoch,
                                loss=loss_epoch["train"]["total_loss"],
                                )
            del loss_epoch
            del losses_te


        return best_model

    def K_fold_train(self):

        #Shuffle data

        #Non_debug
        #TODO: Implement already splitted datasets
        train_s=int((len(self.dataset))*0.8)
        test_s=int(len(self.dataset)-train_s)

        print("train len")
        print(train_s)
        print("test len")
        print(test_s)

        #LOAD INDEXES IS ALREADY EXISTS ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "data_split.pkl" in os.listdir(self.data_dir):
            print("data split found")
            dict=self.load_dict(os.path.join(self.data_dir,"data_split.pkl"))
            train_index=dict["train_index"]
            test_index=dict["test_index"]
        else:
            #NON_DEBUG
            train_index,test_index=torch.utils.data.random_split(range(len(self.dataset)),[train_s, test_s])
            #DEBUG
            #train_index,test_index=torch.utils.data.random_split(range(30),[train_s, test_s])
            dataset_split_index={
                "train_index":train_index,
                "test_index":test_index
            }
            self.save_dict(dataset_split_index,os.path.join(self.data_dir,"data_split.pkl"))

        train_set = torch.utils.data.Subset(self.dataset, train_index)
        test_set = torch.utils.data.Subset(self.dataset, test_index)

        for fold in tqdm(range(self.folds),desc="folds"):
            self.current_fold=fold
            #TODO: check and load current fold
            #USE self.f_model
            self.model=copy.deepcopy(self.template_model)

            #optimizer
            self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3)

            #LOAD OPTIMIZER, MODEL, CURRENT EPOCH AND NUMBER OF EPOCHS FROM CHECKPOINT ------------------------------------------------------------------------------------------------------------------------------------------------------------------
            if "checkpoint.pt" in os.listdir(self.data_dir):
                print("checkpoint found")

                checkpoint=torch.load(os.path.join(self.data_dir,"checkpoint.pt"))
                self.current_epoch=checkpoint["current_epoch"]
                #self.epochs=checkpoint["total_epoch"]
                if self.epochs-1!=self.current_epoch:

                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    print("was the last")

            else:
                checkpoint_epoch=0

            #Epochs
            best_model=self.train_test(
                train_set=train_set,
                test_set=test_set,
            )

            self.loss_DATA_FOLD[fold]=self.loss_DATA

            tqdm.write("fold {fold:.2f}%".format(
                        fold=fold
                        ))

        np.save(os.path.join(self.data_dir,"fold_loss_results"+'.npy'),self.loss_DATA_FOLD)