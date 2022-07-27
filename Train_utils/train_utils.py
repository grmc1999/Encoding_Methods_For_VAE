from tqdm import tqdm
import numpy as np
import torch
import os
import pickle

def save_dict(dict,path):
    file=open(path,'wb')
    pickle.dump(dict,file)
    file.close()

def load_dict(path):
    file=open(path,'rb')
    dict=pickle.load(file)
    file.close()
    return dict

def security_checkpoint(current_epoch,total_epoch,model,optimizer,loss,PATH):
    torch.save({
        'current_epoch':current_epoch,
        'total_epoch':total_epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':loss
    },PATH)

def train(model,optimizer,dataloader,use_cuda,loss_function,in_device=None):
    loss_d=[]
    bce_d=[]
    kld_d=[]
    device="cpu"
    if use_cuda:
        device="cuda"
    if in_device!=None:
        device=in_device
    for idx, batch in tqdm(enumerate(dataloader),desc="instances"):
        r_img,mu,sig=model(batch["PhantomRGB"].to(device))
        loss,bce,kld=loss_function(r_img,batch["PhantomRGB"].to(device),mu,sig)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tqdm.write(
            "total loss {loss:.4f}\tBCE {bce:.4f}\tKLD {kld:.4f}\tbatch {shape:.4f}".format(
                loss=loss.item(),
                bce=bce.item(),
                kld=kld.item(),
                shape=batch["PhantomRGB"].shape[0]
        )
        )


        #SAVE TRAIN DATA
        loss_d.append(loss.item())
        bce_d.append(bce.item())
        kld_d.append(kld.item())
    return loss_d,bce_d,kld_d


def test(model,dataloader,use_cuda,loss_function,in_device=None):
    loss_d=[]
    bce_d=[]
    kld_d=[]
    device="cpu"
    if use_cuda:
        device="cuda"
    if in_device!=None:
        device=in_device
    for idx, batch in tqdm(enumerate(dataloader),desc="Test"):
        r_img,mu,sig=model(batch["PhantomRGB"].to(device))
        loss,bce,kld=loss_function(r_img,batch["PhantomRGB"].to(device),mu,sig)
        
        tqdm.write(
            "total loss {loss:.4f}\tBCE {bce:.4f}\tKLD {kld:.4f}\tbatch {shape:.4f}".format(
                loss=loss.item(),
                bce=bce.item(),
                kld=kld.item(),
                shape=batch["PhantomRGB"].shape[0]
        )
        )
        
        #SAVE TEST DATA
        loss_d.append(loss.item())
        bce_d.append(bce.item())
        kld_d.append(kld.item())
    return loss_d,bce_d,kld_d

def train_test(model,optimizer,train_set,test_set,batch_size,use_cuda,loss_function,epochs,data_train_dir,in_device=None,n_workers=0,checkpoint_epoch=0):
    epoch_loss={}
    epoch_bce={}
    epoch_kld={}

    epoch_loss_train=[]
    epoch_bce_train=[]
    epoch_kld_train=[]

    epoch_loss_test=[]
    epoch_bce_test=[]
    epoch_kld_test=[]

    #Is file already exists charge ------------------------------------------------------------------------------------------------------------------------
    if "loss_results.npy" in os.listdir(data_train_dir):
        print("result register found")
        epoch_bce_train=np.load(os.path.join(data_train_dir,'bce_results.npy'),allow_pickle=True).tolist()['train']
        epoch_bce_test=np.load(os.path.join(data_train_dir,'bce_results.npy'),allow_pickle=True).tolist()['valid']

        epoch_kld_train=np.load(os.path.join(data_train_dir,'kld_results.npy'),allow_pickle=True).tolist()['train']
        epoch_kld_test=np.load(os.path.join(data_train_dir,'kld_results.npy'),allow_pickle=True).tolist()['valid']

        epoch_loss_train=np.load(os.path.join(data_train_dir,'loss_results.npy'),allow_pickle=True).tolist()['train']
        epoch_loss_test=np.load(os.path.join(data_train_dir,'loss_results.npy'),allow_pickle=True).tolist()['valid']

    
    best_result=0

    for epoch in tqdm(range(checkpoint_epoch,epochs),desc="Epoch"):

        drop_train=False
        drop_test=False
        if len(train_set)%batch_size==1:
            drop_train=True
        
        if len(test_set)%batch_size==1:
            drop_test=True

        dataloader_train=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,
                                                    num_workers=n_workers,drop_last=drop_train,persistent_workers=True)
        dataloader_test=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True,
                                                    num_workers=n_workers,drop_last=drop_test,persistent_workers=True)

        loss_tr,bce_tr,kld_tr=train(model,optimizer,dataloader_train,use_cuda,loss_function,in_device)
    
        epoch_loss_train.append(np.mean(np.array(loss_tr)))
        epoch_bce_train.append(np.mean(np.array(bce_tr)))
        epoch_kld_train.append(np.mean(np.array(kld_tr)))
    
        loss_d,bce_d,kld_d=test(model,dataloader_test,use_cuda,loss_function,in_device)
        #loss_d,bce_d,kld_d=test(model,dataloader_train,use_cuda,loss_function)
        
        if (np.mean(np.array(loss_d)))>best_result:
            best_result=(np.mean(np.array(loss_d)))
            best_model=model.state_dict()
    
        epoch_loss_test.append(np.mean(np.array(loss_d)))
        epoch_bce_test.append(np.mean(np.array(bce_d)))
        epoch_kld_test.append(np.mean(np.array(kld_d)))

        tqdm.write("epoch {epoch:.2f}%".format(
                    epoch=epoch
                    ))

        del dataloader_train
        del dataloader_test

        epoch_loss={"train":epoch_loss_train,
                        "valid":epoch_loss_test
                            }

        epoch_bce={"train":epoch_bce_train,
                        "valid":epoch_bce_test
                            }

        epoch_kld={"train":epoch_kld_train,
                        "valid":epoch_kld_test
                            }
        

        np.save(os.path.join(data_train_dir,"loss_results"+'.npy'),epoch_loss)
        np.save(os.path.join(data_train_dir,"bce_results"+'.npy'),epoch_bce)
        np.save(os.path.join(data_train_dir,"kld_results"+'.npy'),epoch_kld)

        #SAVE CHECKPOINT ------------------------------------------------------------------------------------------------------------------------------------------------
        security_checkpoint(current_epoch=epoch,
                            total_epoch=epochs,
                            model=model,
                            optimizer=optimizer,
                            loss=loss_tr,
                            PATH=os.path.join(data_train_dir,"checkpoint.pt")
                            )

    
    return epoch_loss_train,epoch_bce_train,epoch_kld_train,epoch_loss_test,epoch_bce_test,epoch_kld_test,best_model

def K_fold_train(model,
                dataset,
                epochs,
                batch_size,
                use_cuda,
                folds,
                data_train_dir,
                loss_fn,
                n_workers,
                in_device=None
                ):
    fold_loss={}
    fold_bce={}
    fold_kld={}

    #Shuffle data
    train_s=int((len(dataset))*0.8)
    test_s=int(len(dataset)-train_s)
    print("train len")
    print(train_s)
    print("test len")
    print(test_s)

    #LOAD INDEXES IS ALREADY EXISTS ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if "data_split.pkl" in os.listdir(data_train_dir):
        print("data split found")
        dict=load_dict(os.path.join(data_train_dir,"data_split.pkl"))
        train_index=dict["train_index"]
        test_index=dict["test_index"]
    else:
        train_index,test_index=torch.utils.data.random_split(range(len(dataset)),[train_s, test_s])
        dataset_split_index={
            "train_index":train_index,
            "test_index":test_index
        }
        save_dict(dataset_split_index,os.path.join(data_train_dir,"data_split.pkl"))

    #train_set, test_set = torch.utils.data.random_split(dataset, [train_s, test_s])
    train_set = torch.utils.data.Subset(dataset, train_index)
    test_set = torch.utils.data.Subset(dataset, test_index)

    for fold in tqdm(range(folds),desc="folds"):
        #train_set, test_set = torch.utils.data.random_split(dataset, [train_s, test_s])

        ed=model
        #optimizer
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

        #LOAD OPTIMIZER, MODEL, CURRENT EPOCH AND NUMBER OF EPOCHS FROM CHECKPOINT ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if "checkpoint.pt" in os.listdir(data_train_dir):
            print("checkpoint found")

            checkpoint=torch.load(os.path.join(data_train_dir,"checkpoint.pt"))
            checkpoint_epoch=checkpoint["current_epoch"]
            epochs=checkpoint["total_epoch"]
            if epochs-1!=checkpoint_epoch:
                
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print("was the last")

        else:
            checkpoint_epoch=0


        #Epochs
        epoch_loss_train,epoch_bce_train,epoch_kld_train,epoch_loss_test,epoch_bce_test,epoch_kld_test,best_model=train_test(
            model=model,
            optimizer=optimizer,
            train_set=train_set,
            test_set=test_set,
            batch_size=batch_size,
            use_cuda=use_cuda,
            loss_function=loss_fn,
            epochs=epochs,
            data_train_dir=data_train_dir,
            in_device=in_device,
            n_workers=n_workers,
            checkpoint_epoch=checkpoint_epoch
        )

        fold_loss[fold]={"train":epoch_loss_train,
                        "valid":epoch_loss_test
                            }

        fold_bce[fold]={"train":epoch_bce_train,
                        "valid":epoch_bce_test
                            }

        fold_kld[fold]={"train":epoch_kld_train,
                        "valid":epoch_kld_test
                            }
        
        torch.save(best_model,"{fname}.pt".format(fname=os.path.join(data_train_dir,"best"+str(fold))))

        tqdm.write("fold {fold:.2f}%".format(
                    fold=fold
                    ))

    np.save(os.path.join(data_train_dir,"fold_loss_results"+'.npy'),fold_loss)
    np.save(os.path.join(data_train_dir,"fold_bce_results"+'.npy'),fold_bce)
    np.save(os.path.join(data_train_dir,"fold_kld_results"+'.npy'),fold_kld)