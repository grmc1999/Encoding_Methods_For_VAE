import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
def get_columname(PATH):
    for i in list(os.listdir(PATH)):
        file = os.path.join(PATH, i)
        instance = np.load(file, allow_pickle=True)
        column = []
        column += ['z_mu_' + str(j) for j in range(len(instance['z_mu']))]
        column += ['z_sig_' + str(j) for j in range(len(instance['z_sig']))]
        column += ['vp' + str(j) for j in range(len(instance['vp']))]
        column.append('y')
        return column

def load_meta(PATH):
    total = []
    idx = []
    for i in list(os.listdir(PATH)):
        instance_list= []
        file = os.path.join(PATH, i)
        filename = i.split('.')[0]
        image_id, val = filename.split('_')
        instance = np.load(file, allow_pickle=True)
        instance_list.extend(instance['z_mu'])
        instance_list.extend(instance['z_sig'])
        instance_list.extend(instance['vp'])
        instance_list.append(int(instance['y'].item()))
        if int(val) != int(instance['y'].item()):
            print("Error")
            break
        idx.append('train'+str(image_id).zfill(4))
        total.append(instance_list)
    
    total = np.array(total)
    df = pd.DataFrame(data = total, columns=get_columname(PATH), index = idx)
    df = df.astype({'vp0':int, 'vp1':int, 'vp2':int, 'y':int})
    return df


    

class CustomTrain(nn.Module):
    def __init__(self, n_features, n_classes, layers_list, activation=nn.ReLU(), dropout_list=None, batch_norm=True):
        super(CustomTrain, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.layers_list = layers_list
        self.activation = activation
        self.dropout_list = dropout_list
        self.batch_norm = batch_norm
        self.net = []
        self.b_list = []
        if self.dropout_list:
            self.dropout_list = [nn.Dropout(i) for i in self.dropout_list]
        for i in range(len(self.layers_list)):
            if i==0:
                self.b_list.append(nn.BatchNorm1d(self.n_features))
                self.net.append(nn.Linear(self.n_features, self.layers_list[i]))
            else:
                self.b_list.append(nn.BatchNorm1d(self.layers_list[i-1]))
                self.net.append(nn.Linear(self.layers_list[i-1], self.layers_list[i]))
        self.last_layer = nn.Linear(self.layers_list[-1], self.n_classes)
        self.net = nn.ModuleList(self.net)
        
    
    def forward(self, x):

        for i, l in enumerate(self.net):
            #print(i)
            if self.batch_norm:
                x = (self.b_list[i])(x)
            #x.to(device)
            x = self.activation(l(x))
            
            if self.dropout_list:
                x = (self.dropout_list[i])(x)
            
        x = self.last_layer(x)
        if self.n_classes == 1:
            x = torch.sigmoid(x)
   
        return x
    
    def eval(self):
        self.train(False)
        for i in self.b_list:
            i.train(False)

    def cust_train(self):
        self.train(True)
        for i in self.b_list:
            i.train(True)


class RankDataset(Dataset):
    def __init__(self, df, txt):
        self.df = df.drop('y', axis = 1)
        self.txt = txt
        self.array = None
        with open(self.txt, 'r') as file:
            self.array = file.readlines()

    def __getitem__(self, index):
        #print(index, len(self.array))
        a, p, n = (self.array[index]).split(" ")
        n = n[:-1]
        a = self.df.loc[a].values
        p = self.df.loc[p].values
        n = self.df.loc[n].values
        a = a.astype(np.float64)
        p = p.astype(np.float64)
        n = n.astype(np.float64)
        return {'a':torch.from_numpy(a).float(),\
                'p':torch.from_numpy(p).float(), \
                'n':torch.from_numpy(n).float()}



class datapaltas(Dataset):
    def __init__(self, df, scale =True, y_idx=-1):
        self.df = df
        self.y_idx = y_idx

    def __getitem__(self, index):
        X = (self.df.iloc[index,:self.y_idx]).values
        X = X.astype(np.float64)
        X = torch.from_numpy(X).float()
        y = self.df.iloc[index,self.y_idx]
        #print(y)
        y = torch.Tensor([y]).long()
        return {'x':X, 'y':y}
    
    def __len__(self):
        return self.df.shape[0]
    
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc



def train(data, model, ep = 120, save=False, prefix=None, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    idx_result = {}
    a = datapaltas(df = data, y_idx=-1)
    EPOCHS = ep
    skf = StratifiedKFold(shuffle=True)
    ## y_idx -2
    skf.get_n_splits(data, data.iloc[:,-1])
    vec_train = np.array([train_ids for train_ids,_ in skf.split(data, data.iloc[:,-1])])
    vec_test = np.array([test_ids for _,test_ids in skf.split(data, data.iloc[:,-1])])
    acv, tav = [], []
    foldn = 0
    for train_ids, test_ids in zip(vec_train, vec_test):
        acv_, tav_ = [], []
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_loader = torch.utils.data.DataLoader(
                            a, 
                            batch_size=64, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(
                            a,
                            batch_size=1, sampler=test_subsampler)

        EPOCHS = ep
        itt = tqdm(range(EPOCHS))
        for i in itt:
            loss_epoch = 0
            acc_train, acc_test = 0.0, 0.0
            model.cust_train()
            for sample in train_loader:
                X, y = sample['x'], sample['y']
                X, y = X.to(device), y.to(device).flatten()
                y_pred = model(X)
                #print(y_pred, y)
                loss = criterion(y_pred, y)
                acc = multi_acc(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                acc_train += acc.item()
                
            
            model.eval()
            
            with torch.no_grad():
                for sample in test_loader:
                    X_test, y_test = sample['x'], sample['y']
                    X_test, y_test = X_test.to(device), y_test.to(device).flatten()
                    #print(model.training)
                    y_hat = model(X_test)
                    acc = multi_acc(y_hat, y_test)
                    acc_test += acc.item()
                
            acv_.append(acc_test/len(test_loader))
            tav_.append(acc_train/len(train_loader))
            #print(acv_)
            if save and (abs(acc_test/len(test_loader) - max(acv_)<1e-8)):
                    if os.path.exists(os.path.join(os.curdir, prefix+"model_fold_"+str(foldn)+".pth")):
                        os.remove(os.path.join(os.curdir, prefix+"model_fold_"+str(foldn)+".pth"))
                    torch.save(model, os.path.join(os.curdir, prefix+"model_fold_"+str(foldn)+".pth"))
                    itt.set_postfix({'epoch_model': i, 'best_acc':max(acv_)})
            t1, t2 = acc_train/len(train_loader), acc_test/len(test_loader)
            itt.set_description(f"Acc train: {t1:.2f} Acc test: {t2:.2f}")
        acv.append(acv_)
        tav.append(tav_)
        

        foldn += 1
        model.apply(init_normal)
        
    acv = np.array(acv)
    tav = np.array(tav)
    idx_result['acc'] = np.mean(acv, axis = 0)
    idx_result['acc_std'] = np.std(acv, axis=0)
    idx_result['train_acc'] = np.mean(tav, axis=0)
    return idx_result




def trainRank(data, model, optimizer, ep = 120, save=False, prefix=None, device='cpu'):
    criterion = nn.TripletMarginLoss()
    
    idx_result = {}
    a = RankDataset(df = data, txt=os.path.join(os.pardir, 'sampler.txt'))
    EPOCHS = ep
    skf = KFold(shuffle=True)
    ## y_idx -2
    skf.get_n_splits(list(range(len(a.array))))
    vec_train = np.array([train_ids for train_ids,_ in skf.split(list(range(len(a.array))))])
    vec_test = np.array([test_ids for _,test_ids in skf.split(list(range(len(a.array))))])
    foldloss = []
    foldn = 0
    for train_ids, test_ids in zip(vec_train, vec_test):
        acloss = []
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_loader = torch.utils.data.DataLoader(
                            a, 
                            batch_size=64, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(
                            a,
                            batch_size=1, sampler=test_subsampler)

        EPOCHS = ep
        itt = tqdm(range(EPOCHS))
        for i in itt:
            loss_epoch = 0
            model.cust_train()
            for sample in train_loader:
                a, p, n = sample['a'], sample['p'], sample['n']
                a, p, n = a.to(device), p.to(device), n.to(device)
                emb_a, emb_p, emb_n = model(a), model(p), model(n)
                #print(y_pred, y)
                loss = criterion(emb_a, emb_p, emb_n)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            loss_epoch/=len(train_loader)
            itt.set_description(f"Loss: {loss_epoch:.9f}")
            acloss.append(loss_epoch)
            if save and loss_epoch<=(min(acloss)):
                    if os.path.exists(os.path.join(os.curdir, prefix+"model_fold_"+str(foldn)+".pth")):
                        os.remove(os.path.join(os.curdir, prefix+"model_fold_"+str(foldn)+".pth"))
                    torch.save(model, os.path.join(os.curdir, prefix+"model_fold_"+str(foldn)+".pth"))
                    itt.set_postfix({'epoch_model': np.argmin(acloss)+1, 'loss':min(acloss)})

        foldloss.append(acloss)
        foldn += 1
        model.apply(init_normal)
        return {'loss':foldloss}
