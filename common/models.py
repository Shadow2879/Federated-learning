'''
    contains various data and model classes for loading data and training neural networks
'''
from torch import nn
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy,MulticlassPrecision,MulticlassRecall
from common.connect import connect_to_gr_client
import numpy as np
import lightning as L
import shutil
from torch.utils.data import random_split,DataLoader,Dataset
import os
from typing import Sequence
class NNmodel(nn.Module):
    '''
    Torch module containing the model architecture, forward pass and operator overloading for +, -, *, /, ** 
    and a function to see how similar the params of a model are with another.
    '''
    def __init__(self,output_classes=62):
        super().__init__()
        self.model=nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,256),
            nn.Linear(256,128),
            nn.Linear(128,output_classes)
        )
    
    def forward(self,x):
        return self.model(x)
    
    def __add__(self,model2):
        res=OrderedDict()
        for keys in self.state_dict().keys():
            res[keys]=self.state_dict()[keys]+model2.state_dict()[keys]
        res_model=NNmodel()
        res_model.load_state_dict(res)
        return res_model
    
    def __sub__(self,model2):
        res=OrderedDict()
        for keys in self.state_dict().keys():
            res[keys]=self.state_dict()[keys]-model2.state_dict()[keys]
        res_model=NNmodel()
        res_model.load_state_dict(res)
        return res_model
    
    def __mul__(self,val:float):
        res=OrderedDict()
        for keys in self.state_dict().keys():
            res[keys]=self.state_dict()[keys]*val
        res_model=NNmodel()
        res_model.load_state_dict(res)
        return res_model
    
    def __truediv__(self,val:float):
        res=OrderedDict()
        for keys in self.state_dict().keys():
            res[keys]=self.state_dict()[keys]/val
        res_model=NNmodel()
        res_model.load_state_dict(res)
        return res_model
    
    def __pow__(self,val:float):
        res=OrderedDict()
        for keys in self.state_dict().keys():
            res[keys]=self.state_dict()[keys]**val
        res_model=NNmodel()
        res_model.load_state_dict(res)
        return res_model
    
    def get_distance(self,model2,distance_metric:int=2):
        if distance_metric %2 != 0 and distance_metric != 1:
            raise NotImplementedError
        res=(self-model2)**distance_metric
        res=pow(sum([p.sum() for p in res.parameters()]),1/distance_metric)
        return res
    
class LitEMNISTClassifier(L.LightningModule):
    '''
    The lightning Module for 28x28 image classification
    '''
    def __init__(
            self,
            model_path:str,
            model_ver:int,
            metrics:list=[
                MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,
                ],
                output_classes: int=62,
                loss_fn=torch.nn.MSELoss(),
                batch_size=32,
                fed_learning:bool=True):
        super().__init__()
        self.model=NNmodel(output_classes)
        self.output_classes=output_classes
        self.metrics=[i(self.output_classes,average='macro') for i in metrics]
        self.batch_size=batch_size
        self.example_input_array=torch.Tensor(self.batch_size,1,28,28)
        self.loss_fn=loss_fn
        self.fed_learning=fed_learning
        self.save_hyperparameters('output_classes')
        if self.fed_learning:
            self.model_ver=model_ver
            model_wgt=torch.load(model_path)
            self.model.load_state_dict(model_wgt)
            print(f'loaded model{self.model_ver}')

    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(),lr=1e-5)
        return optimizer
    
    def convert_data_float(self,tensors:list):
        tensors[1]=F.one_hot(tensors[1],self.output_classes)
        for i in enumerate(tensors):
            tensors[i[0]]=i[1].float()
        return tensors
    
    def log_metrics(self,loss,phase,pred,target):
        self.log(f'{phase}_loss_epoch',loss.mean(),prog_bar=True,logger=True,on_step=False,on_epoch=True)
        self.log(f'{phase}_loss_step',loss.mean(),prog_bar=True,logger=True,on_step=True,on_epoch=False)
        for i in self.metrics:
            self.log(
                f'{phase}_{i._get_name()}_epoch',
                i.to(pred)(pred,target).mean(),
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=True
            )
            self.log(
                f'{phase}_{i._get_name()}_step',
                i.to(pred)(pred,target).mean(),
                on_epoch=False,
                on_step=True,
                logger=True,
                prog_bar=True
            )

    def forward(self,x):
        return self.model(x)
    
    def predict_step(self,batch,batch_idx):
        x,_=batch
        return self(x)
    
    def training_step(self, batch,batch_idx):
        x,y=self.convert_data_float(batch)
        outputs=self(x)
        loss=self.loss_fn(outputs,y)
        self.log_metrics(loss,'train',outputs,y)
        return loss

    def test_step(self,batch,batch_idx):
        x,y=self.convert_data_float(batch)
        outputs=self(x)
        loss=self.loss_fn(outputs,y)
        self.log_metrics(loss,'test',outputs,y)

    def validation_step(self,batch,batch_idx):
        x,y=self.convert_data_float(batch)
        outputs=self(x)
        loss=self.loss_fn(outputs,y)
        self.log_metrics(loss,'val',outputs,y)

class CustDataset(Dataset):
    def __init__(self,data):
        self.data=data
    def __len__(self):
        return self.data[0].shape[0]
    def __getitem__(self, index):
        return [i[index] for i in self.data]

class PartialEMNISTDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_loc:str,
            client_addr:str,
            splits:Sequence[int | float | str],
            seed:int=42,
            delay:int | float=10,
            tries:int=10,
            batch_size:int=32,
            cpus:int=1,
            ):
        super().__init__()
        self.data_loc=data_loc
        self.batch_size=batch_size
        self.cpus=cpus
        self.generator=torch.Generator().manual_seed(seed)
        self.splits=np.array(splits,dtype=np.float32)
        self.splits/=self.splits.sum()
        self.prepared=False
        self.delay=delay
        self.tries=tries
        self.client_addr=client_addr

    def prepare_data(self):
        if not self.prepared:
            self.client=connect_to_gr_client(
                self.client_addr,
                delay=self.delay,
                tries=self.tries,
                download_files=self.data_loc)
            data_file=self.client.predict(api_name='/serve_client')
            self.data_file=os.path.join(self.data_loc,data_file.split('/')[-1])
            print(f'moving from {data_file} to {self.data_file}')
            shutil.move(data_file,self.data_file)
            ds=CustDataset(torch.load(self.data_file))
            self.train,self.val,self.test=random_split(ds,
                                                    lengths=self.splits.tolist(),
                                                    generator=self.generator)
            self.prepared=True
    
    def train_dataloader(self):
        return DataLoader(self.train,self.batch_size,num_workers=self.cpus)
    
    def val_dataloader(self):
        return DataLoader(self.val,self.batch_size,num_workers=self.cpus)
    
    def test_dataloader(self):
        return DataLoader(self.test,self.batch_size,num_workers=self.cpus)
    
    def predict_dataloader(self):
        return DataLoader(self.test,self.batch_size,num_workers=self.cpus)
