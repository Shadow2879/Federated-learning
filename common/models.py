'''
    contains various data and model classes for loading data and training neural networks
'''
from torch import nn
from collections import OrderedDict
import torchmetrics
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy,MulticlassPrecision,MulticlassRecall
from common.env_path_fns import load_env_var
from common.connect import connect_to_gr_client
import numpy as np
import lightning as L
import shutil
from torch.utils.data import random_split,DataLoader,Dataset
import os
class model(nn.Module):
    '''
    Torch module containing the model architecture, forward pass and operator overloading for +, -, *, /
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
        res_model=model()
        res_model.load_state_dict(res)
        return res_model
    
    def __sub__(self,model2):
        res=OrderedDict()
        for keys in self.state_dict().keys():
            res[keys]=self.state_dict()[keys]-model2.state_dict()[keys]
        res_model=model()
        res_model.load_state_dict(res)
        return res_model
    
    def __mul__(self,val:float):
        res=OrderedDict()
        for keys in self.state_dict().keys():
            res[keys]=self.state_dict()[keys]*val
        res_model=model()
        res_model.load_state_dict(res)
        return res_model
    
    def __truediv__(self,val:float):
        res=OrderedDict()
        for keys in self.state_dict().keys():
            res[keys]=self.state_dict()[keys]/val
        res_model=model()
        res_model.load_state_dict(res)
        return res_model
    
    def __pow__(self,val:float):
        res=OrderedDict()
        for keys in self.state_dict().keys():
            res[keys]=self.state_dict()[keys]**val
        res_model=model()
        res_model.load_state_dict(res)
        return res_model
    
    def get_distance(self,model2,distance_metric=2):
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
            metrics:list[torchmetrics.Metric]=[
                MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,
                ],
                output_classes: int=0,
                loss_fn=torch.nn.MSELoss(),
                fed_learning:bool=True):
        super().__init__()
        self.model=model(output_classes)
        self.output_classes=output_classes if output_classes else load_env_var('OUTPUT_CLASSES','int')
        self.metrics=[i(self.output_classes,average='macro') for i in metrics]
        self.batch_size=load_env_var('CLIENT_BATCH_SIZE','int')
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
        self.log(f'{phase}_loss',loss.mean(),prog_bar=True,logger=True)
        for i in self.metrics:
            self.log(
                f'{phase}_{i._get_name()}',
                i.to(pred)(pred,target).mean(),
                on_epoch=True,
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
            splits:list[float | int],
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
        self.splits=torch.tensor(np.array(splits),requires_grad=False)
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
                                                    lengths=self.splits,
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
