import torch
import lightning as L
import torch.utils.data as data
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import gradio_client
import numpy as np
from common.env_path_fns import load_env_var
class PartialEMNISTDataModule(L.LightningDataModule):
    def __init__(self,seed=42,train_val_test_split:list[float]=None):
        super().__init__()
        self.data_loc=load_env_var('DATA_LOC','path')
        self.batch_size=load_env_var('BATCH_SIZE','int')
        self.cpus=load_env_var('CPUS','int')
        self.generator=torch.Generator().manual_seed(seed)
        self.splits=torch.tensor(train_val_test_split,requires_grad=False)\
              if train_val_test_split is not None else\
                  torch.tensor(np.array(load_env_var('DATA_SPLITS','array')).astype(float),requires_grad=False)

    def prepare_data(self):
        self.client=gradio_client.Client(load_env_var('DATA_SERVER_ADDR','str'),
                                         download_files=self.data_loc)
        self.data_file=self.client.predict(api_name='/serve_client')
        ds=torch.load(self.data_file)
        self.train,self.val,self.test=data.random_split(ds,
                                                lengths=(self.splits*len(ds)).int(),
                                                generator=self.generator)
    
    def train_dataloader(self):
        return DataLoader(self.tain,self.batch_size,num_workers=self.cpus)
    
    def val_dataloader(self):
        return DataLoader(self.val,self.batch_size,num_workers=self.cpus)
    
    def test_dataloader(self):
        return DataLoader(self.test,self.batch_size,num_workers=self.cpus)
    
    def predict_dataloader(self):
        return DataLoader(self.test,self.batch_size,num_workers=self.cpus)
