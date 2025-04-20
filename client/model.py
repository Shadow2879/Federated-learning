'''
Defines a lightning model in for classification tasks on 28x28 images
'''
import torch
from torch import nn
import lightning as L
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy,MulticlassPrecision,MulticlassRecall
import torch.nn.functional as F
import gradio_client
import time
# import dotenv
# import os
from common.env_path_fns import load_env_var
class model(nn.Module):
    '''
    Torch module containing the model architecture and forward pass
    '''
    def __init__(self,output_classes):
        super().__init__()
        self.model=nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,128),#(32*784)x(784,128)+(32,128)=32x128
            nn.Linear(128,output_classes)
        )
    def forward(self,x):
        return self.model(x)
class LitEMNISTClassifier(L.LightningModule):
    '''
    The lightning Module for 28x28 image classification
    '''
    def __init__(
            self,
            metrics:list[torchmetrics.Metric]=[
                MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,
                ],
                output_classes: int=None,
                loss_fn=torch.nn.MSELoss(),
                fed_learning:bool=True):
        super().__init__()
        # dotenv.load_dotenv()
        self.model=model(output_classes)
        self.metrics=metrics
        self.output_classes=output_classes if output_classes is not None else load_env_var('OUTPUT_CLASSES','int')#int(os.environ.get('OUTPUT_CLASSES'))
        self.batch_size=load_env_var('BATCH_SIZE','int')#int(os.environ.get('BATCH_SIZE'))
        self.example_input_array=torch.Tensor(self.batch_size,1,28,28)
        self.loss_fn=loss_fn
        self.fed_learning=fed_learning
        if self.fed_learning:
            self.client=gradio_client.Client(load_env_var('AGG_SERVER_ADDR','str'))#os.environ.get('AGG_SERVER_ADDR')
            print(self.client.view_api())
            self.model_ver=self.client.predict(api_name='/get_model_ver')
            model_wgt_loc=self.client.predict(api_name='/get_model_weights')
            model_wgt=torch.load(model_wgt_loc)
            self.model.load_state_dict(model_wgt)
            print(f'loaded model{self.model_ver}')

    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(),lr=1e-5)
        return optimizer
    
    def convert_data_float(self,tensors:list):
        for i in enumerate(tensors):
            tensors[i[0]]=i[1].float()
        tensors[1]=F.one_hot(tensors[1],self.output_classes)
        return tensors
    
    def log_metrics(self,loss,phase,pred,target):
        self.log(f'{phase}_loss',loss,prog_bar=True,logger=True)
        for i in self.metrics:
            self.log(
                f'{phase}_{i._get_name()}',
                i.to(pred)(pred,target),
                on_epoch=True,
                logger=True,
                prog_bar=True
            )

    def forward(self,x):
        return self.model(x)
    
    def predict_step(self,batch,batch_idx,dataloader_idx=0):
        x,_=batch
        return self(x)
    
    def training_step(self, batch,batch_idx):
        x,y=self.convert_data_float(batch)
        outputs=self(x)
        loss=self.loss_fn(outputs,y)
        # self.log('train_loss',loss,prog_bar=True,logger=True)
        # self.training_ouputs.append(loss.detach())
        self.log_metrics(loss,'train',outputs,y)
        # for i in self.metrics:
        #     self.log(
        #         f'train_{i._get_name()}',
        #         i.to(outputs)(outputs,y),
        #         on_epoch=True,
        #         logger=True,
        #         prog_bar=True,
        #     )
        return loss

    def on_train_epoch_end(self):
        # if self.fed_learning:
        #     f_name=f"/workspace/{str(time.ctime()).replace(' ','_').replace(':','-')}"
        #     torch.save(self.model.state_dict(),f_name)
        #     self.client.predict(f_name,api_name='/upload_model',)
        #     self.client.predict()
        # else:
        return super().on_train_epoch_end()
    

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