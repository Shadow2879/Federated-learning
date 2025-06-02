'''
**DEPRECATED** please use classes from common.models instead.
'''
# '''
# Defines a lightning model in for classification tasks on 28x28 images
# '''
# import torch
# import lightning as L
# import torchmetrics
# from torchmetrics.classification import MulticlassAccuracy,MulticlassPrecision,MulticlassRecall
# import torch.nn.functional as F
# from common.env_path_fns import load_env_var
# from common.models import model

# class LitEMNISTClassifier(L.LightningModule):
#     '''
#     The lightning Module for 28x28 image classification
#     '''
#     def __init__(
#             self,
#             model_path:str,
#             model_ver:int,
#             metrics:list[torchmetrics.Metric]=[
#                 MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,
#                 ],
#                 output_classes: int=0,
#                 loss_fn=torch.nn.MSELoss(),
#                 fed_learning:bool=True):
#         super().__init__()
#         self.model=model(output_classes)
#         self.output_classes=output_classes if output_classes else load_env_var('OUTPUT_CLASSES','int')
#         self.metrics=[i(self.output_classes,average='macro') for i in metrics]
#         self.batch_size=load_env_var('CLIENT_BATCH_SIZE','int')
#         self.example_input_array=torch.Tensor(self.batch_size,1,28,28)
#         self.loss_fn=loss_fn
#         self.fed_learning=fed_learning
#         self.save_hyperparameters('output_classes')
#         if self.fed_learning:
#             self.model_ver=model_ver
#             model_wgt=torch.load(model_path)
#             self.model.load_state_dict(model_wgt)
#             print(f'loaded model{self.model_ver}')

#     def configure_optimizers(self):
#         optimizer=torch.optim.Adam(self.parameters(),lr=1e-5)
#         return optimizer
    
#     def convert_data_float(self,tensors:list):
#         tensors[1]=F.one_hot(tensors[1],self.output_classes)
#         for i in enumerate(tensors):
#             tensors[i[0]]=i[1].float()
#         return tensors
    
#     def log_metrics(self,loss,phase,pred,target):
#         self.log(f'{phase}_loss',loss.mean(),prog_bar=True,logger=True)
#         for i in self.metrics:
#             self.log(
#                 f'{phase}_{i._get_name()}',
#                 i.to(pred)(pred,target).mean(),
#                 on_epoch=True,
#                 logger=True,
#                 prog_bar=True
#             )

#     def forward(self,x):
#         return self.model(x)
    
#     def predict_step(self,batch,batch_idx):
#         x,_=batch
#         return self(x)
    
#     def training_step(self, batch,batch_idx):
#         x,y=self.convert_data_float(batch)
#         outputs=self(x)
#         loss=self.loss_fn(outputs,y)
#         self.log_metrics(loss,'train',outputs,y)
#         return loss

#     def test_step(self,batch,batch_idx):
#         x,y=self.convert_data_float(batch)
#         outputs=self(x)
#         loss=self.loss_fn(outputs,y)
#         self.log_metrics(loss,'test',outputs,y)

#     def validation_step(self,batch,batch_idx):
#         x,y=self.convert_data_float(batch)
#         outputs=self(x)
#         loss=self.loss_fn(outputs,y)
#         self.log_metrics(loss,'val',outputs,y)