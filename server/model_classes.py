import torch
from common.models import model
import os   
import shutil
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy,MulticlassAveragePrecision,MulticlassRecall
class  agg_models():
    def __init__(self,dir,agg_strat,classes=62,data_len:int=100000) -> None:
        self.dir=dir
        os.makedirs(dir,exist_ok=True)
        self.models=[]
        self.prev_g_model=None
        self.model_steps=[]
        self.agg=agg_strat
        self.classes=classes
        self.data_len=data_len

    def add_global_model(self,model) -> None:
        self.prev_g_model=model

    def add_client_model(self,model_path:str,steps:int | None=None) -> None:
        new_mod=model(self.classes)
        shutil.move(model_path,os.path.join(self.dir,f'{len(self.models)}.pth'))
        new_mod.load_state_dict(torch.load(
            os.path.join(self.dir,f'{len(self.models)}.pth'),
            weights_only=True
            ))
        self.models.append(new_mod)
        self.model_steps.append(steps)
        
    def update(self) ->tuple[model,bool]:
        assert self.prev_g_model is not None, "Previous global model not added!"
        try:
            self.new_model=self.agg(self.models,self.prev_g_model)
        except IndexError as e:
            return self.prev_g_model,False
        else:
            return self.new_model,True
    

def do_test_run(g_model,client,metrics=[MulticlassAccuracy(62),MulticlassAveragePrecision(62),MulticlassRecall(62)]):
    client.predict('server',api_name='/set_server')
    data=client.predict(api_name='/serve_client')
    test_data=torch.load(data)
    dl=DataLoader(test_data,batch_size=int(len(test_data)//1024))
    g_model.eval()
    with torch.no_grad() and torch.autocast(device_type='cpu') and tqdm(dl) as pbar:
        for batch in dl:
            stats={}
            inputs,targets=batch
            outputs=g_model(inputs)
            stats['loss']=F.mse_loss(outputs,targets)
            for metric in metrics:
                stats[metric._get_name()]=metric(outputs,targets).mean()
            pbar.set_description(stats)
            pbar.update(1)