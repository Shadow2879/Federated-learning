from datetime import datetime
import torch
from common.models import model
import os   

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

    def add_client_model(self,model_path:str,steps:int=None) -> None:
        self.models.append(
            model(self.classes).load_state_dict(
                torch.load(model_path,weights_only=True)
                ))
        self.model_steps.append(steps)
        
    def update(self) ->tuple[model,bool]:
        assert(self.prev_g_model is not None,"Previous global model not added!")
        try:
            self.new_model=self.agg(self.models,self.prev_g_model)
        except IndexError as e:
            return self.prev_g_model,False
        else:
            return self.new_model,True
    