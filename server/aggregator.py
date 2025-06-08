import torch, shutil, os
from common.models import NNmodel

class  agg_models():
    def __init__(self,dir,agg_strat,classes,def_steps:int=10000) -> None:
        self.dir=dir
        os.makedirs(dir,exist_ok=True)
        self.models=[]
        self.prev_g_model=None
        self.model_steps=[]
        self.agg=agg_strat
        self.classes=classes
        self.def_steps=def_steps

    def add_global_model(self,model) -> None:
        self.prev_g_model=model

    def add_client_model(self,model_path:str,steps:int | None=None) -> None:
        new_mod=NNmodel(self.classes)
        new_path=os.path.join(self.dir,f'{model_path.split("/")[-2]}.pth')
        print(f'moving client model from:{model_path} to {new_path}')
        shutil.move(model_path,new_path)
        new_mod.load_state_dict(torch.load(
            new_path,
            weights_only=True
            ))
        self.models.append(new_mod)
        if steps is None:
            steps=self.def_steps
        self.model_steps.append(steps)
        
    def update(self) ->tuple[NNmodel,bool]:
        assert self.prev_g_model is not None, "Previous global model not added!"
        try:
            self.new_model=self.agg(self.models,self.prev_g_model)
        except IndexError as e:
            return self.prev_g_model,False
        else:
            return self.new_model,True
