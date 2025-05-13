from torch import nn
from collections import OrderedDict

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