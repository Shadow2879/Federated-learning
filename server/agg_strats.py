'''
module with functions to perform various federated aggregation techniques
'''
import torch
from common.models import NNmodel
from common.env_path_fns import load_env_var
from torch import nn

min_models=load_env_var('AGG_SERVER_MIN_MODELS','int')
def ensure_models(models:list[NNmodel]) -> None:

    if len(models)<min_models:
        print(f'need {min_models} but have {len(models)}')
        raise IndexError(f'tried to aggregate {len(models)} but minimum {min_models} required.')

def fed_avg(models:list[NNmodel],prev_global_model:NNmodel)->NNmodel:
    ensure_models(models)
    res=models[0]
    for i in range(len(models)):
        res+=models[i]
    res/=len(models)
    return res

def fed_avg_weighted_steps(models:list[NNmodel],prev_global_model:NNmodel,steps:list[int] | None=None,data_len:int | None=None) -> NNmodel:
    ensure_models(models)
    if data_len is None or steps is None:
        return fed_avg(models,prev_global_model)

    res=models[0]*(steps[0]/len(models))
    for i in enumerate(models):
        res=res+i[1]*(steps[i[0]]/data_len)
    return res
    
def fed_prox(models:list[NNmodel],prev_global_model:NNmodel,proximal_term:float=0.5) -> NNmodel:
    ensure_models(models)
    res=((models[0]-prev_global_model)**2)*(proximal_term/2)
    for i in models:
        res+=((i-prev_global_model)**2)*(proximal_term/2)
    res/=len(models)
    res+=prev_global_model
    return res

def select_clients(models:list[NNmodel],prev_global_model:NNmodel,power:int=2,ratio:float=0.8) ->torch.Tensor:
    ensure_models(models)
    n=int(len(models)*(1 if len(models)*ratio>min_models else ratio))
    distances=[]
    for i in models:
        distances.append(i.get_distance(prev_global_model,power))
    distances=torch.tensor(distances).topk(n)
    print(distances)
    return distances[0]