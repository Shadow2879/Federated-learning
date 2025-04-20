'''
module with functions to perform various federated aggregation techniques
'''
import torch
import model
from torch import nn
# import dotenv
# import os
from common.env_path_fns import load_env_var
# dotenv.load_dotenv()
min_models=load_env_var('MIN_MODELS','int')
def ensure_models(models):
    # min_models=os.environ.get('MIN_MODELS')
    # min_models=

    if len(models)<min_models:
        print('not enough models')
        raise IndexError(f'tried to aggregate {len(models)} but minimum {min_models} required.')

def fed_avg(models:list[nn.Module])->model:
    ensure_models(models)
    res=models[0]
    for i in range(len(models)):
        res+=models[i]
    res/=len(models)
    return res

def fed_nova(models:list[nn.Module],steps:list[int],data_len:int) -> model:
    ensure_models(models)
    res=models[0]*(steps[0]/data_len)
    for i in enumerate(models):
        res=res+i[1]*(steps[i[0]]/data_len)
    res/=len(models)
    return res
    
def fed_prox(models:list[nn.Module],proximal_term:float,prev_global_model:nn.Module):
    ensure_models(models)
    res=((models[0]-prev_global_model)**2)*(proximal_term/2)
    for i in models:
        res+=((i-prev_global_model)**2)*(proximal_term/2)
    res/=len(models)
    res+=prev_global_model
    return res

def select_clients(models:list[nn.Module],prev_global_model:nn.Module,power:int=2,ratio:float=0.8):
    ensure_models(models)
    n=len(models)
    if len(models)*ratio> min_models:#os.environ.get('MIN_MODELS')
        n=len(models)*ratio
    distances=[]
    for i in models:
        distances.append(i.get_distance(prev_global_model,power))
    distances=torch.tensor(distances).topk(n)
    print(distances)
    return distances[0]

