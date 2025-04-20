'''
Defines a lightning model in for classification tasks on 28x28 images
'''
import torch
import lightning as L
from torchmetrics.classification import Accuracy,Precision,Recall
import torch.nn.functional as F
import gradio as gr
# import dotenv
import os
import shutil
import model
import agg_strats
from common.env_path_fns import load_env_var
# dotenv.load_dotenv()
model_ver=0
g_model=model.model()
model_counts={}
UPLOAD_MODEL_DIR=load_env_var('UPLOAD_MODEL_DIR','path')#os.path.join(os.getcwd(),os.environ.get('UPLOAD_MODEL_DIR'))
RUNS=load_env_var('RUNS','int')#int(os.environ.get('RUNS'))
GLOBAL_MODEL_DIR=load_env_var('GLOBAL_MODEL_DIR','path')#os.environ.get('GLOBAL_MODEL_DIR')

for i in range(RUNS):
    model_counts[i]=0
def get_model_weights():
    f_loc=os.path.join(GLOBAL_MODEL_DIR,f'{model_ver}.pth')
    torch.save(g_model.state_dict(),f_loc)
    print(f_loc)
    return f_loc
def get_model_ver():
    print(model_ver)
    return model_ver
def upload_model(model_loc):
    model_counts[model_ver]+=1
    shutil.move(model_loc,os.path.join(UPLOAD_MODEL_DIR,f'_{model_ver}_{model_counts[model_ver]}.pth'))
def agg_weights():
    files=os.listdir(f'UPLOAD_MODEL_DIR')
    models=[]
    for i in files:
        t=model()
        t.load_state_dict(torch.load(i))
        models.append(t)
    selected_models=agg_strats.select_clients(models,g_model)
    new_g_model=agg_strats.fed_avg(selected_models)
    g_model.load_state_dict(new_g_model.state_dict())

with gr.Blocks() as demo:
    gr.Number(value=get_model_ver,every=10,label='get_model_ver')
    gr.File(value=get_model_weights,every=10,label='get_model_weights')
    new_model=gr.File(label='upload_model')
    new_model.upload(
        upload_model,
        new_model,
    )

demo.launch(server_name='0.0.0.0',server_port=int(os.environ.get('SERVER_PORT')))
