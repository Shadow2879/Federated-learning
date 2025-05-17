'''
Defines a lightning model in for classification tasks on 28x28 images
'''
import torch
import lightning as L
from torchmetrics.classification import MulticlassAccuracy,MulticlassAveragePrecision,MulticlassRecall
import torch.nn.functional as F
import gradio as gr
import os
from model_classes import model,agg_models
import agg_strats
from common.env_path_fns import load_env_var
from datetime import datetime,timedelta

model_ver=0
g_model=model()
UPLOAD_MODEL_DIR=load_env_var('UPLOAD_MODEL_DIR','path')
RUNS=load_env_var('RUNS','int')
UPDATE_FREQ=load_env_var('UPDATE_FREQ','int')
model_counts={i:agg_models(
    dir=f'{os.path.join(UPLOAD_MODEL_DIR,str(i))}',
    agg_strat=agg_strats.fed_avg,)
      for i in range(RUNS)}
model_counts[0].add_global_model(g_model)
GLOBAL_MODEL_DIR=load_env_var('GLOBAL_MODEL_DIR','path')
gen_update_time=lambda :datetime.now()+timedelta(seconds=UPDATE_FREQ)
g_update_time=gen_update_time()

gr.set_static_paths(paths=[
    os.path.join(os.getcwd(),GLOBAL_MODEL_DIR),
    os.path.join(os.getcwd(),UPLOAD_MODEL_DIR)
])

def update_model() ->str:
    global g_update_time
    time_to_update=g_update_time-datetime.now()
    if time_to_update<timedelta(seconds=0):
        updated=agg_weights()
        if updated:
            return f'global model updated'
        else:
            g_update_time=gen_update_time()
            return f'not enough models, waiting till {g_update_time} for update.'
    else:
        return f'global model to be updated in :{time_to_update.seconds} seconds.'
    
def get_model_weights() ->str:
    f_loc=os.path.join(GLOBAL_MODEL_DIR,f'{model_ver}.pth')
    torch.save(g_model.state_dict(),f_loc)
    print(f"current modelfile: {f_loc}")
    return f_loc

def get_model_ver()-> int:
    print(f'model_version:{model_ver}')
    return model_ver

def upload_model(model_loc:str,counts:int) -> None:
    print('uploaded data:',model_loc,counts)
    global model_ver
    model_counts[model_ver].add_client_model(model_loc)

def agg_weights() -> bool:
    global model_ver
    g_model,update=model_counts[model_ver].update()
    if update:
        model_ver+=1
        model_counts[model_ver].add_global_model(g_model)
        return update
    return False

with gr.Blocks() as demo:
    gr.Number(value=get_model_ver,every=10,label='get_model_ver')
    gr.File(value=get_model_weights,every=10,label='get_model_weights')
    gr.Text(value=update_model,every=1,label='model update info')
    new_model=gr.File(label='upload_model')
    steps=gr.Number(value=0)
    upload_submit_btn=gr.Button(value='Upload')
    upload_submit_btn.click(
        upload_model,
        [new_model,steps]
    )
    # new_model.upload(
    #     upload_model,
    #     new_model,
    # )

demo.launch(
    server_name='0.0.0.0',
    server_port=load_env_var('SERVER_PORT','int')
    
    )
