'''
Defines a lightning model in for classification tasks on 28x28 images
'''
import torch
# import lightning as L
# from torchmetrics.classification import MulticlassAccuracy,MulticlassAveragePrecision,MulticlassRecall
# import torch.nn.functional as F
import gradio as gr
import os
from model_classes import model,agg_models
import agg_strats
from common.env_path_fns import load_env_var
from datetime import datetime,timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager

model_ver=0
g_model=model()
UPLOAD_MODEL_DIR=load_env_var('AGG_SERVER_UPLOAD_MODEL_DIR','path')
RUNS=load_env_var('AGG_SERVER_RUNS','int')
UPDATE_FREQ=load_env_var('AGG_SERVER_UPDATE_FREQ','int')
models={i:agg_models(
    dir=f'{os.path.join(UPLOAD_MODEL_DIR,str(i))}',
    agg_strat=agg_strats.fed_avg,)
      for i in range(RUNS)}
models[0].add_global_model(g_model)
GLOBAL_MODEL_DIR=load_env_var('AGG_SERVER_GLOBAL_MODEL_DIR','path')
gen_update_time=lambda :datetime.now()+timedelta(seconds=UPDATE_FREQ)
g_update_time=gen_update_time()
periodic_event_scheduler=BackgroundScheduler(executor=ThreadPoolExecutor)
model_counts=[0 for _ in range(RUNS)]
model_update_status=""

gr.set_static_paths(paths=[
    os.path.join(os.getcwd(),GLOBAL_MODEL_DIR),
    os.path.join(os.getcwd(),UPLOAD_MODEL_DIR)
])

def update_model() ->str:
    global g_update_time,model_update_status
    time_to_update=g_update_time-datetime.now()
    if time_to_update<timedelta(seconds=0):
        updated=agg_weights()
        if updated:
            model_update_status=f'global model updated'
        else:
            g_update_time=gen_update_time()
            model_update_status=f'not enough models, waiting till {g_update_time} for update.'
    else:
        model_update_status=f'global model to be updated in :{time_to_update.seconds} seconds.'
    
def get_model_weights() ->str:
    global model_ver
    f_loc=os.path.join(GLOBAL_MODEL_DIR,f'{model_ver}.pth')
    torch.save(g_model.state_dict(),f_loc)
    print(f"current modelfile: {f_loc}")
    return f_loc

def get_model_ver()-> int:
    global model_ver
    print(f'model_version:{model_ver}')
    return model_ver

def upload_model(model_loc:str,counts:int) -> None:
    print('uploaded data:',model_loc,counts)
    global model_ver
    models[model_ver].add_client_model(model_loc)

def agg_weights() -> bool:
    global model_ver
    g_model,update=models[model_ver].update()
    if update:
        model_ver+=1
        models[model_ver].add_global_model(g_model)
        return update
    return False

def get_models() -> None:
    global model_counts
    model_counts=[len(models[i].models) for i in range(RUNS)]
    # return len(models[get_model_ver()].models)

periodic_event_scheduler.add_job(update_model,trigger='interval',seconds=1)
periodic_event_scheduler.add_job(get_model_ver,trigger='interval',seconds=1)
periodic_event_scheduler.add_job(get_model_weights,trigger='interval',seconds=1)
periodic_event_scheduler.add_job(get_models,trigger='interval',seconds=1)

with gr.Blocks() as demo:
    gr.Number(value=lambda :model_ver,every=5,label='get_model_ver')
    gr.File(value=get_model_weights,every=5,label='get_model_weights')
    gr.Number(value=lambda x=3:model_counts[model_ver],every=1,label='models uploaded')
    gr.Text(value=lambda x=3:model_update_status,every=1,label='model update info')
    new_model=gr.File(label='upload_model')
    steps=gr.Number(value=0)
    upload_submit_btn=gr.Button(value='Upload')
    upload_submit_btn.click(
        upload_model,
        [new_model,steps]
    )
#needed for scheduler to stop at the end.
async def lifespan(app:FastAPI):
    periodic_event_scheduler.start()
    yield
    periodic_event_scheduler.shutdown()
app=FastAPI(lifespan=lifespan)
app=gr.mount_gradio_app(app,demo,path='/')
uvicorn.run(app,host='0.0.0.0',port=load_env_var('AGG_SERVER_PORT','int'))
# demo.launch(
#     server_name='0.0.0.0',
#     server_port=load_env_var('AGG_SERVER_PORT','int'),
#     )

# periodic_event_scheduler.start()