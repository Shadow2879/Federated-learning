'''
Defines a lightning model in for classification tasks on 28x28 images
'''
import torch
import gradio as gr
import os,uvicorn
from aggregator import agg_models
import agg_strats
from common.env_path_fns import load_env_var
from datetime import datetime,timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor,ProcessPoolExecutor
from fastapi import FastAPI
from contextlib import asynccontextmanager
from common.models import LitEMNISTClassifier,model,PartialEMNISTDataModule
from torchmetrics.classification import MulticlassAccuracy,MulticlassRecall,MulticlassF1Score,MulticlassPrecision
import lightning as L
import numpy as np
import pandas as pd

UPLOAD_MODEL_DIR=load_env_var('AGG_SERVER_UPLOAD_MODEL_DIR','path')
RUNS=load_env_var('AGG_SERVER_RUNS','int')
UPDATE_FREQ=load_env_var('AGG_SERVER_UPDATE_FREQ','int')
DATA_SERVER_ADDR=load_env_var('DATA_SERVER_ADDR','addr',port_key='DATASET_SERVER_PORT')
OUTPUT_CLASSES=load_env_var('OUTPUT_CLASSES','int')
GLOBAL_MODEL_DIR=load_env_var('AGG_SERVER_GLOBAL_MODEL_DIR','path')
SERVER_PORT=load_env_var('AGG_SERVER_PORT','int')
DATA_DELAY=load_env_var('AGG_SERVER_CONNECTION_DELAY','int')
DATA_TRIES=load_env_var('AGG_SERVER_CONNECTION_TRIES','int')
DATA_LOC=load_env_var('AGG_SERVER_DATA_LOC','path')
DATA_BATCH_SIZE=load_env_var('AGG_SERVER_BATCH_SIZE','int')
DATA_WORKERS=load_env_var('AGG_SERVER_DATA_WORKERS','int')
model_ver=0
g_model=model()
models={i:agg_models(
    dir=f'{os.path.join(UPLOAD_MODEL_DIR,str(i))}',
    classes=OUTPUT_CLASSES,
    agg_strat=agg_strats.fed_avg_weighted_steps,)
      for i in range(RUNS)}
models[0].add_global_model(g_model)
gen_update_time=lambda :datetime.now()+timedelta(seconds=UPDATE_FREQ)
g_update_time=gen_update_time()
periodic_event_scheduler=BackgroundScheduler(
    default_executor='threadpool',
    executors={'threadpool':ThreadPoolExecutor(2),'processpool':ProcessPoolExecutor(1)})
model_counts=pd.DataFrame(np.zeros(RUNS),columns=['uploaded models']).T
model_update_status=""
run_op=''
gr.set_static_paths(paths=[
    os.path.join(os.getcwd(),GLOBAL_MODEL_DIR),
    os.path.join(os.getcwd(),UPLOAD_MODEL_DIR)
])

def update_model() ->None:
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
    # print(f"current modelfile: {f_loc}")
    return f_loc

def get_model_ver()-> int:
    global model_ver
    # print(f'model_version:{model_ver}')
    return model_ver

def upload_model(model_loc:str,counts:int) -> None:
    # print('uploaded data:',model_loc,counts)
    global model_ver
    models[model_ver].add_client_model(model_loc)

def agg_weights() -> bool:
    global model_ver,g_model
    g_model,update=models[model_ver].update()
    if update:
        model_ver+=1
        models[model_ver].add_global_model(g_model)
    return update

def get_models() -> None:
    global model_counts
    for i in range(RUNS):
        model_counts[i]=len(models[i].models)

def test_run() -> None:
    print("Periodic model evaluation started")
    global model_ver,run_op
    litmodel=LitEMNISTClassifier(get_model_weights(),
                        model_ver,
                        metrics=[MulticlassAccuracy,
                                 MulticlassF1Score,
                                 MulticlassPrecision,
                                 MulticlassRecall],
                        output_classes=OUTPUT_CLASSES,
                        fed_learning=True,)
    litdata=PartialEMNISTDataModule(DATA_LOC,
                                    DATA_SERVER_ADDR,
                                    splits=[0.,0.,1.],
                                    delay=DATA_DELAY,
                                    tries=DATA_TRIES,
                                    batch_size=DATA_BATCH_SIZE,
                                    cpus=DATA_WORKERS)
    trainer=L.Trainer(
        max_epochs=1,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    run_op=trainer.test(litmodel,datamodule=litdata)

periodic_event_scheduler.add_job(update_model,trigger='interval',seconds=1)
periodic_event_scheduler.add_job(get_model_ver,trigger='interval',seconds=1)
periodic_event_scheduler.add_job(get_model_weights,trigger='interval',seconds=1)
periodic_event_scheduler.add_job(get_models,trigger='interval',seconds=1)
periodic_event_scheduler.add_job(test_run,trigger='interval',seconds=UPDATE_FREQ,max_instances=3,executor='threadpool')

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Number(value=get_model_ver,every=5,label='get_model_ver')
            gr.File(value=get_model_weights,every=5,label='get_model_weights')
        with gr.Column():
            gr.Dataframe(value=lambda:model_counts,every=1,label='models uploaded',datatype='number',type='numpy',max_height=250)
            gr.Text(value=lambda:model_update_status,every=1,label='model update info')
        with gr.Column():
            new_model=gr.File(label='upload_model')
            steps=gr.Number(value=0)
            upload_submit_btn=gr.Button(value='Upload')
    upload_submit_btn.click(
        upload_model,
        [new_model,steps]
    )
    with gr.Row():
        gr.TextArea(run_op)
        
#needed for scheduler to stop at the end.
@asynccontextmanager
async def lifespan(app:FastAPI):
    print('starting server')
    periodic_event_scheduler.start()
    yield
    print('stopping server')
    periodic_event_scheduler.shutdown(wait=True)

app=FastAPI(lifespan=lifespan)
app=gr.mount_gradio_app(app,demo,path='/')
uvicorn.run(app,host='0.0.0.0',port=SERVER_PORT)
# add code in order to run test runs in the background while waiting for update using background scheduler +process pool exec.
# integrate it with the frontend.