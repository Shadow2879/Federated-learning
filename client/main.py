from common.models import LitEMNISTClassifier,PartialEMNISTDataModule
import lightning as L, torch, os
from lightning.pytorch.callbacks import ModelSummary,EarlyStopping,Timer
from common.env_path_fns import load_env_var
from common.connect import connect_to_gr_client
from gradio_client import handle_file
from datetime import timedelta
from lightning.pytorch.loggers import MLFlowLogger

torch.set_float32_matmul_precision('medium')
L.seed_everything(42,workers=True)

DEBUG_MODE=load_env_var('DEBUG_MODE','bool')
TRAIN_TIME=load_env_var('CLIENT_TRAIN_DURATION','int')
COMBINE_STEPS=load_env_var('CLIENT_COMBINE_STEPS','int')
CLIENT_CONNECTION_TRIES=load_env_var('CLIENT_CONNECTION_TRIES','int')
CLIENT_CONNECTION_DELAY=load_env_var('CLIENT_CONNECTION_DELAY','int')
DS_LOC=load_env_var('CLIENT_DATA_LOC','path')
DATA_SPLITS=load_env_var('CLIENT_DATA_SPLITS','array')
DATA_WORKERS=load_env_var('CLIENT_WORKERS','int')
OUTPUT_CLASSES=load_env_var('OUTPUT_CLASSES','int')
MLFLOW_TAG=load_env_var('CLIENT_MLFLOW_TAG','str')
CLIENT_BATCH_SIZE=load_env_var('CLIENT_BATCH_SIZE','int')
MLFLOW_EXP_NAME=load_env_var('MLFLOW_EXP_NAME','str')
DEPLOY=load_env_var('DEPLOY','bool')
DATA_SERVER_ADDR,DATASET_SERVER_PORT=load_env_var('DATA_SERVER_ADDR','addr',port_key='DATASET_SERVER_PORT')
AGG_SERVER_ADDR,AGG_SERVER_PORT=load_env_var('AGG_SERVER_ADDR','addr',port_key='AGG_SERVER_PORT')
MLFLOW_TRACKING_URI,MLFLOW_SERVER_PORT=load_env_var('MLFLOW_TRACKING_URI','addr',port_key='MLFLOW_SERVER_PORT')

if DEPLOY:
    print('configured to pull/push data from containers.')
else:
    src='http://localhost:'
    print(f'configured to push/pull data from {src}')
    DATA_SERVER_ADDR=src+DATASET_SERVER_PORT
    AGG_SERVER_ADDR=src+AGG_SERVER_PORT
    MLFLOW_TRACKING_URI=src+MLFLOW_SERVER_PORT
    os.environ['MLFLOW_TRACKING_URI']=MLFLOW_TRACKING_URI
    os.environ['DATA_SERVER_ADDR']=DATA_SERVER_ADDR
    os.environ['AGG_SERVER_ADDR']=AGG_SERVER_ADDR


data=PartialEMNISTDataModule(
    DS_LOC,
    DATA_SERVER_ADDR,
    DATA_SPLITS,
    delay=CLIENT_CONNECTION_DELAY,
    tries=CLIENT_CONNECTION_TRIES,
    batch_size=CLIENT_BATCH_SIZE,
    cpus=DATA_WORKERS
)
server=connect_to_gr_client(
    AGG_SERVER_ADDR,
    CLIENT_CONNECTION_DELAY,
    CLIENT_CONNECTION_TRIES,
    )
if DEBUG_MODE:
    print(os.environ.items())
    print(server.view_api())
    
model_metrics={}
g_model_ver=lambda :server.predict(api_name='/get_model_ver')
g_model_file=lambda :server.predict(api_name='/get_model_weights')
g_model_sdict=lambda :torch.load(g_model_file())
c_model=LitEMNISTClassifier(g_model_file(),g_model_ver(),output_classes=OUTPUT_CLASSES,batch_size=CLIENT_BATCH_SIZE)
logger=MLFlowLogger(MLFLOW_EXP_NAME,
                    tracking_uri=MLFLOW_TRACKING_URI,
                    tags={'device':MLFLOW_TAG,'cuda':str(torch.cuda.device_count())},
                    synchronous=False,)

for i in range(COMBINE_STEPS):
    while(c_model.model_ver<=i):
        trainer=L.Trainer(
            callbacks=[
                EarlyStopping('val_loss_epoch',patience=4),
                ModelSummary(max_depth=-1),
                Timer(duration=timedelta(seconds=TRAIN_TIME),interval="step")
                ],
                log_every_n_steps=1,
                num_sanity_val_steps=2,
                logger=logger
        )
        trainer.fit(c_model,datamodule=data)
        model_metrics[f'{c_model.model_ver}_train']=trainer.callback_metrics
        trainer.validate(c_model,datamodule=data)
        model_metrics[f'{c_model.model_ver}_val']=trainer.callback_metrics
        torch.save(c_model.model.state_dict(),f'/workspace/{c_model.model_ver}.pth')
        server.predict(handle_file(f'/workspace/{c_model.model_ver}.pth'),trainer.global_step,api_name='/upload_model')
        
    else:
        c_model.model.load_state_dict(g_model_sdict())
        c_model.model_ver=g_model_ver()

final_model=LitEMNISTClassifier(g_model_file(),g_model_ver(),output_classes=OUTPUT_CLASSES,batch_size=CLIENT_BATCH_SIZE)
trainer=L.Trainer(logger=logger)
trainer.test(final_model,datamodule=data)
model_metrics[f'{final_model.model_ver}_test']=trainer.callback_metrics
print(model_metrics)
trainer.strategy.connect(final_model)
trainer.save_checkpoint('./final_model.ckpt')