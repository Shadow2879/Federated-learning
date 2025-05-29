from data import PartialEMNISTDataModule
from model import LitEMNISTClassifier
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelSummary,EarlyStopping,Timer
import os
from common.env_path_fns import load_env_var
from common.connect import connect_to_gr_client
from gradio_client import handle_file
from datetime import timedelta

torch.set_float32_matmul_precision('medium')
L.seed_everything(42,workers=True)
data=PartialEMNISTDataModule()
model_metrics={}
DEBUG_MODE=load_env_var('CLIENT_DEBUG_GR_CLIENT','int')
TRAIN_TIME=load_env_var('CLIENT_TRAIN_DURATION','int')
COMBINE_STEPS=load_env_var('CLIENT_COMBINE_STEPS','int')
CLIENT_CONNECTION_TRIES=load_env_var('CLIENT_CONNECTION_TRIES','int')
CLIENT_CONNECTION_DELAY=load_env_var('CLIENT_CONNECTION_DELAY','int')
server=connect_to_gr_client(
    load_env_var('CLIENT_AGG_SERVER_ADDR','addr','AGG_SERVER_PORT'),
    CLIENT_CONNECTION_DELAY,
    CLIENT_CONNECTION_TRIES,
    )
if DEBUG_MODE:
    print(os.environ.items())
    print(server.view_api())
g_model_ver=lambda :server.predict(api_name='/get_model_ver')
g_model_file=lambda :server.predict(api_name='/get_model_weights')
g_model_sdict=lambda :torch.load(g_model_file())
c_model=LitEMNISTClassifier(g_model_file(),g_model_ver(),output_classes=62)
for i in range(COMBINE_STEPS):
    while(c_model.model_ver<=i):
        trainer=L.Trainer(
            # accelerator="cpu",
            callbacks=[
                EarlyStopping('val_loss',patience=4),
                ModelSummary(max_depth=-1),
                Timer(duration=timedelta(seconds=TRAIN_TIME),interval="step")
                ],
                num_sanity_val_steps=2,
                # profiler='simple'
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

final_model=LitEMNISTClassifier(g_model_file(),g_model_ver(),output_classes=62)
trainer=L.Trainer()
trainer.test(final_model,datamodule=data)
model_metrics[f'{final_model.model_ver}_test']=trainer.callback_metrics
print(model_metrics)
trainer.strategy.connect(final_model)
trainer.save_checkpoint('./final_model.ckpt')