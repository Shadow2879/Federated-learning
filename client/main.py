from data import PartialEMNISTDataModule
from model import LitEMNISTClassifier
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelSummary,EarlyStopping,Timer
import time
import os
from common.env_path_fns import load_env_var

torch.set_float32_matmul_precision('medium')
L.seed_everything(42,workers=True)
data=PartialEMNISTDataModule()
model=LitEMNISTClassifier()
model_metrics={}
c_model_ver=-1
DEBUG_MODE=load_env_var('DEBUG_CLIENT','int')
if DEBUG_MODE:
    print(os.environ.items())
for _ in range(load_env_var('COMBINE_STEPS','int')):
    while(model.model_ver==c_model_ver):
        model=LitEMNISTClassifier()
        time.sleep(3)
    c_model_ver=model.model_ver
    trainer=L.Trainer(
        min_epochs=3,
        callbacks=[
            EarlyStopping('val_loss',patience=4),
            ModelSummary(max_depth=-1),
            Timer(duration=90,interval="step")
            ],
            num_sanity_val_steps=2,
            profiler='simple'
    )
    trainer.fit(model,datamodule=data)
    model_metrics[f'{model.model_ver}_train']=trainer.callback_metrics
    trainer.validate(model,datamodule=data,return_predictions=False,)
    model_metrics[f'{model.model_ver}_val']=trainer.callback_metrics
    torch.save(model.state_dict(),f'/workspace/{model.model_ver}.pth')
    model.client.predict(f'/workspace/{model.model_ver}.pth',api_name='/upload_model')
trainer=L.Trainer()
trainer.test(model,datamodule=data)
model_metrics[f'{model.model_ver}_test']=trainer.callback_metrics
print(model_metrics)