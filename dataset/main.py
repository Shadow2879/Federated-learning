import torchvision
import gradio as gr
import os
import torch
import time
import shutil
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
from common.env_path_fns import load_env_var
DS_LOC=load_env_var('DATASET_LOC','path')
FILE_DIR=load_env_var('DATASET_FILE_DIR','path')
BS=load_env_var('DATASET_BATCH_SIZE','int')
BF=load_env_var('DATASET_BATCH_FACTOR','int')
SERVER_PORT=load_env_var('DATASET_SERVER_PORT','int')
ds=EMNIST(root=DS_LOC,split='byclass',download=True,transform=torchvision.transforms.ToTensor())
server_ds=EMNIST(root=DS_LOC,split='byclass',download=True,transform=torchvision.transforms.ToTensor(),train=False)
dl=DataLoader(ds,batch_size=BS,shuffle=True,)
server_dl=DataLoader(server_ds,batch_size=BS,shuffle=False)
runs=0

def serve_client(num_data_batches=int(len(ds)*torch.rand(1)[0]//(BS*BF)),server=False):
    global runs
    runs+=1
    if runs-1:# prevent generating file before starting server
        if server:
            f_name=os.path.join(FILE_DIR,'server.pt')
        #     res=[next(iter(server_dl))]
        #     for _ in range(len(server_dl)):
        #         res
        #     torch.save()
        # print(num_data_batches)
        res=[next(iter(dl))]
        for _ in range(num_data_batches-1):
            res.append(next(iter(dl)))
        res=[torch.concat([p[0] for p in res]),torch.concat([p[1] for p in res])]
        f_name=f'{num_data_batches}_{time.strftime(time.ctime())}.pt'
        f_name.replace(' ','_')
        f_name=os.path.join(FILE_DIR,f_name)
        print(f_name)
        torch.save(res,f_name)
        return f_name
    
with gr.Blocks() as demo:
    gr.File(serve_client,file_count='single',label='data')
demo.launch(server_port=SERVER_PORT,server_name='0.0.0.0')
print(f'data requests: {runs-1}')


#clear previously generated data files if any
for file_name in os.listdir(FILE_DIR):
    file_path=os.path.join(FILE_DIR,file_name)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

print('all files deleted')