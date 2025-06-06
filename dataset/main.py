import gradio as gr
import os
import torch
import time
import shutil
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Compose,Normalize
from common.env_path_fns import load_env_var
torch.set_float32_matmul_precision('medium')

DEBUG_MODE=load_env_var('DEBUG_MODE','bool')
DS_LOC=load_env_var('DATASET_LOC','path')
FILE_DIR=load_env_var('DATASET_FILE_DIR','path')
BS=load_env_var('DATASET_BATCH_SIZE','int')
BF=load_env_var('DATASET_BATCH_FACTOR','int')
DATA_SERVER_ADDR,SERVER_PORT=load_env_var('DATA_SERVER_ADDR','addr','DATA_SERVER_PORT')
ds=EMNIST(root=DS_LOC,split='byclass',download=True,transform=Compose([ToTensor(),Normalize(0.1736,0.3316)]))
server_ds=EMNIST(root=DS_LOC,split='byclass',download=True,transform=Compose([ToTensor(),Normalize(0.1736,0.3316)]),train=False)
dl=DataLoader(ds,batch_size=BS,shuffle=True,)
server_dl=DataLoader(server_ds,batch_size=BS*16,shuffle=True)
runs=0
is_server=False

if DEBUG_MODE:
    print(os.environ.items())
    print(len(ds),len(server_ds))

def serve_client() -> str | None:
    num_data_batches=int(len(ds)*torch.rand(1)[0]//(BS*BF))
    global runs,is_server
    runs+=1
    f_name=''
    if runs-1:# prevent generating file before starting server
        res=[]
        if is_server:
            f_name=os.path.join(FILE_DIR,'server.pt')
            if 'server.pt' not in os.listdir(FILE_DIR):
                for data,_ in zip(iter(server_dl),range(1_000_000)):
                    res.append(data)
                res=[torch.concat([p[0] for p in res]),torch.concat([p[1] for p in res])]
                is_server=False
        else:
            for data,_ in zip(iter(dl),range(num_data_batches-1)):
                res.append(data)
            res=[torch.concat([p[0] for p in res]),torch.concat([p[1] for p in res])]
            f_name=f'{num_data_batches}_{time.strftime(time.ctime())}.pt'
            f_name=f_name.replace('  ',' ').replace(' ','_')
            f_name=os.path.join(FILE_DIR,f_name)
        print(f_name)
        torch.save(res,f_name)
        print(num_data_batches)
        return f_name
    return None

def set_server(y:bool) -> None:
    global is_server
    is_server=bool(y)
    print(is_server)
with gr.Blocks() as demo:
    gr.File(serve_client,file_count='single',label='data')
    t=gr.Text(label='server',visible=False,)
    t.change(
        set_server,
        inputs=[t],
    )
demo.launch(server_port=int(SERVER_PORT),server_name='0.0.0.0')
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