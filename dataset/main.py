import gradio as gr, os, torch, time, shutil, gc
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Compose,Normalize
from common.env_path_fns import load_env_var
from tqdm import tqdm
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
runs=torch.zeros(2,dtype=torch.int32)

if DEBUG_MODE:
    print(os.environ.items())
    print(len(ds),len(server_ds))

def serve_client(progress=gr.Progress(track_tqdm=True)) -> str|None:
    num_data_batches=int(len(ds)*torch.rand(1)[0]//(BS*BF))
    global runs
    print(runs)
    f_name=None
    res=[]
    with tqdm(range(num_data_batches*BS),desc='generating client file') as pbar:
        for data,_ in zip(iter(dl),range(num_data_batches-1)):
            res.append(data)
            pbar.update(BS)
    res=[torch.concat([p[0] for p in res]),torch.concat([p[1] for p in res])]
    f_name=f'{num_data_batches}_{time.strftime(time.ctime())}.pt'
    f_name=f_name.replace('  ',' ').replace(' ','_')
    f_name=os.path.join(FILE_DIR,f_name)
    print(f'{num_data_batches} for client generated.')
    print(f'generated file: {f_name}')
    torch.save(res,f_name)
    del res
    gc.collect()
    runs[0]+=1
    return f_name

def serve_server(progress=gr.Progress(track_tqdm=True)) -> str | None:
    global runs
    print(runs)
    f_name=None
    res=[]
    fdir='/'.join(FILE_DIR.split('/')[:-2])
    f_name=os.path.join(fdir,'server.pt')
    if 'server.pt' not in os.listdir(fdir):
        with tqdm(range(len(server_ds)),desc='generating server file') as pbar:
            for data,_ in zip(iter(server_dl),range(1_000_000)):
                res.append(data)
                pbar.update(BS*16)
        res=[torch.concat([p[0] for p in res]),torch.concat([p[1] for p in res])]
        torch.save(res,f_name)
        print(f'generated file: {f_name}')
    del res
    gc.collect()
    runs[1]+=1
    return f_name

def get_client_runs():
    return runs[0].item()

def get_server_runs():
    return runs[1].item()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            client_file=gr.File(None,file_count='single',label='data_client')
        with gr.Column():
            client_runs=gr.Number(get_client_runs,every=1,label='client data requests')
    with gr.Row():
        client_btn=gr.Button(value='get client data')
        client_btn.click(
            serve_client,
            inputs=None,
            outputs=[client_file],
            api_name='serve_client',
        )
    with gr.Row():
        with gr.Column():
            server_file=gr.File(None,file_count='single',label='data_server')
        with gr.Column():
            server_runs=gr.Number(get_server_runs,every=1,label='server data requests')
    with gr.Row():
        server_btn=gr.Button(value='get server data')
        server_btn.click(
            serve_server,
            inputs=None,
            outputs=[server_file]
        )
        
demo.queue().launch(server_port=int(SERVER_PORT),server_name='0.0.0.0')
print(f'server data requests: {runs[0]}, client data requests: {runs[1]}')

#clear previously generated data files if any
for file_name in os.listdir(FILE_DIR):
    file_path=os.path.join(FILE_DIR,file_name)
    if 'server.pt' not in file_path:
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

print('client files deleted')