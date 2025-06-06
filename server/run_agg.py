import subprocess
from common.env_path_fns import load_env_var
def start_mlflow_server(port:str | int):
    proc=subprocess.run(['mlflow', 'server', '-h', '0.0.0.0', '-p', str(port)])
if __name__=='__main__':
    addr,port=load_env_var('MLFLOW_TRACKING_URI','addr','MLFLOW_PORT')
    print('launching on:',port)
    start_mlflow_server(port)
