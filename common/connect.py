from gradio_client import Client
import time
from httpcore import ConnectError

def connect_to_gr_client(address:str,delay:int=10,tries:int=10,*args,**kwargs) -> Client:
    for i in range(tries):
        print(f'Attempt {i+1} to connect to {address}')
        try:
            client=Client(address,*args)
            return client
        except ConnectError as e:
            print(f"Connection refused, trying again in {delay} seconds.")
            time.sleep(delay)
            continue
    raise ConnectionRefusedError(f'Attempting to connect to {address} failed after {tries*delay} seconds.')