'''
    handles connecting to a gradio server using gradio client.
'''
from gradio_client import Client
import time

def connect_to_gr_client(address:str,delay:int | float=10,tries:int=10,*args,**kwargs) -> Client:
    for i in range(tries):
        print(f'Attempt {i+1}/{tries} to connect to {address}')
        try:
            client=Client(address,*args,**kwargs)
            return client
        except Exception as e:
            print(f"{e}, trying again in {delay} seconds.")
            time.sleep(delay)
            continue
    raise ConnectionAbortedError(f'Attempting to connect to {address} failed after {tries*delay} seconds.')