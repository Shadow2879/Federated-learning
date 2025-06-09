'''
    handles loading .env variables 
'''
import os, dotenv
from typing import Literal,overload,Any

def get_path(loc):
    return os.path.join(os.getcwd(),loc)

def ensure_dir_exists(loc):
    os.makedirs(loc,exist_ok=True)

def gen_var_path(var):
    path=get_path(var) 
    ensure_dir_exists(path)
    return path

def get_var(key:str):
    dotenv.load_dotenv()
    var=os.environ.get(key)
    if isinstance(var,str):
        return var
    raise KeyError(f'{key} was not found')

@overload
def defaults(var:Any,type_:Literal['int']) -> int: ...
@overload
def defaults(var:Any,type_:Literal['str','path','array']) -> str: ...
@overload
def defaults(var:Any,type_:Literal['bool']) -> bool: ...

def defaults(var:Any,type_) -> int | str | bool:
    if var is None:
        match type_:
            case "int":
                var=1
            case "path":
                var=os.getcwd()
            case "str":
                var=''
            case "array":
                var=''
            case "bool":
                var=False
            case default:
                raise NotImplementedError(f'no default value for type "{type_}" env vars implemented')
    return var

@overload
def load_env_var(key:str,type_:Literal['int']) -> int: 
    '''
    Loads a .env variable, converts it into an `int` and returns it.
    '''
    ...
@overload
def load_env_var(key:str,type_:Literal['bool']) -> bool: 
    '''
    Loads a .env variable, converts it into a `bool` and returns it.
    '''
    ...
@overload
def load_env_var(key:str,type_:Literal['path']) -> str: 
    '''
    Loads a .env variable, joins it with `os.getcwd()`.
    A directory at that path is created and the path is returned. 
    '''
    ...
@overload
def load_env_var(key:str,type_:Literal['str']) -> str:
    '''
    Loads a .env variable and returns it.
    '''
    ...
@overload
def load_env_var(key:str,type_:Literal['array'],sep:str=',') -> list[str]: 
    '''
    Loads a .env variable, splits it based on `sep` and returns the `list`.
    '''
    ...
@overload
def load_env_var(key:str,type_:Literal['addr'],sep:str=',',port_key:str='') -> tuple[str,str]: 
    '''
    Loads a .env variable, creates another variable with `port_key=addr port` and returns the full address along as well as the port.
    '''
    ...

def load_env_var(key:str,type_,sep:str=',',port_key:str='') -> int | str | list | bool | tuple:
    try:
        var=get_var(key)
    except KeyError as e:
        print(e)
        var=None
    if isinstance(var,type(None)):
        var=defaults(var,type_)
    match type_:
        case 'int':
            var=int(var)
        case 'bool':
            tf_str={'True':['true','y','yes','1','t'],'False':['false','n','no','0','f']}
            if var==1 or var in tf_str['True']:
                var=True
            elif var==0 or var in tf_str['False']:
                var=False
            else:
                raise TypeError(f'failed to convert {var} to {type_}')
        case 'str':
            pass
        case 'array':
            var=var.split(sep)
        case 'addr':
            print(var)
            p=var[-(str.index(var,':')):]
            if port_key != "":
                os.environ[port_key]=p
            return var,p
        case 'path':
            var=gen_var_path(var)
        case default:
            raise NotImplementedError(f'loading env vars with type {type_} is not yet implemented')
    return var
