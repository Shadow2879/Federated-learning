'''
    handles loading .env variables 
'''
import os
import dotenv
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
    dotenv.load_dotenv('.envs')
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
def load_env_var(key:str,type_:Literal['int'])->int:
    ...
@overload
def load_env_var(key:str,type_:Literal['str','path','addr'])->str: ...
@overload
def load_env_var(key:str,type_:Literal['array'],sep:str=',')->list[str]: ...
@overload
def load_env_var(key:str,type_:Literal['addr'],sep:str=',',port_key:str='')->str: ...

def load_env_var(key:str,type_,sep:str=',',port_key:str='') -> int | str | list[str] | bool:
    '''
    loads an env variable and transfroms it based on type_.

    Params:
        key: the name of the environment variable.
        type_: the type_ of the variable. This determines how the variable is transformed before being returned.

                if it is an integer, it is casted to int and returned (defaults to 1).

                if it is a path, the path is merged with the current working directory and a folder is created at the location of the merged path (defaults to cwd).

                if it is a string, the variable is returned as is (defaults to an empty string).

                if it is an array, the variable is returned as an array by using an appropriate separator as defined by arr_sep (defaults to empty list).

                if it is an address, the variable along with its port are combined to give back an address.
                
        arr_sep (optional): the separator to use when splitting the array.
        port_key (optional): the envrion key which contains the port of the address.
    '''
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
            p=load_env_var(port_key,'str')
            var+=':'+p
        case 'path':
            var=gen_var_path(var)
        case default:
            raise NotImplementedError(f'loading env vars with type {type_} is not yet implemented')
    return var
