'''
    handles loading .env variables 
'''
import os
import dotenv
from typing import Literal
def get_path(loc):
    return os.path.join(os.getcwd(),loc)

def ensure_dir_exists(loc):
    os.makedirs(loc,exist_ok=True)

def gen_var_path(var):
    path=get_path(var) 
    ensure_dir_exists(path)
    return path

def defaults(var,type):
    if var is None:
        match type:
            case "int":
                var=1
            case "path":
                var='/'
            case "str":
                var=''
            case "array":
                var=''
            case default:
                raise NotImplementedError(f'no default value for  type "{type}" env vars')
    return var

def load_env_var(key:str,type:Literal['int','path','str','array','addr'],port_key:str | None=None,arr_sep=',',) ->int | str | list[str]:
    '''
    loads an env variable and transfroms it based on type.

    Params:
        key: the name of the environment variable.
        type: the type of the variable. This determines how the variable is transformed before being returned.

                if it is an integer, it is casted to int and returned (defaults to 1).

                if it is a path, the path is merged with the current working directory and a folder is created at the location of the merged path (defaults to cwd).

                if it is a string, the variable is returned as is (defaults to an empty string).

                if it is an array, the variable is returned as an array by using an appropriate separator as defined by arr_sep (defaults to empty list).

                if it is an address, the variable along with its port are combined to give back an address.
                
        arr_sep (optional): the separator to use when splitting the array.
        port_key (optional): the envrion key which contains the port of the address.
    '''
    dotenv.load_dotenv()
    var=os.environ.get(f'{key}')

    match type:
        case "int":
            var=int(defaults(var,type))
        case 'path':
            gen_var_path(defaults(var,type))
        case 'array':
            var=defaults(var,type).split(arr_sep)
        case 'str':
            var=defaults(var,type)
        case "addr":
            var=defaults(var,type)
            addr_port=load_env_var(port_key,'str')
            var=var+':'+addr_port
        case default:
            raise NotImplementedError(f'loading env vars with type {type} is not yet implemented')
        
    if isinstance(var,int):
        return var
    elif isinstance(var,str):
        return var
    else:
        return var