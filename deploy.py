'''reads the .env file, creates and writes .env files in various dirs specified by the .env file, copies the common dir to each of those dirs and runs docker compose'''

import os
import shutil
import subprocess

def create_env_file(folder:str,envs:list[str],mock=False,remove=False):
    if 'global' in folder:
        folder=''
    path=os.path.join(os.getcwd(),folder,'.env')
    if mock:
        print(path)
        for i in envs:
            print(i)
    else:
        if remove:
            os.remove(path)
        with open(os.path.join(path,'w')) as f:
            for i in envs:
                f.write(i+'\n')
def split_envs(lines:list[str]):
    idxs=[i[0] for i in enumerate(lines) if i[1].startswith('#')]
    res=[lines[idxs[-1]+1:] for _ in range(len(idxs))]
    for i,j,k in zip(idxs[:-1],idxs[1:],range(len(idxs))):
        [res[k].append(t) for t in lines[i+1:j]]
    # res.append(lines[idxs[-1]:])
    return [lines[i].replace('#','') for i in idxs],res
def read_envs(loc='.envs'):
    lines=[]
    with open(loc) as f:
        data=f.readlines()
        for line in data:
            line=line.replace('\n','').replace(' ','')
            if len(line):
                lines.append(line)
    return lines

if __name__ =="__main__":
    MOCK=True
    REMOVE=False
    lines=read_envs()
    folders,vars=split_envs(lines)
    for i in zip(folders,vars):
        create_env_file(i[0],i[1],mock=MOCK,remove=REMOVE)
        if 'global' in i[0]:
            continue
        if REMOVE:
            shutil.rmtree(os.path.join(folders,'common'))
        if not MOCK:
            shutil.copytree('./common',os.path.join(os.getcwd(),i[0],'common'))
    if not MOCK:
        subprocess.call(["docker","compose","up"])