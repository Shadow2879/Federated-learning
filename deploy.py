'''reads the .env file, creates and writes .env files in various dirs specified by the .env file, copies the common dir to each of those dirs and runs docker compose'''

import os
import shutil
import subprocess
import argparse

def create_env_file(folder:str,envs:list[str],mock=False,remove=False,printenvs=False,showpaths=False):
    if printenvs:
        print(folder+' env vars')
        [print(i) for i in envs]
        print('\n',end='')
    if 'global' in folder:
        folder=''
    path=os.path.join(os.getcwd(),folder,'.env')
    if showpaths:
        print('removed: ' if remove else 'created: ',path)
    if mock:
        return
    else:
        if remove:
            os.remove(path)
            return
        with open(path,'w') as f:
            for i in envs:
                f.write(i+'\n')

def split_envs(lines:list[str],printenvs=True):
    idxs=[i[0] for i in enumerate(lines) if i[1].startswith('#')]
    res=[lines[idxs[-1]+1:] for _ in range(len(idxs))]
    for start,end,k in zip(idxs[:-1],idxs[1:],range(len(idxs))):
        [res[k].append(t) for t in lines[start+1:end]]
    if printenvs:
        print('env variables identified:\n',res,'\n')
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

if __name__=="__main__":
    parser=argparse.ArgumentParser(description='A script which handles setting up and starting the project.')
    parser.add_argument('--showpaths',action='store_true',help='Show paths of files which would be created')
    parser.add_argument('--printenvs',action='store_true',help='Show env variables')
    parser.add_argument('--rm',action='store_true',help='Remove created files')
    parser.add_argument('-d','--deploy',action='store_true',help='Start project deployment')
    parser.add_argument('-m','--mock',action='store_true',help='Dont create or remove files.')
    args=parser.parse_args()
    print('User inputs:',str([str(i[0])+':'+str(i[1]) for i in vars(args).items()])[1:-1])
    SHOWPATHS=args.showpaths
    REMOVE=args.rm
    DEPLOY=args.deploy
    PRINTENVS=args.printenvs
    MOCK=args.mock
    if MOCK:
        print('Mock run')
    lines=read_envs()
    folders,vars=split_envs(lines,PRINTENVS)
    for i in zip(folders,vars):
        create_env_file(i[0],i[1],remove=REMOVE,printenvs=PRINTENVS,showpaths=SHOWPATHS,mock=MOCK)
        if 'global' in i[0]:
            continue
        path=os.path.join(os.getcwd(),i[0],'common')
        if REMOVE:
            if SHOWPATHS:
                print('removed: ',path)
            if not MOCK:
                shutil.rmtree(path)
        if not REMOVE and not MOCK:
            print(f'copied ./common to {path}')
            shutil.copytree('./common',path)
    if DEPLOY and not MOCK:
        subprocess.call(["docker","compose","build","--parallel"])