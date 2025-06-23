import subprocess
import argparse

def get_and_remove_images(search_term:str,image_id:dict,verify:bool) -> None:
    print(f'trying to remove images with {search_term} in the name')
    rm_id={}
    for i in image_id.items():
        if search_term in i[0]:
            rm_id[i[0]]=i[1]
    if len(rm_id)==0:
        print('no images present')
        return
    print(rm_id)
    if verify:
        go_ahead=input(f'remove images with {search_term}? [y/n]')
    else:
        go_ahead='y'
    if go_ahead=='y':
        print('removing images')
        for i in rm_id.items():
            subprocess.call(['docker','rmi',i[1]])
    elif go_ahead=='n':
        print('no images removed')
    else:
        print('please say y or n')
        
class UserInput():
    def __init__(self) -> None:
        pass
    def user_input(self)->str:
        self.var=input('enter the (partial)name of the image(s) you want to remove. provide no input to exit\n')
        return self.var
        
def load_images()->dict:
    result=subprocess.run([f'docker image ls'],stdout=subprocess.PIPE,shell=True)
    images=[]
    image_id={}
    images=str(result.stdout.decode()).split('\n')
    print('available images\n','\n'.join(iter(images)))
    images=images[1:-1]
    data={i:[] for i in range(len(images))}
    for i in enumerate(images):
        text=i[1].split(' ')
        for j in enumerate(text):
            if len(j[1])>=1:
                data[i[0]].append(j[1])
    for i in data.values():
        image_id[':'.join(j for j in i[:2])]=i[2]
    return image_id

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='A script to automate removal of images')
    parser.add_argument('-i','--image',default=None,help='images to remove separated by commas')
    parser.add_argument('-nv','--no-verify',default=True,action='store_false',help='whether to request user verification before removing image')
    args=parser.parse_args()
    image_id=load_images()
    if args.image is None:
        var=UserInput()
        while var.user_input():
            get_and_remove_images(var.var,image_id,args.no_verify)
    else:
        images=args.image.split(',')
        for i in images:
            get_and_remove_images(i,image_id,args.no_verify)