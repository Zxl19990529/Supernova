from PIL import Image
import os
import shutil
import argparse
import os
import shutil

parse = argparse.ArgumentParser()
parse.add_argument('--folder',type=str,help='the folder of the train/test')
parse.add_argument('--mode',type=str,default='test',help='test or train')
args = parse.parse_args()
path = args.folder

extract_path = './extract'+'_'+args.mode
extract_a = './extract_a'+'_'+args.mode
extract_b = './extract_b'+'_'+args.mode
extract_c = './extract_c'+'_'+args.mode

if not os.path.exists(extract_a):
    os.mkdir(extract_a)
if not os.path.exists(extract_b):
    os.mkdir(extract_b)
if not os.path.exists(extract_c):
    os.mkdir(extract_c)


if not os.path.exists(extract_path):
    os.mkdir(extract_path)
for root,dirs,files in os.walk(path):
    for i in range(len(files)):
        if(files[i][-3:] == 'jpg'):
            file_path = root + '/' + files[i]
            new_file_path = extract_path + '/' + files[i]
            shutil.move(file_path,new_file_path)

for root,dirs,files in os.walk(extract_path):
    for i in range(len(files)):
        if(files[i][-3:] == 'jpg'):
            file_path = root + '/' + files[i]
            new_file_path = extract_path + '/' + files[i]
            shutil.move(file_path,new_file_path)

for filename in os.listdir(extract_path):
    basenaem = filename.split('.')[0]
    extend = basenaem.split('_')[-1]
    img = os.path.join(extract_path,filename)
    if extend == 'a':
        shutil.move(img,extract_a)
    elif extend == 'b':
        shutil.move(img,extract_b)
    elif extend =='c':
        shutil.move(img,extract_c)