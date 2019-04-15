import numpy as np 
from PIL import Image
import os
import cv2
from scipy import signal
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--original_folder',default='',type=str)
parser.add_argument('--mode',default='test',type=str)
parser.add_argument('--save_folder',default='clear_a')
args = parser.parse_args()

count = 0
root = args.original_folder+'_'+args.mode
save_dir = args.save_folder+'_'+args.mode

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for filename in os.listdir(root):
    img = os.path.join(root,filename)
    img = Image.open(img).convert('L')
    img = np.array(img)
    threhold = 155
    # print(threhold)
    img = np.where(img<=threhold,0,img)
    img = signal.medfilt(img,3)
    img = Image.fromarray(np.uint8(img)).convert('L')
    img.save(os.path.join(save_dir,filename))
    count +=1
    print(filename,count)