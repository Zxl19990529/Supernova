import numpy as np
import os
from PIL import Image
from utils import merge_img
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir',default='./clear_merged',type=str)
parser.add_argument('--mode',default='test',type=str)
args = parser.parse_args()
count =0
output_dir = args.output_dir+'_'+args.mode
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
for line in open('record_'+args.mode+'.txt'):
    # print(filename)
    img_a = line.strip() + '_a.jpg'
    img_b = line.strip() + '_b.jpg'
    img_c = line.strip() + '_c.jpg'
    img_a='./clear_a'+'_'+args.mode+'/'+img_a # clear_a_test
    img_b='./clear_b'+'_'+args.mode+'/'+img_b
    img_c='./clear_c'+'_'+args.mode+'/'+img_c
    new_img = merge_img(img_a, img_b, img_c)
    save_path = os.path.join(output_dir,line.strip()+'.jpg')
    new_img.save(save_path)
    count+=1
    print(save_path,' saved',count)
