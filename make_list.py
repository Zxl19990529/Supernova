import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode',default = 'test',type=str,help='test or train')
args = parser.parse_args()

root= './extract_a'+'_'+args.mode

for filename in os.listdir(root):
    basename = filename.split('.')[0]
    basename = basename.split('_')[0]
    f = open('record_'+args.mode+'.txt','a')
    f.writelines(basename+'\n')
    f.close()