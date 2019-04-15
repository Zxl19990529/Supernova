import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train-dataset','--train_dataset',type=str,help='the path to train dataset')
parser.add_argument('--extend',type)
args = parser.parse_args()

# src_dir_path = "./af2019-cv-training-20190312/"
src_dir_path = args.train_dataset

to_dir_path = "./data/VOCdevkit/VOC2007/JPEGImages/"

key = '_a'

if not os.path.exists(to_dir_path):
	print("to_dir_path not exist,so create the dir")
	os.mkdir(to_dir_path)

if os.path.exists(src_dir_path):
	print("src_dir_path exitst")

for parent, dirnames, filename in os.walk(src_dir_path):
	for i in range(len(filename)):
		if key in filename[i]:
			print("find "+key+" in "+ filename[i])
			print("copy to "+to_dir_path+filename[i])
			print(parent+filename[i])
			print(to_dir_path+filename[i])
			newname = filename[i][:-6] + '.jpg'  # move _a and add .jpg
			print(newname)
			shutil.copy(parent+'/'+filename[i],to_dir_path+newname)
