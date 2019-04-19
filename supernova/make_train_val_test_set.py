# -*- coding: utf-8 -*-
# @Author  : matthew
# @File    : make_train_val_test_set.py
# @Software: PyCharm

# trainval 0.5 of all
# test 0.5 of all
# train 0.5 of trainval
# val 0.5 of trainval

import os
import random

xmlfilepath = './data/VOCdevkit/VOC2007/Annotations'
count = 0
for filename in os.listdir(xmlfilepath):
    count +=1
    filename = filename.split('.')[0]
    f = open('./data/VOCdevkit/VOC2007/ImageSets/Main/train.txt','a')
    f.writelines(filename+'\n')
    f.close()
    if count%200 == 0:

        f_val = open('./data/VOCdevkit/VOC2007/ImageSets/Main/val.txt','a')
        f_val.writelines(filename+'\n')
        f_val.close()
    
    
