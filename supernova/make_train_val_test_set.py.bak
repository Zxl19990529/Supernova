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


def _main():
    trainval_percent = 1
    train_percent = 1
    xmlfilepath = './data/VOCdevkit/VOC2007/Annotations'
    total_xml = os.listdir(xmlfilepath)

    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)   # 0.5 of num
    train = random.sample(trainval, tr)  # 0.5 of trainval

    ftrainval = open('./data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w')
    ftest = open('./data/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w')
    ftrain = open('./data/VOCdevkit/VOC2007/ImageSets/Main/train.txt', 'w')
    fval = open('./data/VOCdevkit/VOC2007/ImageSets/Main/val.txt', 'w')

    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


if __name__ == '__main__':
    _main()
