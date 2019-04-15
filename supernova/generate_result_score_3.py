import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import argparse
import os
import numpy as np
import cv2
import pandas as pd

from numpy import *


parser = argparse.ArgumentParser()

parser.add_argument('--config',default='./cascade_rcnn_r101_fpn_1x_future_2class.py',type=str)
#parser.add_argument('--img',default='visualization/0a0ac43e05c1187304913cd710bbd494.jpg',type=str)
parser.add_argument('--checkpoint',default='./checkpoints/epoch_83.pth',type=str)
parser.add_argument('--img_folder',default='../clear_merged_test',type=str)
parser.add_argument('--output',default='./submit.csv',type=str)
parser.add_argument('--threshold',default=0.1,type=float)
args = parser.parse_args()


def test(checkpoint,img_folder,config,output,threshold):
    cfg = mmcv.Config.fromfile(config)
    cfg.model.pretrained = None

    if os.path.exists(output):
        os.remove(output)

    f = open(output, 'a')
    f.writelines('id,x1,y1,x2,y2,x3,y3,havestar\n')
    f.close()

    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, checkpoint)
    # print("Load model successfully")

    count = 0

    for filename in os.listdir(img_folder):
        count = count + 1
        if (count % 50 ==0):
            print("count: ",count)
        # print("filename: ",filename,"count: ",count)
        img = mmcv.imread(os.path.join(img_folder,filename))
        image_input = cv2.imread(os.path.join(img_folder,filename))
        image_shape = image_input.shape

        #print("img: ",img)
        # img = mmcv.imread(args.img)
        result = inference_detector(model, img, cfg)
        #print("result: ",result)
        # show_result(img, result)
        # print("show result")

        center = []
        ###---get bbox---###

        bbox_result = result

        #print("result: ",result)
        bboxes = np.vstack(bbox_result)
        labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
        labels = np.concatenate(labels)
        if threshold > 0:
                assert bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > threshold
                bboxes = bboxes[inds, :]
                labels = labels[inds]
        flag = 0
        for bbox, label in zip(bboxes, labels):
            #print("bbox: ",bbox)
            #print("label: ",label)
            bbox_int = bbox.astype(np.int32)
            # left_top = (bbox_int[0], bbox_int[1])
            # right_bottom = (bbox_int[2], bbox_int[3])
            x_min,y_min = bbox_int[0], bbox_int[1]
            x_max,y_max = bbox_int[2], bbox_int[3]
            class_names=['0','1']
            label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
            if label_text == '1':
                flag = 1
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])
            # print('x_min:{},y_min:{},x_max:{},y_max:{},label:{}'.format(x_min,y_min,x_max,y_max,label))
            x,y=int((x_max+x_min)/2),int((y_max+y_min)/2)
            center.append([x,y])
        #print("len(center): ",len(center))# 2
        #print(center)# [[483, 182], [482, 180]]
        if len(center)==2:
            center.append(center[0])
        if len(center)==1:
            center.append(center[0])
            center.append(center[0])
        if len(center)==0:
            label = 0
            value_range = int(image_shape[0]) if int(image_shape[0])<int(image_shape[1]) else int(image_shape[1])
            x,y= np.random.randint(value_range),np.random.randint(value_range)
            center.append([x,y])
            center.append([x,y])
            center.append([x,y])
        basename = filename.split('.')[0]
        x1,y1=center[0]
        x2,y2=center[1]
        x3,y3=center[2]
        x1,x2,x3,y1,y2,y3 = str(x1),str(x2),str(x3),str(y1),str(y2),str(y3)
        result_write = [basename,x1,y1,x2,y2,x3,y3,str(flag)]
        result_write = ','.join(result_write)+'\n'
        f = open(output,'a')
        f.writelines(result_write)
        f.close()

def calED(vec1,vec2):
    #dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    dist = np.linalg.norm(vec1 - vec2)
    return dist


def F1(list1,list2):
    #print("list1: ",len(list1))
    #print("list2: ",len(list2))
    P, R, TP, FN, FP, TN =0, 0, 0, 0, 0, 0
    N = 0
    t = 0
    for i in range(len(list1)-1):
        if (list1[i][3] == '1'):
            N+=1
        #N+=1
        for j in range(len(list2)):
            if(list1[i][0] == list2[j][0]):
                if (list1[i][3] == '1') and (list2[j][7] == '1'):
                    TP += 1
                    #N+=1
                elif (list1[i][3] == '1') and (list2[j][7] == '0'):
                    FN += 1
                elif (list1[i][3] == '0') and (list2[j][7] == '1'):
                    FP += 1
                    #N+=1
                elif (list1[i][3] == '0') and (list2[j][7] == '0'):
                    TN += 1
                else:
                    print("No this case!")
                if (list1[i][3] == '1'):
                    #N+=1
                    # v0 = np.array(list1[i][1],list1[i][2])
                    x,y  = int(list1[i][1]),int(list1[i][2])
                    v0 = np.array([x,y])
                    #print("v0: ",v0)
                    # v1 = np.array(list2[i][1:3])
                    x1, y1 = int(list2[j][1]), int(list2[j][2])
                    v1 = np.array([x1, y1])
                    x2, y2 = int(list2[j][3]), int(list2[j][4])
                    v2 = np.array([x2, y2])
                    x3, y3 = int(list2[j][5]), int(list2[j][6])
                    v3 = np.array([x3, y3])
                    if (calED(v1, v0) < 15 or calED(v2, v0) < 15 or calED(v3, v0) < 15):
                        #print(calED(v1,v0))
                        t = t + 1
                        #print('t',t)
            else:
                continue
    P = TP/(TP+FP)
    R = TP/(TP+FN)

    F1 = 2 * P * R / (P + R)

    print("F1: ",F1,"N: ",N,"t: ",t)
    print("P: ",P,"R: ",R)
    #print("num:",num)

    return F1,N,t

def compute_score(F1, total_N, total_t):
    S1 =  total_t + F1
    S = S1/(total_N + 1)
    return S

def read_csv(file_name):
    f = open(file_name, 'r')
    content = f.read()
    final_list = list()
    rows = content.split('\n')
    for row in rows:
        final_list.append(row.split(','))
    return final_list

if __name__ == '__main__':
    test(args.checkpoint,args.img_folder,args.config,args.output,args.threshold)
    #csv_name_1 = './test_1259.csv'   # test_csv
    #csv_name_2 = args.output    # generate csv
    #csv_name_2 = '/media/disk/liuhongzhi/test_100.csv'
    #f1 = read_csv(csv_name_1)
    # f2 = read_csv(csv_name_2)
    #print(len(f1))
    #print(len(f2))
    #compute_F1,total_num,t_num = F1(f1,f2)
    #score = compute_score(compute_F1,total_num,t_num)
    #print("score: ",score)

    # f = open('/media/disk/liuhongzhi/VOCdevkit2007_novall/work_dirs_50epoch_2bs/log_map.txt', 'a')
    # f.write(str(args.checkpoint) + ' ' + str(args.threshold) + ' ' + str(compute_F1)+' '+ str(t_num)+' '+str(score) + '\n')
    # f.close()
    
# CUDA_VISIBLE_DEVICES=1 python ./tools/generate_result.py --config ./configs/cascade_rcnn_r101_fpn_1x_future_2class.py --checkpoint /media/disk/liuhongzhi/VOCdevkit2007_novall/work_dirs_50epoch_2bs/test_checkpoint/epoch_${num}.pth --img_folder /media/disk/liuhongzhi/VOCdevkit2007_novall/test_1047 --output /media/disk/liuhongzhi/test_csv/test_${num}_${threshold}.csv --threhold 0.01