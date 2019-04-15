import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import argparse
import os
import numpy as np
import cv2
parser = argparse.ArgumentParser()

parser.add_argument('--config',default='./cascade_rcnn_r101_fpn_1x_future_2class.py',type=str)
parser.add_argument('--img',default='visualization/0a0ac43e05c1187304913cd710bbd494.jpg',type=str)
parser.add_argument('--checkpoint',default='checkpoints/epoch_117.pth',type=str)
parser.add_argument('--img_folder',default='./visualization',type=str)
parser.add_argument('--output',default='./test.csv',type=str)
parser.add_argument('--threhold',default=0.0001,type=float)
args = parser.parse_args()

cfg = mmcv.Config.fromfile(args.config)
cfg.model.pretrained = None

if os.path.exists(args.output):
    os.remove(args.output)

f = open(args.output,'a')
f.writelines('id,x1,y1,x2,y2,x3,y3,havestar\n')
f.close()
# 加载模型
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, args.checkpoint)

# 读入图片

for filename in os.listdir(args.img_folder):
    img = mmcv.imread(os.path.join(args.img_folder,filename))

    # img = mmcv.imread(args.img)
    result = inference_detector(model, img, cfg)
    show_result(img, result)
    
    center = []
    ###---get bbox---###

    bbox_result = result
    bboxes = np.vstack(bbox_result)
    labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
    labels = np.concatenate(labels)
    if args.threhold > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > args.threhold
            bboxes = bboxes[inds, :]
            labels = labels[inds]
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        # left_top = (bbox_int[0], bbox_int[1])
        # right_bottom = (bbox_int[2], bbox_int[3])
        x_min,y_min = bbox_int[0], bbox_int[1]
        x_max,y_max = bbox_int[2], bbox_int[3]
        class_names=['0','1']
        label_text = class_names[
                label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        # print('x_min:{},y_min:{},x_max:{},y_max:{},label:{}'.format(x_min,y_min,x_max,y_max,label))
        x,y=int((x_max+x_min)/2),int((y_max+y_min)/2)
        center.append([x,y])
    print(len(center))# 2
    print(center)# [[483, 182], [482, 180]]
    if len(center)==2:
        center.append(center[0])
    if len(center)==1:
        center.append(center[0])
        center.append(center[0])
    basename = filename.split('.')[0]
    x1,y1=center[0]
    x2,y2=center[1]
    x3,y3=center[2]
    x1,x2,x3,y1,y2,y3 = str(x1),str(x2),str(x3),str(y1),str(y2),str(y3)
    result_write = [basename,x1,y1,x2,y2,x3,y3,str(label)]
    result_write = ','.join(result_write)+'\n'
    f = open(args.output,'a')
    f.writelines(result_write)
    f.close()
    # test a list of images
    # imgs = ['test1.jpg', 'test2.jpg']
    # for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    #     print(i, imgs[i])
    #     show_result(imgs[i], result)