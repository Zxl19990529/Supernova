# -*- coding: utf-8 -*-
# @Author  : OUC
# @File    : utils.py
# @Software: PyCharm

import csv
import os

def read_csv(csv_path, pre_dir):

    label_dict = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = True
        for line in reader:
            
            if header:
                header = False
                continue
            
            image_path = os.path.join(pre_dir, line[0])
            
            bbox_object = []
        
            for i in range(1,4):
              bbox_object.append(line[i])

            
            label_dict.setdefault(image_path, bbox_object)
    return label_dict


def write_csv(result_dict, out_path='out.csv'):
    
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
       
        writer.writerow(['name', 'coordinate'])

        for image in result_dict.keys():
            image_name = os.path.split(image)[-1]
            bbox = result_dict.get(image, [])
            bbox_rs = ';'.join(['_'.join(str(int(id)) for id in i) for i in bbox])
            writer.writerow([image_name, bbox_rs])


if __name__ == '__main__':
    label_dict = read_csv(csv_path=r'./train_b.csv',
                             pre_dir=r'/home/matthew/dataset')
    write_csv(label_dict)
