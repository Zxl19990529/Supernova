# -*- coding: UTF-8 -*-
import os
from utilscsv import *
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import cv2

countnum = 0
save_todir = './data/VOCdevkit/VOC2007/Annotations/'

def save_xml(image_name, bbox_class, save_dir=save_todir, width=1609, height=500, channel=3):

  global countnum

  path = './data/VOCdevkit/VOC2007/JPEGImages/'+ image_name + '.jpg'

  img = cv2.imread(path)  # read image.jpg from dirfile
  size = img.shape
  width = size[1]
  height = size[0]
  channel = size[2]

  node_root = Element('annotation')
  node_folder = SubElement(node_root, 'folder')
  node_folder.text = 'JPEGImages'

  node_filename = SubElement(node_root, 'filename')
  node_filename.text = image_name + '.jpg'

  node_size = SubElement(node_root, 'size')
  node_width = SubElement(node_size, 'width')
  node_width.text = '%s' % width
  node_height = SubElement(node_size, 'height')
  node_height.text = '%s' % height
  node_depth = SubElement(node_size, 'depth')
  node_depth.text = '%s' % channel


  # transfer 8 class into 2 class in order to train
  if (bbox_class[2]=='newtarget') or (bbox_class[2]=='isstar') or (bbox_class[2]=='asteroid') or (bbox_class[2]=='isnova') or (bbox_class[2]=='known'):
      bbox_class[2] = '1'
  else:
      bbox_class[2] = '0'

  print("bbox_class: ",bbox_class)

  # for i in range(len(bbox_class)):
  if int(bbox_class[0]) <6 or abs(int(bbox_class[0])-width)<6:  # x coordiante near boundary

        if int(bbox_class[1]) <6 or abs(int(bbox_class[1])-height)<6:  # y coordiante near boundary
           print("x near bbox_class[1]: ",int(bbox_class[1]))
           # left is minimum
           if int(bbox_class[0]) ==1:
               left = int(bbox_class[0])
           else:
               left = int(bbox_class[0])-1

           top = int(bbox_class[1])-1

           # right is maxmium
           if int(bbox_class[0]) == width:
              right = int(bbox_class[0])
           else:
              right = int(bbox_class[0])+1

           bottom = int(bbox_class[1]) + 1

        else:                                                     # y coordiante away from boundary
            print("x near y away bbox_class[0]: ", int(bbox_class[0]))
            left = int(bbox_class[0]) - 1
            top = int(bbox_class[1]) - 1
            right = int(bbox_class[0]) + 1
            bottom = int(bbox_class[1]) + 1

  elif int(bbox_class[1]) <6 or abs(int(bbox_class[1])-height)<6:  # y coordiante near boundary
           print("y near bbox_class[1]: ",int(bbox_class[1]))
           left = int(bbox_class[0])-1
           top = int(bbox_class[1])-1
           right = int(bbox_class[0])+1
           bottom = int(bbox_class[1]) + 1

  else:
         left, top, right, bottom = int(bbox_class[0])-5, int(bbox_class[1])-5, int(bbox_class[0]) + 5, int(bbox_class[1]) + 5

  if (left >=1 and left <= width) and (top >=1 and top <= height) and (right >=1 and right <= width) and (bottom >=1 and bottom <= height):
        countnum += 1
        print("lefttop and rightbottom are in the range!", countnum)
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = '%s' % bbox_class[2]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom

  else:
        # print("There is an error: ",node_filename.text)
        # file_object = open('log.txt', 'a+')
        # file_object.writelines("There is an error: "+ node_filename.text + '\t')
        # file_object.writelines(str(left)+' '+str(top)+' '+str(right)+' '+ str(bottom)+'\n') 
        # file_object.close()
        if left <1 :
            left =0
        if right >=width:
            right = width
        if bottom < 1:
            bottom = 0
        if top >= height:
            top = height
        print("lefttop and rightbottom are in the range!", countnum)
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = '%s' % bbox_class[2]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom


  xml = tostring(node_root, pretty_print=True)
  dom = parseString(xml)

  save_xml = os.path.join(save_dir, node_filename.text.replace('jpg', 'xml'))
  with open(save_xml, 'wb') as f:
        f.write(xml)

  return



def change2xml(label_dict={}):
    for image in label_dict.keys():
        image_name = os.path.split(image)[-1]
        bbox_object = label_dict.get(image, [])
        save_xml(image_name, bbox_object)
    return


if __name__ == '__main__':
    # step 2
    # make_voc_dir()

    # step 3
    # label_dict = utils.read_csv(csv_path=r'./train_b.csv',
    #                             pre_dir=r'/home/matthew/dataset')
    # rename_image(label_dict)

    # step 3


    label_dict = read_csv(csv_path=r'./list.csv',
                                pre_dir=r'./JPEGImages')
    change2xml(label_dict)
