import cv2
import xml.etree.ElementTree as ET
import os
import sys
import lxml
import shutil
import argparse,sys

# user input files path
min_size = 800


def search_jpg_xml(image_dir, label_dir):
    # find out all of sepecified file
    image_ext = '.jpg'
    img = [fn for fn in os.listdir(image_dir) if fn.endswith(image_ext)]
    label_ext = '.xml'
    label = [fn for fn in os.listdir(label_dir) if fn.endswith(label_ext)]
    return img, label


def copyfile():
    # if "Annotations_temp" in os.listdir(path):
    #     shutil.rmtree(path + "/Annotations_temp")
    # if "JPEGImages_temp" in os.listdir(path):
    #     shutil.rmtree(path + "/JPEGImages_temp")
    save_annotation_path = path + "/Annotations_temp/"
    save_jpg_path = path + "/JPEGImages_temp/"
    shutil.copytree(path + "/TMP_Ann", save_annotation_path)
    shutil.copytree(path + "/TMP_JPEG_1", save_jpg_path)
    return save_jpg_path, save_annotation_path

def write_xml_jpg(jpg_path, annotation_path,model):
    img, label = search_jpg_xml(jpg_path, annotation_path)
    # sorted(img)
    # sorted(label)
    # print(img)
    print(len(label))
    if "Annotations_2" not in os.listdir(path):
        os.mkdir(path + "/Annotations_2")
    # if "JPEGImages_1" not in os.listdir(path):
    #     os.mkdir(path + "/JPEGImages_1")
    # new_image_path = path + "/JPEGImages_1/"
    new_annotation_path = path + "/Annotations_2/"
    cou = 0
    for index, file in enumerate(label):
        # print(index,file)
        cur_img = cv2.imread(jpg_path + img[index])
        width = cur_img.shape[1]
        height = cur_img.shape[0]

        cur_xml = ET.parse(annotation_path + file)
        root = cur_xml.getroot()
        for node in root:
            if node.tag == 'filename':
                # print(node.text)
                node.text = node.text[:-4]+'_sxm.jpg'
                if node.text == '6913f923a11c10cfe14ef8695851a731_sxm.jpg':
                    print('6913f923a11c10cfe14ef8695851a731_sxm')
            elif node.tag == 'size':
                new_width  = int(node[0].text)
                new_height = int(node[1].text)
            elif node.tag == 'object':
                xmin = int(node[2][0].text)  # bbox position
                ymin = int(node[2][1].text)
                xmax = int(node[2][2].text)
                ymax = int(node[2][3].text)

                if model:
                    node[2][0].text = str(int(new_width - xmax))
                    node[2][1].text = str(int(ymin))
                    node[2][2].text = str(int(new_width - xmin))
                    node[2][3].text = str(int(ymax))
                    file_name = file[:-4] + '_zym.xml'
                else:
                    # print('flag ')
                    node[2][0].text = str(int(xmin))
                    node[2][1].text = str(int(new_height - ymax))
                    node[2][2].text = str(int(xmax))
                    node[2][3].text = str(int(new_height - ymin))
                    file_name = file[:-4]+'_sxm.xml'
        cur_xml.write(new_annotation_path + file_name)
        print(index,file_name)
        cou += 1
    print('count: ',cou)
    # shutil.rmtree(path + "JPEGImages_2")
    # shutil.rmtree(path + "Annotations_2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument('--path', dest='path', default='./data/VOCdevkit/VOC2007', type=str)
    parser.add_argument('--path', dest='path', default='/home/zhb/Desktop/Supernova-master/supernova/data/VOCdevkit/VOC2007', type=str)
    parser.add_argument('--image_path', dest='image_path', default='./TMP_JPEG_1', type=str)
    parser.add_argument('--xml_path', dest='xml_path', default='./TMP_Ann', type=str)
    parser.add_argument('--model', dest='model', default=True, type=bool)  # True for horrizon  False for vertical

    '''
    path = "./test_folder"
    image_path = path + "/h_mirror_img/"  # image path with .jpg ending
    label_path = path + "/copy_star_xml/"  # label path with .xml ending
    '''

    # print(parser.parse_args())
    args = parser.parse_args()

    path = args.path
    image_path = args.image_path
    label_path = args.xml_path
    model = args.model
    jpg_path, annotation_path = copyfile()
    write_xml_jpg(jpg_path, annotation_path,model)
    write_xml_jpg(jpg_path, annotation_path,False)
    # print(len(jpg_path))
    # print(len(annotation_path))