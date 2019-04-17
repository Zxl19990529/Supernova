import cv2
import copy
import os

"""
#水平镜像可按公式
#I = i
#J = N - j + 1
#垂直镜像可按公式
#I = M - i + 1
#J = j
#对角镜像可按公式
#I = M - i + 1
#J = N - j + 1
"""

import argparse,sys

parser = argparse.ArgumentParser()

parser.add_argument('--imgs_path',dest='imgs_path',default = './data/VOCdevkit/VOC2007/Images',type = str)
parser.add_argument('--save_path',dest='save_path',default = './data/VOCdevkit/VOC2007/JPEGImages',type = str)

print (parser.parse_args())
args = parser.parse_args()

imgs_path = args.imgs_path
save_path = args.save_path


def mirror_imgs(imgs_path, save_path):
  for name in os.listdir(imgs_path):
    print(name)
    image = cv2.imread(os.path.join(imgs_path, name), 1);
    height = image.shape[0]
    width = image.shape[1]
    # channels = image.shape[2]
    iLR = copy.deepcopy(image)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制

    for i in range(height):
      for j in range(width):
        iLR[i, width - 1 - j] = image[i, j]
    # cv2.imshow('image', image)
    # cv2.imshow('iLR', iLR)
    save_name = name[:-4]+'_zym'+'.jpg'

    cv2.imwrite(os.path.join(save_path, save_name), iLR,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 保存图片
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def horizontal_mirror_imgs(imgs_path, save_path):
  for name in os.listdir(imgs_path):
    print(name)
    image = cv2.imread(os.path.join(imgs_path, name), 1);
    height = image.shape[0]
    width = image.shape[1]
    # channels = image.shape[2]
    iLR = copy.deepcopy(image)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制

    for i in range(height):
      for j in range(width):
        iLR[i, width - 1 - j] = image[i, j]
    # cv2.imshow('image', image)
    # cv2.imshow('iLR', iLR)
    save_name = name[:-4]+'_zym'+'.jpg'

    cv2.imwrite(os.path.join(save_path, save_name), iLR,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 保存图片
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def vertical_mirror_imgs(imgs_path, save_path):
  for name in os.listdir(imgs_path):
    print(name)
    image = cv2.imread(os.path.join(imgs_path, name), 1);
    height = image.shape[0]
    width = image.shape[1]
    # channels = image.shape[2]
    iLR = copy.deepcopy(image)  # 获得一个和原始图像相同的图像，注意这里要使用深度复制

    for i in range(height):
      for j in range(width):
        iLR[height - 1 - i, j] = image[i, j]
    # cv2.imshow('image', image)
    # cv2.imshow('iLR', iLR)
    save_name = name[:-4]+'_sxm'+'.jpg'

    cv2.imwrite(os.path.join(save_path, save_name), iLR,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 保存图片
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
'''
imgs_path = '/home/henry/Files/FutureAi/VOC2007_6280/all_star/copy_star_image'
save_path = "/home/henry/Files/FutureAi/VOC2007_6280/all_star/copy_star_image_mirror"
'''

if not os.path.exists(save_path):
    os.makedirs(save_path)
#mirror_imgs(imgs_path, save_path)

horizontal_mirror_imgs(imgs_path,save_path)
vertical_mirror_imgs(imgs_path, save_path)


