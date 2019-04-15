import numpy as np 
from PIL import Image

def merge_img(img_a,img_b,img_c):
    img_a = Image.open(img_a)
    img_b = Image.open(img_b)
    img_c  =Image.open(img_c)
    img_a = np.array(img_a)
    img_b = np.array(img_b)
    img_c = np.array(img_c)
    new_img = np.zeros((img_a.shape[0],img_a.shape[1],3))
    new_img[:,:,0] = img_a
    new_img[:,:,1] = img_b
    new_img[:,:,2] = img_c
    new_img = Image.fromarray(np.uint8(new_img))
    return new_img