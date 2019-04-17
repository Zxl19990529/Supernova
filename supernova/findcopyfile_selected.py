import os
import shutil
import argparse,sys

parser = argparse.ArgumentParser()

parser.add_argument('--srcimage_dir_path',dest='srcimage_dir_path',default = "./data/train",type = str)
parser.add_argument('--srcxml_dir_path',dest='srcxml_dir_path',default = "./data/xml",type = str)
parser.add_argument('--imageto_dir_path',dest='imageto_dir_path',default = "./data/VOCdevkit/VOC2007/JPEGImages",type = str)
parser.add_argument('--xmlto_dir_path',dest='xmlto_dir_path',default = "./data/VOCdevkit/VOC2007/Annotations",type = str)
parser.add_argument('--txt_path',dest='txt_path',default = './list.csv',type = str)

print (parser.parse_args())
args = parser.parse_args()

'''
srcimage_dir_path = "./VOC2007_6280/VOC_nova_all/JPEGImages"
srcxml_dir_path = "./VOC2007_6280/VOC_nova_all/Annotations_2class_0401"

imageto_dir_path = "./VOC2007_6280/all_star/copy_star_image/"
xmlto_dir_path = "./VOC2007_6280/all_star/copy_star_xml/"

txt_path = './csv_reader.txt'
'''

key = '_a'

count = 0

srcimage_dir_path = args.srcimage_dir_path
srcxml_dir_path = args.srcxml_dir_path

imageto_dir_path = args.imageto_dir_path
xmlto_dir_path = args.xmlto_dir_path

txt_path = args.txt_path

if not os.path.exists(imageto_dir_path):
	print("to_dir_path not exist,so create the dir")
	os.mkdir(imageto_dir_path)

if not os.path.exists(xmlto_dir_path):
	print("to_dir_path not exist,so create the dir")
	os.mkdir(xmlto_dir_path)


# if os.path.exists(src_dir_path):
#	 print("src_dir_path exitst")

fr = open(txt_path)
stringClass = [line.strip().split(',') for line in fr.readlines()]
# print("stringClass: ",stringClass)

for i in range(len(stringClass)):
	if stringClass[i][3] == 'newtarget' or stringClass[i][3] == 'isstar' or stringClass[i][3] == 'asteroid' or stringClass[i][3] == 'isnova' or stringClass[i][3] == 'known':
		image_name = stringClass[i][0] + '.jpg'
		xml_name = stringClass[i][0] + '.xml'
		count +=1
		print(image_name,' ',count)
		shutil.copy(srcimage_dir_path+'/'+image_name,imageto_dir_path+'/'+image_name)
		shutil.copy(srcxml_dir_path + '/' + xml_name, xmlto_dir_path +'/'+ xml_name)


