import os
from PIL import Image,ImageDraw2

img_folder = '../clear_merged_test'
save_dir = img_folder+'_vis'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
flag=1
for line in open('./submit.csv'):
    if flag:
        flag=0
        continue
    id,x1,y1,x2,y2,x3,y3,havestar=line.strip().split(',')
    x1,y1,x2,y2,x3,y3 = int(x1),int(y1),int(x2),int(y2),int(x3),int(y3)
    # print(id,x1,y1,x2,y2,x3,y3,havestar) # 7f0cf4a6262372bb93077e1611ddfd0b 149 124 134 84 150 195 0
    img = os.path.join(img_folder,id+'.jpg')
    img = Image.open(img)
    draw = ImageDraw2.Draw(img)
    pen = ImageDraw2.Pen('white')
    font = ImageDraw2.Font(color='white',file='./micross.ttf')
    x1_min=x1-10
    x1_max=x1+10
    y1_min=y1-10
    y1_max=y1+10
    x2_min=x2-10 
    x2_max=x2+10
    y2_min=y2-10
    y2_max=y2+10
    x3_min=x3-10
    x3_max=x3+10
    y3_min=y3-10
    y3_max=y3+10
    draw.line([(x1_min,y1_min),(x1_max,y1_min),(x1_max,y1_max),(x1_min,y1_max),(x1_min,y1_min)],pen)
    draw.text((x1_max,y1_max),str(havestar),font=font)
    draw.line([(x2_min,y2_min),(x2_max,y2_min),(x2_max,y2_max),(x2_min,y2_max),(x2_min,y2_min)],pen)
    draw.text((x2_max,y2_max),str(havestar),font=font)
    draw.line([(x3_min,y3_min),(x3_max,y3_min),(x3_max,y3_max),(x3_min,y3_max),(x3_min,y3_min)],pen)
    draw.text((x3_max,y3_max),str(havestar),font=font)
    img.save(os.path.join(save_dir,id+'.jpg'))
    print(os.path.join(save_dir,id+'.jpg'))