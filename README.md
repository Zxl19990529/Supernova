说明： 我们在testB 上的成绩是0.5297, 这是基于 https://github.com/open-mmlab/mmdetection 的一个改版。

# 2019 未来杯高校AI挑战赛 区域赛作品

比赛网址： https://ai.futurelab.tv/

* 战队编号：274
* 战队名称: 超新星和月亮
* 战队成员：
  - Zxl19990529
  - 浪迹天涯
  - OUC_LiuX	
  - zhanghaoxu	
  - HenryLiu	

## 概述

采用目标检测，cascade + fpn。 

## 系统要求

### 硬件环境要求

* CPU:  i7-8700u
* GPU:  GTX-1070
* 内存:  16G
* 硬盘:  1T
* 其他:  -

### 软件环境要求

* 操作系统: Ubuntu 16.04 
* CUDA 9.0
* Pytorch 1.0+
* python3.6+
* anaconda3
* cython

首先搭建环境: `conda env create -f supernova.yml`  
激活环境: `conda activate supernova`  
`pip install cython`  
`pip install torch torchvision`
编译文件 ：  
`cd supernova`  
`sh ./compile.sh`   
`pip install .`

需要预训练模型 resnext101 https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d-a5af3160.pth  
下载好后需要放到 ~/.torch/models文件夹中
### 数据集

训练集需要按照数据预处理中的步骤进行处理，转化成VOC格式的数据集。  

## 数据预处理

### 方法概述
对于测试集，先把所有图片都提取到一个文件夹，然后进行降噪处理，最后把差值图、新图和历史图融合为RGB彩图。 对于训练集，需要转化成VOC格式进行训练。

### 操作步骤

#### 测试集预处理
确保当前路径和此 **README.md** 路径在同一级。
首先把文件夹中的文件都提取到一个文件夹。  
`python extract.py --folder <测试集/训练集解压后的文件夹路径> --mode <test/train>`  
对图片进行降噪。  
`python clean.py --original_folder extract_a --save_folder clear_a --mode <test/train>`  
`python clean.py --original_folder extract_b --save_folder clear_b --mode <test/train>`  
`python clean.py --original_folder extract_c --save_folder clear_c --mode <test/train>`  
以上程序运行完毕后，再运行以下脚本产生没有扩展名的文件列表。  
`pythom make_list.py --mode <test/train>`  
产生降噪后的彩图。  
`python merge.py --output_dir ./clear_merged --mode <test/train>`

预处理后的图像保存在名为` clear_merged_test`的文件夹中。

#### 训练集预处理  
首先把图像都提取到当前目录，确保和当前 README.md 路径在同一级。  
`python extract.py --folder <测试集/训练集解压后的文件夹路径> --mode <test/train>`  
对提取后的图像进行降噪处理。  
`python clean.py --original_folder extract_a --save_folder clear_a --mode <test/train>`  
`python clean.py --original_folder extract_b --save_folder clear_b --mode <test/train>`  
`python clean.py --original_folder extract_c --save_folder clear_c --mode <test/train>`  
以上程序运行完毕后，再运行以下脚本产生没有扩展名的文件列表。  
`pythom make_list.py --mode <test/train>`  
产生降噪后的彩图。  

`python merge.py --output_dir ./clear_merged --mode <test/train>`  

进入supernova路径`cd supernova`    

1. 把降噪后的图片整理好。`cp ../clear_merged_train/* ./data/VOCdevkit/VOC2007/JPEGImages/`
1. 生成xml文件 `python vocxml_make.py`  
2. 生成train.txt 文件`python make_train_val_test_set.py`  

### 模型

模型文件大小：639MB

## 训练

### 训练方法概述

用cascade_rcnn resnext101 结合fpn 进行训练。

### 训练操作步骤

支持多块GPU的训练,1 对应的是显卡数目，CUDA_VISIBLE_DEVICES= 对应显卡编号。  
```shell
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh ./cascade_rcnn_x101_32x4d_fpn_1x_2class.py 1 --validate 
```

此外支持的训练参数有:

- --validate: 每k个epoch显示验证集结果
- --work_dir <WORK_DIR>: 如果指定该参数，则会在`cascade_rcnn_x101_32x4d_fpn_1x_2class`中设置工作区。

在 WORK_DIR中会生成以下文件:

- log 日志文件
- checkpoint文件
- 最新checkpoint的链接

### 训练结果保存与获取

保存在checkpoints（如果指定WORK_DIR,去WORD_DIR路径下查看checkpoint）文件夹里。本次测试用的是epoch_19.pth

## 测试

### 方法概述

首先对测试集进行数据预处理。然后执行测试程序。

### 操作步骤

进入`supernova`文件夹：  
`cd supernova` 
激活环境 `conda activate supernova`  
执行测试程序 
`python generate_result_2.py --test-dataset ../clear_merged_test/ --prediction-file ./submit.csv`


## 其他

这个仓库里没有权重，权重在 ：  
链接:https://pan.baidu.com/s/1UNVkxzxgFDrq6IF1p4e4xA 提取码:l18w 复制这段内容后打开百度网盘手机App，操作更方便哦