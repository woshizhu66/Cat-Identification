import os
import random


train_val_percent = 0.9
train_percent = 0.9
xml_filepath = 'D:/System/yolov5-6.0/yolov5-6.0/data/Annotations'
txt_save_path = 'D:/System/yolov5-6.0/yolov5-6.0/data/ImageSets'
total_xml = os.listdir(xml_filepath)

num = len(total_xml)
nlist = range(num)
tv = int(num * train_val_percent)
tr = int(tv * train_percent)
trainval = random.sample(nlist, tv)
train = random.sample(trainval, tr)

p_train_val = open('D:/System/yolov5-6.0/yolov5-6.0/data/ImageSets/trainval.txt', 'w')
p_test = open('D:/System/yolov5-6.0/yolov5-6.0/data/ImageSets/test.txt', 'w')
p_train = open('D:/System/yolov5-6.0/yolov5-6.0/data/ImageSets/train.txt', 'w')
p_val = open('D:/System/yolov5-6.0/yolov5-6.0/data/ImageSets/val.txt', 'w')

for i in nlist:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        p_train_val.write(name)
        if i in train:
            p_train.write(name)
        else:
            p_val.write(name)
    else:
        p_test.write(name)

p_train_val.close()
p_train.close()
p_val.close()
p_test.close()
