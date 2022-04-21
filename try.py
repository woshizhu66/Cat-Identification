import pandas as pd
import os
from PIL import ImageFile
from sklearn.model_selection import train_test_split
import random
import shutil
from shutil import copy2
from sklearn.datasets import load_files
from tensorflow.keras import utils
import numpy as np
from glob import glob
from tensorflow.keras.applications.resnet50 import ResNet50

# 定义ResNet50模型
ResNet50_model = ResNet50(weights='imagenet')

filelist = []

for dirname, d, filenames in os.walk('D:/System/archive (1)/dataset/images'):
    for filename in filenames:
        filelist.append(os.path.join(dirname, filename))

breeds = pd.read_csv('D:/System/archive (1)/dataset/data/cats.csv')
breeds['breed'].unique()

count = 0
validDir = []
for dirname, _, filenames in os.walk('D:/System/archive (1)/dataset/images'):
    for dire in _:
        for dirname2, _2, filenames2 in os.walk(os.path.join(dirname, dire)):
            for file in filenames2:
                count = count + 1
            if (count > 150):
                validDir.append(dire)
            count = 0
len(validDir)


# 定义函数来加载train，test和validation数据集
def load_dataset(path):
    data = load_files(path)
    cat_files = np.array(data['filenames'])
    cat_hot = utils.to_categorical(np.array(data['target']), 40)
    return cat_files, cat_hot


# 加载train，test和validation数据集
train_files, train_hot = load_dataset('D:/project/Dataset/train')
valid_files, valid_hot = load_dataset('D:/project/Dataset/validation')
test_files, test_hot = load_dataset('D:/project/Dataset/test')

# 加载狗品种列表
cat_names = [item[20:-1] for item in sorted(glob("D:/project/Dataset/train/*/"))]

# 打印数据统计描述
print(cat_names)
print(train_hot)
print('There are %d total cat categories.' % len(cat_names))
print('There are %s total cat images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training cat images.' % len(train_files))
print('There are %d validation cat images.' % len(valid_files))
print('There are %d test cat images.' % len(test_files))

# 图像预处理
from tensorflow.keras.preprocessing import image
from tqdm import tqdm


def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


ImageFile.LOAD_TRUNCATED_IMAGES = True

# Keras中的数据预处理过程
train_tensors = paths_to_tensor(train_files).astype('float32') / 255
valid_tensors = paths_to_tensor(valid_files).astype('float32') / 255
test_tensors = paths_to_tensor(test_files).astype('float32') / 255
