# -*- coding: utf-8 -*-
# xml解析包
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

dsets = ['train', 'test', 'val']
classes = ['Abyssinian', 'American Bobtail', 'American Curl', 'Applehead Siamese', 'Balinese', 'Birman',
           'British Shorthair', 'Norwegian Forest Cat', 'Turkish Angora', 'Turkish Van']


def transfer(size1, abox):  # size:(original w,original h) , box:(xmin,xmax,ymin,ymax)
    ow = 1. / size1[0]  # 1/w
    oh = 1. / size1[1]  # 1/h
    x1 = (abox[0] + abox[1]) / 2.0  # The center point x-coordinate of the object in the figure
    y1 = (abox[2] + abox[3]) / 2.0  # The center point y-coordinate of the object in the figure
    w1 = abox[1] - abox[0]  # The actual pixel width of the object
    h1 = abox[3] - abox[2]  # The actual pixel height of the object
    x = x1 * ow  # The coordinate ratio of the object's center point x
    w = w1 * ow  # The width ratio of the object's width
    y = y1 * oh  # The ratio of the coordinates of the object's center point, y
    h = h1 * oh  # The width ratio of the object's width
    return x, y, w, h


def transfer_annotation(image_id):
    # Correspondingly, the corresponding folder is found through the year, and the xml file of the corresponding
    # image_id is opened, which corresponds to the bundle file
    infile = open('D:/System/yolov5-6.0/yolov5-6.0/data/Annotations/%s.xml' % (image_id), encoding='utf-8')
    # Prepare to write the corresponding label in the corresponding image_id, respectively <object-class> <x> <y>
    # <width> <height>
    outfile = open('D:/System/yolov5-6.0/yolov5-6.0/data/labels/%s.txt' % (image_id), 'w', encoding='utf-8')
    # Parse the xml file
    tree = ET.parse(infile)
    # Obtain the corresponding key-value pairs
    root = tree.getroot()
    # Get the size of the picture
    size = root.find('size')
    # If the tag within xml is empty, the judgment condition is added
    if size != None:
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            # Skip if the category does not correspond to our predetermined class file, or if difficult==1
            if cls not in classes or int(difficult) == 1:
                continue
            # Find the id by category name
            cls_id = classes.index(cls)
            # Locate the bndbox object
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            print(image_id, cls, b)

            bb = transfer((w, h), b)
            if bb[0] < 0 or bb[0] > 1:
                print(image_id)
            elif bb[1] < 0 or bb[1] > 1:
                print(image_id)
            elif bb[2] < 0 or bb[2] > 1:
                print(image_id)
            elif bb[3] < 0 or bb[3] > 1:
                print(image_id)

            outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
print(wd)

for im_set in dsets:

    # First find the labels folder to create if it does not exist
    if not os.path.exists('D:/System/yolov5-6.0/yolov5-6.0/data/labels/'):
        os.makedirs('D:/System/yolov5-6.0/yolov5-6.0/data/labels/')
    ids = open('D:/System/yolov5-6.0/yolov5-6.0/data/ImageSets/%s.txt' % (im_set)).read().split()
    list_file = open('D:/System/yolov5-6.0/yolov5-6.0/data/%s.txt' % im_set, 'w')
    # Write and wrap the corresponding file _id and full path
    for iid in ids:
        list_file.write('data/images/%s.jpg\n' % (iid))
        transfer_annotation(iid)

    list_file.close()
