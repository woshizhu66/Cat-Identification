import math

import numpy as np

import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    set_logging
from utils.torch_utils import select_device

import os
from xml.etree import ElementTree as ET


# 定义一个创建一级分支object的函数
def create_object(root, xi, yi, xa, ya, obj_name):  # 参数依次，树根，xmin，ymin，xmax，ymax
    # 创建一级分支object
    _object = ET.SubElement(root, 'object')
    # 创建二级分支
    name = ET.SubElement(_object, 'name')
    print(obj_name)
    name.text = str(obj_name)
    pose = ET.SubElement(_object, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(_object, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(_object, 'difficult')
    difficult.text = '0'
    # 创建bndbox
    bndbox = ET.SubElement(_object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = '%s' % xi
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '%s' % yi
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = '%s' % xa
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = '%s' % ya


# 创建xml文件的函数
def create_tree(imgdir, image_name, h, w):
    global annotation
    # 创建树根annotation
    annotation = ET.Element('annotation')
    # 创建一级分支folder
    folder = ET.SubElement(annotation, 'folder')
    # 添加folder标签内容
    folder.text = imgdir

    # 创建一级分支filename
    filename = ET.SubElement(annotation, 'filename')
    filename.text = image_name

    # 创建一级分支source
    source = ET.SubElement(annotation, 'source')
    # 创建source下的二级分支database
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    # 创建一级分支size
    size = ET.SubElement(annotation, 'size')
    # 创建size下的二级分支图像的宽、高及depth
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'

    # 创建一级分支segmented
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


weights = r'D:/System/yolov5-6.0/yolov5-6.0/best.pt'
opt_device = ''  # device = 'cpu' or '0' or '0,1,2,3'
imgsz = 640
opt_conf_thres = 0.25
opt_iou_thres = 0.5

# Initialize
set_logging()
device = select_device(opt_device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()  # to FP16

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


def predict_path(path, name2):
    box_list = []  # 创建坐标列表

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Set Dataloader & Run inference
    im0s = cv2.imread(path)  # BGR
    img = letterbox(im0s, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # pred = model(img, augment=opt.augment)[0]
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres)
    draw_1 = im0s

    # Process detections
    ret = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):

            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]}'
                prob = round(float(conf) * 100, 2)  # round 2
                xywh = xyxy
                x = (int(xywh[0]) + int(xywh[2])) / 2
                y = (int(xywh[1]) + int(xywh[3])) / 2
                w = (int(xywh[2]) - int(xywh[0]))
                h = (int(xywh[3]) - int(xywh[1]))

                a = math.ceil(x - w / 2)
                b = math.ceil(x + w / 2)
                cc = math.ceil(y - h / 2)
                d = math.ceil(y + h / 2)

                if label == 'cat':
                    box_list.append([a, cc, b, d, w])
                    print(1)
                    # draw_1 = cv2.rectangle(draw_1, (a, d), (b, cc), (0, 255, 0), 2)
                    # cv2.putText(draw_1, 'Cat', (a + 22, c - 8), 3, 1, (70, 255, 0), 1, cv2.LINE_AA)
                # elif label == 'dog':
                #     print(2)
                #     cv2.putText(raw_1, 'Dog', (a+22, c-8), 3, 1, (70, 255, 0), 1, cv2.LINE_AA)
                print('found:', x, y, w, h, label)

    # cv2.imwrite("D:/project/cat_sum/img/{}2/{}_{}.jpg".format(name2, name2, count), draw_1) # 将画过矩形框的图片保存到当前文件夹
    print(box_list)
    return box_list


for dirname, di, filenames in os.walk('D:/project/cat_sum/pic'):
    for dirn in di:
        # c=0
        for dirname2, di2, filenames2 in os.walk(os.path.join(dirname, dirn)):

            for fName in filenames2:
                image = cv2.imread(os.path.join(dirname2, fName))
                box_list = predict_path(os.path.join(dirname2, fName), os.path.basename(dirname2))
                (h, w) = image.shape[:2]
                create_tree(dirname2, fName, h, w)

                for box in box_list:
                    label_id = box[4]
                    create_object(annotation, box[0], box[1], box[2], box[3], os.path.basename(dirname2))
                    tree = ET.ElementTree(annotation)
                    # if coordinates_list==[]:
                    #     break

                    # 将树模型写入xml文件

                tree.write(
                    '{}/{}.xml'.format("D:/project/cat_sum/Annotation", fName.strip('.jpg')))

                # c = c + 1
