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


# Define a function that creates a first-level branch object
def create_object(root, xi, yi, xa, ya, obj_name):  # 参数依次，树根，xmin，ymin，xmax，ymax
    # Create a first-level branch object
    _object = ET.SubElement(root, 'object')
    # Create a second-level branch object
    name = ET.SubElement(_object, 'name')
    print(obj_name)
    name.text = str(obj_name)
    pose = ET.SubElement(_object, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(_object, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(_object, 'difficult')
    difficult.text = '0'
    # create bndbox
    bndbox = ET.SubElement(_object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = '%s' % xi
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = '%s' % yi
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = '%s' % xa
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = '%s' % ya


# Functions for creating xml files
def create_tree(imgdir, image_name, h, w):
    global annotation
    # Create a tree root annotation
    annotation = ET.Element('annotation')
    # 创建一级分支folder
    folder = ET.SubElement(annotation, 'folder')
    # Create a first-level branch folder
    folder.text = imgdir

    # Create first level branch filename
    filename = ET.SubElement(annotation, 'filename')
    filename.text = image_name

    # Create first level branch source
    source = ET.SubElement(annotation, 'source')
    # Create a secondary branch database under source
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    # Create first level branchsize
    size = ET.SubElement(annotation, 'size')
    # Create the width, height and depth of the secondary branch image under size
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'

    # Create first level branch segmented
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


weights = r'D:/System/yolov5-6.0/yolov5-6.0/runs/train/exp20/weights/best.pt'
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


def identify_breed(imag, xmi, xma, ymi, yma):
    label2 = "cat"
    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Set Dataloader & Run inference
    im0s1 = cv2.imread(imag)  # BGR
    im0s1 = im0s1[ymi:yma, xmi:xma]
    img1 = letterbox(im0s1, new_shape=imgsz)[0]

    # Convert
    img1 = img1[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img1 = np.ascontiguousarray(img1)

    img1 = torch.from_numpy(img1).to(device)
    img1 = img1.half() if half else img1.float()  # uint8 to fp16/32
    img1 /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img1.ndimension() == 3:
        img1 = img1.unsqueeze(0)

    # Inference
    # pred = model(img1, augment=opt.augment)[0]
    pred = model(img1)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres)

    # Process detections
    ret = []

    for i, det in enumerate(pred):  # detections per image
        print(det)
        print(len(det))
        print(len(det))
        print(len(det))
        if len(det) > 0:
            print(len(det))
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img1.shape[2:], det[:, :4], im0s1.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label2 = f'{names[int(cls)]}'
                # cv2.putText(im0s1, label2, (xmi + 22, ymi - 8), 3, 1, (70, 255, 0), 1, cv2.LINE_AA)
                # cv2.imshow('image', im0s1)
    return label2
