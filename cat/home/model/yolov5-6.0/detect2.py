import math

import numpy as np

import cv2
import torch
from numpy import random
import evaluator

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    set_logging
from utils.torch_utils import select_device, time_sync


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


weights = r'D:/System/yolov5-6.0/yolov5-6.0/cat&dog.pt'
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


def predict_path(path):
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

    # Process detections
    ret = []
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            draw_1 = im0s
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
                position = (x, y, w, h)
                ret_i = [label, prob, position]
                ret.append(ret_i)
                a = math.ceil(x - w / 2)
                b = math.ceil(x + w / 2)
                c = math.ceil(y - h / 2)
                d = math.ceil(y + h / 2)

                if label == 'cat':
                    print(1)
                    draw_1 = cv2.rectangle(draw_1, (a, d), (b, c), (0, 255, 0), 2)
                    cv2.putText(draw_1, 'Cat', (a + 22, c - 8), 3, 1, (70, 255, 0), 1, cv2.LINE_AA)
                # elif label == 'dog':
                #     print(2)
                #     cv2.putText(draw_1, 'Dog', (a+22, c-8), 3, 1, (70, 255, 0), 1, cv2.LINE_AA)
                print('found:', x, y, w, h, label)
    cv2.imshow("draw_0", draw_1)  # 显示画过矩形框的图片
    c = cv2.waitKey(0)

    cv2.imwrite("D:/laptop/Desktop/camera/result.jpg", draw_1)  # 将画过矩形框的图片保存到当前文件夹
    print(ret)
    return ret


# predict example
# img_path = r"D:/laptop/Desktop/man-in-bed-cradling-cat.jpg"
# im0s = cv2.imread(img_path)  # BGR
# predict(im0s)
# predict_path(img_path)
# # # evaluator example
# evaluator.evaluator_v1(dir_path=r'/home/jiantang/桌面/enn/workcode/v5/yolov5/data_2c_val/img/',
#                        class_num=2,
#                        class_name=['chock-pair', 'wheel'],
#                        func=predict,
#                        threshold_1=0.5,
#                        threshold_2=0.25)
