import argparse
import time
from pathlib import Path

import os
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def label_xywh2xyxy(x):
    y =  np.copy(x)
    y[1,:]=y[0,:]
    y[0][0]-=x[1][0]*0.5
    y[0][1]-=x[1][1]*0.5

    y[1][0]+=x[1][0]*0.5
    y[1][1]+=x[1][1]*0.5
    return y
def label_xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y =  np.copy(x)
    y[0][0] = x[:,0].mean()  # x center
    y[0][1] = x[:,1].mean()  # y center
    y[1][0] = x[1][0] - x[0][0]   # width
    y[1][1] = x[1][1] - x[0][1]  # height
    return y


def make_facenao_lines(face_label_list,img):
    # h,w,c=img.shape
    h=1
    w=1

    lines=''
    for face_label in face_label_list:
        line='0 '
        rect_label,faceland_label=face_label

        rect_label_xywh=np.array(rect_label)
        rect_label_xywh=label_xyxy2xywh(rect_label_xywh)
        for pt in rect_label_xywh:
            line+=str(float(pt[0])/w)+' '+str(float(pt[1])/h)+' '

        for pt in faceland_label:
            line+=str(float(pt[0])/w)+' '+str(float(pt[1])/h)+' 2.0 '
        line=line.rstrip(' ')+'\n'
        lines+=line
    lines=lines.rstrip('\n')
    return lines


def parse_path(filepath):
    filename=os.path.basename(filepath)
    ext='.'+filename.split('.')[-1]
    path_noext=filepath[0:len(filepath)-len(ext)]
    filekey=filename[0:len(filename)-len(ext)]
    return filekey,ext,path_noext

def imgpath_to_txtpath(imgpath):
    filekey,ext,path_noext=parse_path(imgpath)
    txtpath=path_noext+'.txt'
    return txtpath

def detect():
    # weights=r'.\weights\yolov7s-face.pt'
    weights=r'.\weights\yolov7-w6-face.pt'

    source=r'Z:\workspace\yoloface\dataset\crawler'
    kpt_label=5
    imgsz=640
    conf_thres=0.25
    iou_thres=0.45
    classes=0
    agnostic_nms=False


    device = select_device('cpu')
    half = False  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if isinstance(imgsz, (list,tuple)):
        assert len(imgsz) ==2; "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16


    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:

        print(path)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]
        # print('+++',pred)
        # print(pred[...,4].max())
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres,iou_thres, classes=classes, agnostic=agnostic_nms, kpt_label=kpt_label)
        t2 = time_synchronized()

        # print('---',pred)

        # Process detections
        face_label_list=[]
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # print('det')

                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)
                
                imh,imw,imc=im0.shape
                # print('imc,imh,imw:',imc,imh,imw)

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # kpts = det[det_index, 6:]
                    num_array=det[det_index,:].cpu().numpy()
                    # print(np.arange(2, num_array.shape[0] - 5,3))
                    pts_array = np.delete(num_array[6:], np.arange(2, num_array.shape[0] - 6,3)) 
                    kpts=pts_array

                    # print(pts_array)
                    print(kpts)
                    # exit(0)
                    kpts[::2]/=imw
                    kpts[1::2]/=imh
                    kpts=kpts.tolist()

                    # print(xywh)
                    # print(kpts)

                    kpts=np.array(kpts).reshape(5,-1)

                    xywh=np.array(xywh).reshape(2,2)
                    xyxy=label_xywh2xyxy(xywh)

                    face_label_list.append((list(xyxy),list(kpts)))

        ano_lines=make_facenao_lines(face_label_list,im0)
        txt_path=imgpath_to_txtpath(path)
        print(ano_lines)
        with open(txt_path,'w') as f:
            f.writelines(ano_lines)

if __name__ == '__main__':
    detect()
