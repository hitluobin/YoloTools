#!/usr/bin/python
# -*- coding: utf-8 -*-
# detector for cjzc
from ctypes import *
import os
import cv2
import math
import random

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
base_path = os.path.split(os.path.realpath(__file__))[0]
lib = CDLL("{}/libdarknet.so".format(base_path), RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

# # display the pic after detecting. 2018.04.25
# def showPicResult(image, r, out_img="./predictions.jpg"):
#     img = cv2.imread(image)
#     cv2.imwrite(out_img, img)
#     '''
#     r是一个list，其中存放了检测图片的信息，包括class（检测到的物体类别）、置信度、坐标，其中坐标包括x、y、w、h。
#     x、y：从左上角到物体中心点距离（float类型）；w、h：目标区域的宽和高（float类型）。
#     '''
#     for i in range(len(r)):
#         x1=r[i][2][0]-r[i][2][2]/2
#         y1=r[i][2][1]-r[i][2][3]/2
#         x2=r[i][2][0]+r[i][2][2]/2
#         y2=r[i][2][1]+r[i][2][3]/2
#         im = cv2.imread(out_img)
#         # 此函数实在目标区域绘制一个矩形
#         # 参数1：原图片，参数2：绘制矩形的左上角坐标
#         # 参数3: 右下角坐标，参数4: 矩形线条颜色，参数5: 线条粗细。
#         #（由于cv2中坐标为整数，所以程序中进行了类型转换float-->int）
#         cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3)
#         #This is a method that works well.
#         cv2.imwrite(out_img, im)
#     cv2.imshow('yolo_image_detector', cv2.imread(out_img))
#     cv2.waitKey(10)
#     #cv2.destroyAllWindows()

#base_path = os.path.split(os.path.realpath(__file__))[0]
#net = load_net(base_path + "/cfg/luobin_logo_yolov3-tiny.cfg", base_path + "/model/luobin_logo_yolov3-tiny_40000.weights", 0)
#meta = load_meta(base_path + "/cfg/luobin_logo.data")



if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    net = load_net("./cfg/joyachen_cjzc_yolov3.cfg ", "./backup/joyachen_cjzc_yolov3.backup", 0)
    meta = load_meta("/data1/joyachen/data/cjzc_data/joya_cjzc.data")
    import time
    while True:
        path = input("input your img or video road : ")
        detect_begin = time.time()
        r = detect(net, meta, image_path)
        detect_end = time.time()
        


        print("detect cost time: {}".format(str(detect_end - detect_begin)))
        print(r)
        