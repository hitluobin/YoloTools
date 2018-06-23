from __future__ import division
import os
from PIL import Image
import xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
import random
ImgPath = 'F:/Code/Joya-cjzc/data/JPEGImages/' 
AnnoPath = 'F:/Code/Joya-cjzc/data/Annotations/'


annolist = os.listdir(AnnoPath)
random.shuffle(annolist)


for anno in annolist:
    
    anno_pre, ext = os.path.splitext(anno)
    xmlfile = AnnoPath + anno 
    imgfile = ImgPath+anno_pre+".jpg"
    img = Image.open(imgfile)

    #print(xmlfile)
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement

    filenamelist = annotation.getElementsByTagName('filename')
    objectlist = annotation.getElementsByTagName('object')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        # print 'namelist:',namelist
        objectname = namelist[0].childNodes[0].data
        bndbox = objects.getElementsByTagName('bndbox')
        for box in bndbox: #only 1
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(x2_list[0].childNodes[0].data)
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)
            w = x2 - x1
            h = y2 - y1
            box = [x1,y1,x2,y2]
            rect = plt.Rectangle((x1,y1),w,h,linewidth=3,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
    plt.imshow(img)
    plt.ion()
    plt.pause(1)
    plt.close()
