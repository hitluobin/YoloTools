import os
import xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
AnnoPath = 'F:/Code/Joya-cjzc/data/Annotations/'


annolist = os.listdir(AnnoPath)
lens = len(annolist)
dst_boxes = []

dst_w = 416
dst_h = 416


for anno in annolist:
    xmlfile = AnnoPath + anno
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement
    
    filenamelist = annotation.getElementsByTagName('filename')
    objectlist = annotation.getElementsByTagName('object')
    sizelist = annotation.getElementsByTagName('size')
    '''if sizelist.length == 0:
        os.remove(xmlfile)
        print(xmlfile)
        continue'''
    widthnode = sizelist[0].getElementsByTagName('width')
    heightnode = sizelist[0].getElementsByTagName('height')
    src_w = widthnode[0].childNodes[0].data
    src_h = heightnode[0].childNodes[0].data
    
    scale_w = float(src_w)/dst_w
    scale_h = float(src_h)/dst_h

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
            dst_boxes.append([w/scale_w,h/scale_h])

#print(boxeswh)
#boxeswh=np.array(boxeswh)
from sklearn.cluster import KMeans
estimator = KMeans(n_clusters=9)
result = estimator.fit(dst_boxes)
dst_boxes=np.array(dst_boxes)
fig = plt.figure(1)
ax = fig.add_subplot(111)
classes = list(estimator.labels_)
#facecolors = ['b','g','r','c','m','y','k','w','orange']
plt.scatter(dst_boxes[:,0],dst_boxes[:,1],15.0*np.array(classes),15.0*np.array(classes))
print(estimator.cluster_centers_)
plt.show()

