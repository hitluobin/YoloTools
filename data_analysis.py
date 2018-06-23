from __future__ import division
import os
from PIL import Image
import xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
import random
AnnoPath = 'F:/Code/Joya-cjzc/data/Annotations/'


annolist = os.listdir(AnnoPath)

dict = {}

index = 0
for anno in annolist:
	xmlfile = AnnoPath + anno
	#print(xmlfile)
	DomTree = xml.dom.minidom.parse(xmlfile)
	annotation = DomTree.documentElement
	objectlist = annotation.getElementsByTagName('object')
	for objects in objectlist:
		namelist = objects.getElementsByTagName('name')
		objectname = namelist[0].childNodes[0].data
		if dict.get(objectname,-1) == -1:
			dict[objectname] = 1
			index=index+1
		else:
			dict[objectname] = dict[objectname]+1
dict_len = len(dict)
dictsorted = sorted(dict.items(),key=lambda d:d[1], reverse = True)
for elements in dictsorted:
    print("the class: %s, the numbers: %d"%(elements[0],elements[1]))

#for key in dict:
#    print("u'%s',"%(key),end='')

#for key in dict:
#    print(key)