import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# output r: 
# 9毫米弹药
# 0.808782935143
#(822.6806030273438, 356.8907470703125, 46.227237701416016, 38.44511032104492)

imgfile = "F:\Code\Joya-cjzc\data\JPEGImages\IMG_0071.MP4_frame6903.jpg"
img = Image.open(img_file)
fig = plt.figure()
ax = fig.add_subplot(111)
for pred in r:
    classname = pred[0]
    prob = pred[1]
    box = pred[2]
    rect = plt.Rectangle((box[0],box[1]),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
plt.imshow(img)
