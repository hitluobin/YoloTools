from os import listdir
import argparse
import numpy as np
import sys
import os
import shutil
import random 
import math
import matplotlib.pyplot as plt

# :) 这里设置自己的参数和路径

# 网络中用于训练的宽与高
width_in_cfg_file = 416.
height_in_cfg_file = 416.

# 包含有每个图像标注方框 txt 的路径（即darknet数据准备中的labels文件夹）
labelspath = "F:/Code/Joya-cjzc/data/labels/"

# 聚类个数
num_clusters = 9

# :0

def IOU(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else:
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) 
    return np.array(similarities) 
def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        sum+= max(IOU(X[i],centroids)) 
    return sum/n
def visualization(X,classindices):
    # have a look
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0],X[:,1],15.0*np.array(classindices),15.0*np.array(classindices))
    plt.show()
def kmeans(X,centroids,eps):
    N = X.shape[0] #rows
    iterations = 0
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)    
    iter = 0
    old_D = np.zeros((N,k))
    while True:
        D = [] 
        iter+=1           
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D)
        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D)))) 
        assignments = np.argmin(D,axis=1) # 取出每个点最近的一个中心
        
        if (assignments == prev_assignments).all() :
            anchors = centroids.copy()
            print(anchors.shape)

            widths = anchors[:,0]
            sorted_indices = np.argsort(widths)
            visualization(X,assignments)
            print('Anchors = \n', anchors[sorted_indices])
            return
        # 重置中心点
        centroid_sums=np.zeros((k,dim),np.float)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]        
        for j in range(k):            
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))
        
        prev_assignments = assignments.copy()     
        old_D = D.copy()  

if __name__=="__main__":
    
    annotation_dims = []
    labelsfnames = os.listdir(labelspath)

    size = np.zeros((1,1,3))
    for labelsfname in labelsfnames:
        f = open(labelspath + labelsfname)
        for line in f.readlines():
            line = line.rstrip('\n')
            w,h = line.split(' ')[3:]            
            annotation_dims.append((float(w)*width_in_cfg_file,float(h)*height_in_cfg_file))
    annotation_dims = np.array(annotation_dims)
  
    eps = 0.005
    # 原点中随机取9个点
    indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
    centroids = annotation_dims[indices]
    kmeans(annotation_dims,centroids,eps)
