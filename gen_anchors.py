
from os import listdir
import argparse
#import cv2
import numpy as np
import sys
import os
import shutil
import random 
import math
import matplotlib.pyplot as plt
width_in_cfg_file = 416.
height_in_cfg_file = 416.

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
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities) 

def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
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
            d = 1 - IOU(X[i],centroids)  # 每个点和中心的距离
            D.append(d)
        D = np.array(D) # D.shape = (N,k)
        
        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))
            
        #assign samples to centroids 
        assignments = np.argmin(D,axis=1) # 取出每个点最近的一个中心？
        
        if (assignments == prev_assignments).all() :
            anchors = centroids.copy()
            print(anchors.shape)

            widths = anchors[:,0]
            sorted_indices = np.argsort(widths)
            visualization(X,assignments)
            print('Anchors = \n', anchors[sorted_indices])
            return

        #calculate new centroids
        #就是求平均啊(′д｀ )…彡…彡
        centroid_sums=np.zeros((k,dim),np.float)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]        
        for j in range(k):            
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))
        
        prev_assignments = assignments.copy()     
        old_D = D.copy()  

def main(argv):
    '''parser = argparse.ArgumentParser()
    parser.add_argument('-filelist', default = '//path//to//voc//filelist//train.txt', 
                        help='path to filelist/n' )
    parser.add_argument('-output_dir', default = 'generated_anchors/anchors', type = str, 
                        help='Output anchor directory/n' )  
    parser.add_argument('-num_clusters', default = 0, type = int, 
                        help='number of clusters/n' )  

   
    args = parser.parse_args()'''
    
    tralistfile = "F:/Code/Joya-cjzc/data/tra.txt"
    num_clusters = 9


    f = open(tralistfile)
    tralist_ = f.readlines() 
    tralist = [line.rstrip('\n') for line in tralist_]
    tralist = [line.replace("/data1/joyachen/data/cjzc_data/JPEGImages","F:/Code/Joya-cjzc/data/labels") for line in tralist]
    tralist = [line.replace(".jpg",".txt") for line in tralist]
    f.close()
    
    annotation_dims = []
    
    size = np.zeros((1,1,3))
    for line in tralist:

        print(line)
        f = open(line)
        for line in f.readlines():
            line = line.rstrip('\n')
            w,h = line.split(' ')[3:]            
            #print(w,h)
            #annotation_dims.append(map(float,(w,h)))
            annotation_dims.append((float(w)*width_in_cfg_file,float(h)*height_in_cfg_file))
    annotation_dims = np.array(annotation_dims)
  
    eps = 0.005
    # 原点中随机取9个点
    indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
    centroids = annotation_dims[indices]
    kmeans(annotation_dims,centroids,eps)
    print('centroids.shape', centroids.shape)

if __name__=="__main__":
    main(sys.argv)
