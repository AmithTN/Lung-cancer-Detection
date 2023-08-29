# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:53:59 2022

@author: LINGARAJ
"""
from skimage.segmentation import clear_border
from skimage import io
from sklearn import cluster
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_multiotsu
from skimage.color import rgb2gray
import numpy as np


from skimage.filters import threshold_multiotsu

# read input and convert to range 0-1

def segmenation(path,res):
    g_image = io.imread(path)
    
    
    image = rgb2gray(g_image)
    
    # display result
    io.imshow(image)
    io.show()
    h, w = image.shape
    # reshape to 1D array
    image_2d = image.reshape(h*w,1)
    
    # set number of colors
    numcolors = 4
    
    # do kmeans processing
    kmeans_cluster = cluster.KMeans(n_clusters=int(numcolors))
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    
    # need to scale result back to range 0-255
    newimage = cluster_centers[cluster_labels].reshape(h, w)*255.0
    newimage = newimage.astype('uint8')
    
    io.imshow(newimage)
    io.show()
    # threshold to keep only middle gray values
    lower = (100)
    upper = (200)
    thresh = cv2.inRange(newimage, lower, upper)
    io.imshow(thresh)
    io.show()
    # get contours and corresponding areas and indices
    cntrs_info = []
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    index=0
    for cntr in contours:
        area = cv2.contourArea(cntr)
        cntrs_info.append((index,area))
        index = index + 1
    
    # sort contours by area
    def takeSecond(elem):
        return elem[1]
    cntrs_info.sort(key=takeSecond, reverse=True)
    
    # draw two largest contours as white filled on black background
    result = np.zeros_like(newimage)
    index_first = cntrs_info[0][0]
    cv2.drawContours(result,[contours[index_first]],0,(255),-1)
    index_second = cntrs_info[1][0]
    cv2.drawContours(result,[contours[index_second]],0,(255),-1)
    
    # display result
    io.imshow(result)
    io.show()
    
    
    thresholds = threshold_multiotsu(image,classes=4)
    
    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)
    
    io.imshow(regions)
    io.show()
    w,h  = regions.shape
    lesion = np.zeros(shape=(w,h))
    for i in range(w):
      for j in range(h):
          #if punomina use this not in [2]
          #if covid use in[1,2]
        if res == 0:
           if regions[i][j]  not in [2]:
             lesion[i][j]=0
           else:
             lesion[i][j]=1 
        else:
            if regions[i][j]   in [1,2]:
              lesion[i][j]=0
            else:
              lesion[i][j]=1
          
    final_lesion = clear_border(lesion,mask=result.astype(bool))
    io.imshow(final_lesion)
    io.show()
    def compute_area(mask, pixdim):
        """
        Computes the area (number of pixels) of a binary mask and multiplies the pixels
        with the pixel dimension of the acquired CT image
        Args:
            lung_mask: binary lung mask
            pixdim: list or tuple with two values
    
        Returns: the lung area in mm^2
        """
        mask[mask >= 1] = 1
      
        lung_pixels = np.sum(mask)
        return lung_pixels * pixdim[0] * pixdim[1]
    
    area_lung = compute_area(result,(200,200))
    area_lesion = compute_area(final_lesion,(200,200))
    
    per = (area_lesion/area_lung)*100
    
    
   
    
    return per 
