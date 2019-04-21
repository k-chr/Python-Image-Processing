import numpy as np
import sys as kys
import os
from collections import defaultdict
from operator import attrgetter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect_
import cv2
from skimage.segmentation import chan_vese
from scipy.ndimage import binary_fill_holes, imread
from skimage.morphology import disk, star, diamond, medial_axis,octagon, rectangle, square, erosion, dilation, skeletonize
from skimage.measure import regionprops, label
from skimage.filters import gaussian, apply_hysteresis_threshold
from skimage.feature import canny
list1=[]
list2=["canny_gauss","manual","findContours","hysteresis_gauss","adaptive_gauss"]
img = cv2.imread("coins.png", cv2.IMREAD_GRAYSCALE)
img_filled = 255*binary_fill_holes(canny(gaussian(img, sigma=1)))
list1.append(img_filled)
img_filled = 255*binary_fill_holes(cv2.threshold(img.copy(), 130, 255, cv2.THRESH_BINARY)[1])
list1.append(img_filled)
img_filled, contours,_ = (cv2.findContours(cv2.threshold(img, 96, 255, cv2.THRESH_BINARY)[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
img_filled = 255*binary_fill_holes(255*img_filled)
list1.append(img_filled)
img_filled = (255*gaussian(img.copy(), sigma=1, mode='reflect')).astype(np.uint8)
img_filled = (255*(apply_hysteresis_threshold(img_filled/255, 0.3, 0.5))).astype(np.uint8)
list1.append(img_filled)

copy = img.copy()
cv2.adaptiveThreshold((255*gaussian(img.copy(), sigma=1, mode='reflect')).astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 2001, -22, copy)
list1.append(copy)
i=0;
for im in list1:
    img_data = regionprops(label(im//255, neighbors=8))
    max_area = max(img_data, key = attrgetter('area')).area
    print(sum(map(lambda item: item.area, img_data)))
    copy2 = np.zeros((img.shape[0], img.shape[1], 3))
    copy = im.copy()
    copy2[:,:,0] = copy
    copy2[:,:,1] = copy
    copy2[:,:,2] = copy
    for tup in img_data:
        x,y,w,h = tup.bbox
        colorc=(0,0,255) if max_area/tup.area > 1.3 else (255,0,0) 
        cv2.rectangle(copy2,(y,x),(h,w), colorc, 2)
    cv2.imwrite(list2[i]+ '_'+"pieniadz.jpg", copy2)
    i+=1
