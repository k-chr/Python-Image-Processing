#lab6_zad3
import numpy as np
import sys as kys
import os
from collections import defaultdict
from operator import attrgetter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect_
import cv2
from scipy.ndimage import binary_fill_holes, imread
from skimage.morphology import disk, star, diamond, medial_axis,octagon, rectangle, square, erosion, dilation, skeletonize
from skimage.measure import regionprops, label
from skimage.feature import canny
img = cv2.imread('planes.png', cv2.IMREAD_GRAYSCALE)
copy = 255 - img.copy()
img_data = regionprops(label(copy//255, neighbors=8))
min_solidity = min(img_data, key = attrgetter('solidity')).solidity
min_extent =  min(img_data, key = attrgetter('extent')).extent
k
print(sum(map(lambda item: item.area, img_data)))
copy2 = np.zeros((img.shape[0], img.shape[1], 3))
copy2[:,:,0] = img
copy2[:,:,1] = img
copy2[:,:,2] = img
copy3 = copy2.copy()
copy4 = copy3.copy()
i = 0
for tup in img_data:
    x,y,w,h = tup.bbox
    colorc=(255,0,0) if min_solidity/tup.solidity > 0.9 else (0,0,255) 
    if min_solidity/tup.solidity > 0.9:
        cv2.rectangle(copy2,(y,x),(h,w), colorc, 2)
        i+=1
print(i)
cv2.imwrite("findPlane.jpg", copy2)
i = 0
for tup in img_data:
    x,y,w,h = tup.bbox
    colorc=(0,255,0) if min_extent/tup.extent > 0.9 else (0,0,255) 
    if min_extent/tup.extent > 0.9:
        cv2.rectangle(copy3,(y,x),(h,w), colorc, 2)
        i+=1
print(i)
cv2.imwrite("findPlane2.jpg", copy3)
i = 0
for tup in img_data:
    x,y,w,h = tup.bbox
    colorc=(0,255,0) if min_extent/tup.extent > 0.9 else (0,0,255) if min_extent/tup.extent > 0.55 else (255,0,0) 
    if colorc == (0,255,0):
        i+=1
    cv2.rectangle(copy4,(y,x),(h,w), colorc, 2)
print(i)
cv2.imwrite("findPlane3.jpg", copy4)
