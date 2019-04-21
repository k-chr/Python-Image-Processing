#lab3 zad2
import skimage as sk
from skimage import feature
import scipy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
img  = cv2.imread("gears.bmp", cv2.IMREAD_GRAYSCALE)
fun_list = [sk.filters.prewitt, sk.filters.sobel, sk.feature.canny]
name = ["prewitt_", "sobel_", "canny_"]
path = "edgy"
if path not in [dir_ for dir_ in os.listdir() if os.path.isdir(dir_) == True]:
    os.mkdir(path)

#Basic edges-detection algorithms (Canny, Prewitt, Sobel)
for fun in fun_list:
    for g in range(3):
        for i in range(2):
            img_c = img.copy() if g == 0 else (255*sk.filters.gaussian(img.copy(), sigma=g, mode='reflect')).astype(np.uint8)
            img_c = img_c if i == 0 else cv2.threshold(img_c.copy(), 157, 255, cv2.THRESH_BINARY)[1]
            list1 = []
            img_c = (fun(img_c)*255).astype(np.uint8)
            list1.append(img)
            list1.append(img_c)
            list1.append(255* sc.ndimage.binary_fill_holes(img_c))
            cv2.imwrite(path + '/' + name[fun_list.index(fun)] + "gauss_{0}_threshed_{1}".format(g,i)+".png", np.hstack(list1))

#cv2.findContours algorithm
mode = {"RETR_EXTERNAL_" : cv2.RETR_EXTERNAL,
        "RETR_LIST_" : cv2.RETR_LIST,
        "RETR_CCOMP_" : cv2.RETR_CCOMP,
        "RETR_TREE_" : cv2.RETR_TREE}
method = {"CHAIN_APPROX_NONE_" : cv2.CHAIN_APPROX_NONE,
          "CHAIN_APPROX_SIMPLE_" : cv2.CHAIN_APPROX_SIMPLE,
          "CHAIN_APPROX_TC89_KCOS_" : cv2.CHAIN_APPROX_TC89_KCOS,
          "CHAIN_APPROX_TC89_L1_" : cv2.CHAIN_APPROX_TC89_L1}
for edom in mode.items():
    for dohtem in method.items():
        for i in range(2):
            list1 = []
            img_c = img.copy() if i == 0 else cv2.threshold(img.copy(), 155, 255, cv2.THRESH_BINARY_INV)[1]
            list1.append(img.copy())
            img_c, contours,_ = (cv2.findContours(img_c, edom[1], dohtem[1]))      
            list1.append(cv2.drawContours(np.zeros(img.shape), contours, -1, 255, 1))
            list1.append(255* sc.ndimage.binary_fill_holes(img_c))
            cv2.imwrite(path + '/' + "cv2FindContours_" + edom[0] + dohtem[0] + "Thresh_{0}".format(i) + ".png", np.hstack(list1))

#hysteresis
img4 = 255 - (255*sk.filters.gaussian(img.copy(), sigma=0.65, mode='reflect')).astype(np.uint8)
cv2.imwrite(path+ '/' + "thresh.png",img4)
img4 = 255*(sk.filters.apply_hysteresis_threshold(img4/255, 0.43, 0.55)).astype(np.uint8)
holes = 255*sc.ndimage.binary_fill_holes(img4)
cv2.imwrite(path+ '/' + "hysteresis.png", np.hstack([img,img4,holes]))

img4 = (255*sk.filters.gaussian(img.copy(), sigma=0.4, mode='reflect')).astype(np.uint8)
img4 = 255*sk.filters.sobel(cv2.threshold(img4.copy(), 155, 255, cv2.THRESH_BINARY_INV)[1])
cv2.imwrite(path+ '/' + "initsobel.png",img4)
img4 = 255*(sk.filters.apply_hysteresis_threshold(img4/255, 0.3, 0.45)).astype(np.uint8)
holes = 255*sc.ndimage.binary_fill_holes(img4)
cv2.imwrite(path+ '/' + "hysteresis2.png", np.hstack([img,img4,holes]))

