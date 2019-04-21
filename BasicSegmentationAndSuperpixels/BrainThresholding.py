#lab3 zad1
import skimage as sk
from skimage.filters import threshold_otsu, threshold_yen, threshold_li, threshold_isodata, threshold_triangle, threshold_mean, threshold_minimum
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
#my threshold, initially based on cv2 TRUNC thresh. Written for initial preprocessing
def my_thresh(nd_arr, _type_):
    return nd_arr if _type_ == 0 else np.array([[0 if nd_arr[i][j] < 40 else 40 if nd_arr[i][j] < \
           220 else nd_arr[i][j] for j in range(nd_arr.shape[1])] for i in range(nd_arr.shape[0])]) if _type_ == 1 else \
           np.array([[0 if nd_arr[i][j] < 122 else nd_arr[i][j] for j in range(nd_arr.shape[1])] for i in range(nd_arr.shape[0])]) \
           if _type_ == 2 else np.array([[0 if nd_arr[i][j] < 85 else 122 if nd_arr[i][j] < \
           171 else nd_arr[i][j] for j in range(nd_arr.shape[1])] for i in range(nd_arr.shape[0])])

str_list = ["otsu","yen","li","isodata","triangle","mean","minimum"]
my_tr = ["no_thresh_","40_220_thresh_","122_thresh_","85_171_thresh_"]
img  = cv2.imread("brain_tumor.bmp", cv2.COLOR_BGR2GRAY)
list1 = [threshold_otsu, threshold_yen, threshold_li, threshold_isodata, threshold_triangle, threshold_mean, threshold_minimum]
is_Gauss = [0,1,2]

#manual thresh
copy = img.copy()
cv2.threshold(img.copy(), 222, 255, cv2.THRESH_BINARY, copy)
cv2.imwrite("manual_thresh.png", copy)
copy = 255*sk.filters.gaussian(img.copy(), sigma=1, mode='reflect')
cv2.threshold(copy.copy(), 222, 255, cv2.THRESH_BINARY, copy)
cv2.imwrite("manual_thresh_gauss1.png", copy)
copy = 255*sk.filters.gaussian(img.copy(), sigma=2, mode='reflect')
cv2.threshold(copy.copy(), 222, 255, cv2.THRESH_BINARY, copy)
cv2.imwrite("manual_thresh_gauss2.png", copy)

#algorithms computing automatically global value of threshold
for n in is_Gauss:
    new = 255*sk.filters.gaussian(img.copy(), sigma=n, mode='reflect') if n > 0 else img.copy()
    for i in range(4):
        copy =  my_thresh(new, i)
        gauss = "no_Gauss" if n == 0 else "Gauss_Sigma=1" if n == 1 else "Gauss_Sigma=2"
        path = "init_" + my_tr[i] + gauss
        if path not in [dir_ for dir_ in os.listdir() if os.path.isdir(dir_) == True]:
            os.mkdir(path)
        for func in list1:
            cv2.imwrite(path + '/' + str_list[list1.index(func)]+".png", 255* (copy > func(copy)))

#adaptive/ local thresholding (mean/Gaussian)            
copy = img.copy()
path = "adaptive_thresh"
if path not in [dir_ for dir_ in os.listdir() if os.path.isdir(dir_) == True]:
            os.mkdir(path)
for n in is_Gauss:
    new = np.array(255*sk.filters.gaussian(img.copy(), sigma=n, mode='reflect')).astype(np.uint8) if n > 0 else img.copy()
    for i in range(3, 20004, 1000):
        for j in range(-22, 23, 10):
            cv2.adaptiveThreshold(new.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, i, j, copy)
            cv2.imwrite(path + '/' + "gauss_filtered_{2}_adaptive_gauss_{0}_{1}.png".format(i, j,n), copy)
            cv2.adaptiveThreshold(new.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, i, j, copy)
            cv2.imwrite(path + '/' + "gauss_filtered_{2}_adaptive_mean_{0}_{1}.png".format(i, j,n), copy)

