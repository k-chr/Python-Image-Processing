#lab3 zad3
import skimage as sk
from skimage import feature
import scipy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import disk
img = cv2.imread("fish.bmp")
copy = img.copy()

paths = ["felzenshwalb_", "watershed_", "slic_", "quickshift_"]
for path in paths:
    if path not in [dir_ for dir_ in os.listdir() if os.path.isdir(dir_) == True]:
        os.mkdir(path)
        
for n in range (500, 2001, 500):
    for j in np.arange (-1.1, 1.1, 0.001):
        new = sk.segmentation.watershed(feature.canny(cv2.cvtColor(copy.copy(), cv2.COLOR_BGR2GRAY), sigma=0.001), markers = n, compactness=j)
        cv2.imwrite(paths[1]+ '/' + "markers_{0}_compactness_{1}_.jpg".format(n,j), 255*sk.segmentation.mark_boundaries(img, new, color = (255, 5, 45), mode = 'thick'), [int(cv2.IMWRITE_JPEG_QUALITY), 75])

for i in range(0, 281, 10):
    for j in np.arange(-11, 41, 1):
        new = sk.segmentation.felzenszwalb(copy.copy(), scale = i, sigma = j, min_size = 0)
        cv2.imwrite(paths[0]+ '/' + "scale_{0}_sigma_{1}_.jpg".format(i,j), 255*sk.segmentation.mark_boundaries(img, new, color = (255, 5, 45), mode = 'thick'), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        
for i in range(100, 1601, 500):
    for n in range (100, 501, 100):
        for c in np.arange (1, 15, 0.5):
            for s in range( 1, 15): 
                new = sk.segmentation.slic(copy.copy(), n_segments = n, compactness = c, max_iter = i, sigma = s, spacing = None, min_size_factor = 0.5, max_size_factor=3)
                cv2.imwrite(paths[2]+ '/' + "segments_{0}_compactness_{1}_iter_{2}_sigma_{3}_.jpg".format(i,n,c,s), 255*sk.segmentation.mark_boundaries(img, new, color = (255, 5, 45), mode = 'thick'), [int(cv2.IMWRITE_JPEG_QUALITY), 75])

for i in range(2, 17 ,2):
    for j in np.arange(-2, 6.1, 0.5):
        for k in np.arange(-2, 9, 1):
            new = sk.segmentation.quickshift(copy.copy(), ratio=j, kernel_size=i, max_dist=11, return_tree=False, sigma=k)
            cv2.imwrite(paths[3]+ '/' + "ratio_{0}_kernel_{1}_sigma_{2}_seed_{3}_.jpg".format(j,i,k,r), 255*sk.segmentation.mark_boundaries(img, new, color = (255, 5, 45), mode = 'thick'), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
