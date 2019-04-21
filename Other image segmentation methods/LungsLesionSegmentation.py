#lab4 zad1
import skimage as sk
from skimage import feature
import scipy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
img = sk.io.imread("lungs_lesion.bmp")
h_img1 = sk.io.imread("lungs_lesion_seeds1.bmp")
h_img2 = sk.io.imread("lungs_lesion_seeds2.bmp")
h_img3 = sk.io.imread("lungs_lesion_seeds3.bmp")
copy = img.copy()
for i in range(0, 100000, 5000):
    new = sk.segmentation.random_walker(img.copy(), h_img1, beta = i)
    print(new)
    lsit1 = []
    lsit1.append(img)
    lsit1.append(h_img1)
    lsit1.append(255*new)
    xd = np.vstack(lsit1)
    sk.io.imshow(xd, cmap = 'gray')
    sk.io.show()
