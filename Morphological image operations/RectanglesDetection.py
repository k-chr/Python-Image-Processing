#lab5 zad2
import skimage as sk
from skimage import feature
import scipy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import disk, star, diamond, octagon, rectangle, square, erosion, dilation, skeletonize

img = sk.io.imread("figures.bmp", True)
mask = sk.io.imread("mask3.bmp", True)
mask2 = sk.io.imread("mask2.bmp", True)
hit = sc.ndimage.morphology.binary_hit_or_miss(img, mask)
print(len(np.nonzero(hit)[0]))
found = 0
for i in range(img.shape[0] - mask.shape[0]):
    for j in range(img.shape[1]-mask.shape[1]):
        if (mask == np.array(img[i:(i+mask.shape[0]),j:(j+mask.shape[1])])).all():
            found += 1 
print(found)
plt.imshow(hit, cmap = "gray")
plt.axis('off')
plt.show()
