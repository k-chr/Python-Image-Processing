import skimage as sk
from skimage import feature
import scipy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import disk,binary_dilation, binary_closing, binary_erosion, star, diamond, octagon, rectangle, square, erosion, dilation, skeletonize
img = sk.io.imread('teeth.jpg', True)
grey = img.copy()
img_t = cv2.threshold(255*grey, 160, 170, cv2.THRESH_BINARY_INV)[1]
img_tmp  = 255 -img_t
plt.axis('off')
plt.imshow(img_tmp, cmap='gray')
plt.show()
img_e = binary_dilation(img_t, rectangle(2,3))
img_tmp  = 255- 255*img_e 
plt.imshow(img_tmp, cmap='gray')
plt.show()
img_e = binary_closing(img_e, rectangle(8,2))
img_tmp  = 255- 255*img_e 
plt.imshow(img_tmp, cmap='gray')
plt.show()
img_e = binary_dilation(img_e, octagon(10,2))
img_tmp  = 255- 255*img_e 
plt.imshow(img_tmp, cmap='gray')
plt.show()

img_e = binary_dilation(img_e, star(8))
img_tmp  = 255- 255*img_e 
plt.imshow(img_tmp, cmap='gray')
plt.show()
for i in range(3):
    img_e = binary_erosion(img_e, star(3 + i))
img_e  = 255- 255*img_e 
plt.imshow(img_e, cmap='gray')
plt.axis('off')
plt.show()
