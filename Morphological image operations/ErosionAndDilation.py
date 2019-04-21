#lab5 zad1
import skimage as sk
from skimage import feature
import scipy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import disk, star, diamond, octagon, rectangle, square, erosion, dilation
img = 255 - sk.io.imread("bw1.bmp", True)
img_e = img.copy()
img_d = img.copy()
for i in range(0,3):
    img_e = erosion(img_e.copy(), rectangle(6,4))
    img_d = dilation(img_d.copy(), rectangle(6,4))
    plt.imshow(np.hstack([255-img,255-img_e, 255-img_d]), cmap='gray')
    plt.title("rectangle(6,4), iteration: " + str(i+1), {'fontsize': 28,
                                 'fontweight' : 'bold',
                                 'verticalalignment': 'baseline',
                                 'horizontalalignment': 'center'})
    plt.axis('off')
    plt.show()

img_e = img.copy()
img_d = img.copy()
for i in range(0,3):
    img_e = erosion(img_e.copy(), disk(5))
    img_d = dilation(img_d.copy(), disk(5))
    plt.imshow(np.hstack([255-img,255-img_e, 255-img_d]), cmap='gray')
    plt.title("disk(5), iteration: " + str(i+1), {'fontsize': 28,'fontweight' : 'bold', 'verticalalignment': 'baseline','horizontalalignment': 'center'})

    plt.axis('off')
    plt.show()
    

img_e = img.copy()
img_d = img.copy()
for i in range(0,3):
    img_e = erosion(img_e.copy(), star(5))
    img_d = dilation(img_d.copy(), star(5))
    plt.imshow(np.hstack([255-img,255-img_e, 255-img_d]), cmap='gray')
    plt.title("star(5), iteration: " + str(i+1), {'fontsize': 28,
                                 'fontweight' : 'bold',
                                 'verticalalignment': 'baseline',
                                 'horizontalalignment': 'center'})
    plt.axis('off')
    plt.show()

img_e = img.copy()
img_d = img.copy()
for i in range(0,3):
    img_e = erosion(img_e.copy(), diamond(5))
    img_d = dilation(img_d.copy(), diamond(5))
    plt.imshow(np.hstack([255-img,255-img_e, 255-img_d]), cmap='gray')
    plt.title("diamond(5), iteration: " + str(i+1), {'fontsize': 28,
                                 'fontweight' : 'bold',
                                 'verticalalignment': 'baseline',
                                 'horizontalalignment': 'center'})
    plt.axis('off')
    plt.show()
img_e = img.copy()
img_d = img.copy()
for i in range(0,3):
    img_e = erosion(img_e.copy(), octagon(5,2))
    img_d = dilation(img_d.copy(), octagon(5,2))
    plt.imshow(np.hstack([255-img,255-img_e, 255-img_d]), cmap='gray')
    plt.title("octagon(5,2), iteration: " + str(i+1), {'fontsize': 28,
                                 'fontweight' : 'bold',
                                 'verticalalignment': 'baseline',
                                 'horizontalalignment': 'center'})
    plt.axis('off')
    plt.show()
img_e = img.copy()
img_d = img.copy()
for i in range(0,3):
    img_e = erosion(img_e.copy(), rectangle(5,2))
    img_d = dilation(img_d.copy(), rectangle(5,2))
    plt.imshow(np.hstack([255-img,255-img_e, 255-img_d]), cmap='gray')
    plt.title("rectangle(5,2), iteration: " + str(i+1), {'fontsize': 28,
                                 'fontweight' : 'bold',
                                 'verticalalignment': 'baseline',
                                 'horizontalalignment': 'center'})
    plt.axis('off')
    plt.show()
