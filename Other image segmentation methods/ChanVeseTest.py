#lab4_zad4
from skimage.segmentation import chan_vese
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import skimage as sk
def FUN(layer, choice, i):
    FUNFUN = {1:np.random.normal(0.0, i, layer.shape),
              2:np.random.uniform(0.0,i, layer.shape),
              3:np.random.poisson(i, layer.shape)}
    return FUNFUN[choice]
def add_noise(img, choice, i):
    layer = img.copy()
    temp_img = img.copy()
    layer = img.copy()
    noise = FUN(layer,choice, i)
    layer = layer + noise
    return layer
img1 = sk.io.imread('objects1.jpg', True)
img2 = sk.io.imread('objects2.jpg', True)
img3 = sk.io.imread('objects3.jpg', True)
img = [img1, img2, img3]
for n in range(0,3):
    for k in range(1,4):
        for j in [0.33, 0.44, 0.55, 0.77]:
            noisy = add_noise(img[2], k, j)
            for i in range (400, 501, 100):
                
                out = chan_vese(255-noisy, max_iter = i)
                d = [noisy, out]
                arr = np.hstack(d)
                plt.imshow(sk.color.gray2rgb(arr))
                plt.axis('off')
                l = "objects1" if n==0 else "objects2" if n ==1 else "objects3" 
                st = "gauss" if k == 1 else "poisson" if k == 3 else "uniform"
                plt.title("iteration_limit: " + str(i) + ", for img: " + l + ", with noise: " + st + ", and parameter = " +str(j), fontdict = {'fontsize': 28, 'fontweight' : 'bold', 'verticalalignment': 'baseline', 'horizontalalignment': "center"})
                plt.show()
