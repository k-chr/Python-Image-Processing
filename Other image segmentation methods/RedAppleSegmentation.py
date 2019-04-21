#lab4 zad2
import skimage as sk
from skimage import feature
import scipy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
img = sk.io.imread("apples.jpg")
green = img[:,:,1]
copy = green.copy()
copy = np.array([[255 if copy[i][j] < 20 else 100 if copy[i][j] == 255 else 0 for j in range(img.shape[1])] for i in range(img.shape[0])])
##for i in range(img.shape[0]):
##    for j in range(img.shape[1]):
list1 = []
list1.append(green)
list1.append(copy)
sk.io.imshow(copy, cmap = 'gray')
sk.io.show()
new = sk.segmentation.random_walker(green.copy(), copy, beta = 20)
new *=255
new-=255
#new = np.array([[0 if copy[i][j] < 105 else 255 for j in range(new.shape[1])] for i in range(new.shape[0])])
list1.append(new)
out = np.hstack(list1)
sk.io.imshow(out, cmap = 'gray')
sk.io.show()
