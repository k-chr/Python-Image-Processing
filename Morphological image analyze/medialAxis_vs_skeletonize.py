#lab6 zad1
import skimage as sk
from skimage import feature
import scipy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import disk, star, diamond, medial_axis,octagon, rectangle, square, erosion, dilation, skeletonize

img = sk.io.imread("bw.png", True)
img2 = 255*sk.io.imread("psio.png", True)
img3 = sk.io.imread("horse.png", True)
img_ = 255-np.array([[255 if img[i][j] > 100 else 0 for j in range(0,img.shape[1])]for i in range(0, img.shape[0])])
img2_ = 255-np.array([[255 if img2[i][j] > 100 else 0 for j in range(0,img2.shape[1])]for i in range(0, img2.shape[0])])
img3_ = 255 - np.array([[255 if img3[i][j] > 127 else 0 for j in range(0,img3.shape[1])]for i in range(0, img3.shape[0])])
##out = [np.vstack([img, 255-img_]),np.vstack([img2, 255-img2_]),np.vstack([img3, 255-img3_])]
img_sk = 255 - 255*skeletonize(img_/255)
plt.imsave("img1_sk.png", img_sk, cmap = 'gray')

img_mat = 255 - 255*medial_axis(img_/255)
plt.imsave("img1_mat.png", img_mat, cmap = 'gray')
cv2.imwrite("stack1.png", np.hstack([img, img_sk, img_mat]))
img_sk = 255 - 255*skeletonize(img2_/255)
plt.imsave("img2_sk.png", img_sk, cmap = 'gray')

img_mat = 255 - 255*medial_axis(img2_//255)
plt.imsave("img2_mat.png", img_mat, cmap = 'gray')
cv2.imwrite("stack2.png", np.hstack([img2, img_sk, img_mat]))
img_sk = 255 - 255*skeletonize(img3_/255)
plt.imsave("img3_sk.png", img_sk, cmap = 'gray')

img_mat = 255 - 255*medial_axis(img3_/255)
plt.imsave("img3_mat.png", img_mat, cmap = 'gray')
cv2.imwrite("stack3.png", np.hstack([img3, img_sk, img_mat]))
#out_ = np.hstack([out[0], np.vstack([img_sk, img_mat])])
##plt.imshow(out_, cmap = "gray")
##plt.axis('off')
##plt.show()
##plt.imshow(out[1], cmap = "gray")
##plt.axis('off')
##plt.show()
##plt.imshow(out[2], cmap = "gray")
##plt.axis('off')
##plt.show()
