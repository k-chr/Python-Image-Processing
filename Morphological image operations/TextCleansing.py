#lab5 zad2
import skimage as sk
from skimage import feature
import scipy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import disk, star,opening, closing, diamond, octagon, rectangle, square, ball, erosion, dilation, binary_opening, binary_closing

img = 255 - sk.io.imread("bw2.bmp", True)


##img_e = erosion(img.copy(), rectangle(7,4))
##img_d = dilation(img_e.copy(), octagon(10,2))
##img_d = erosion(img_d.copy(), disk(3))
##sk.io.imshow(255-img, cmap = "gray")
##sk.io.show()
##img_d = 255 - np.array([[255 if img_d[i][j] > 140 else 0 for j in range(img.shape[1])] for i in range(img.shape[0])])
##sk.io.imshow(img_d, cmap = "gray")
##sk.io.show()
img_e = erosion(img.copy(), rectangle(7,4))
img_d = dilation(img_e.copy(), octagon(10,2))
img_d = erosion(img_d.copy(), disk(3))
img_d = 255 - np.array([[255 if img_d[i][j] > 140 else 0 for j in range(img.shape[1])] for i in range(img.shape[0])])
out = np.hstack([255-img, img_d])

plt.imshow(out, cmap = "gray")
plt.title("Negacja->Erozja(prostokąt(7x4))->dylacja(ośmiokąt(10x2))->\nerozja(koło(3))->progowanie->Negacja", {'fontsize': 18,'fontweight' : 'bold', 'verticalalignment': 'baseline','horizontalalignment': 'center'})
plt.axis("off")
plt.show()
img1 = binary_closing(img, rectangle(2,3))
img1 = binary_opening(img1, disk(3))
img1 = dilation(img1, disk(2))
img1 = 255 - 255*erosion(img1, disk(2))

out = np.hstack([255-img, 255 - 255*img1])

plt.imshow(out, cmap = "gray")
plt.title("Negacja->Otwarcie(prostokąt(2x3))->zamknięcie(koło(3))->\ndylacja(koło(2))->erozja(koło(2)->Negacja", {'fontsize': 18,'fontweight' : 'bold', 'verticalalignment': 'baseline','horizontalalignment': 'center'})
plt.axis("off")
plt.show()
