#lab4_zad3
import skimage.segmentation as seg
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage import data
from skimage.color import rgb2gray
img = plt.imread('cat.jpg')

s = np.linspace(0, 2*np.pi, 400)
s1 = s
x = 470 + 315*np.cos(s)
y = 150 + 149*np.sin(s)
x1 = 464 + 315*np.cos(s)
y1 = 135 + 149*np.sin(s)
init = np.array([x, y]).T
init2 = np.array([x1,y1]).T
x =455 + 315*np.cos(s)
y =156 + 149*np.sin(s)
x1 =470 + 318*np.cos(s)
y1=150 + 153*np.sin(s)
init3 =np.array([x, y]).T
init4 = np.array([x1, y1]).T
snake = seg.active_contour(gaussian(rgb2gray(img), 3), init, alpha=0.015, beta=0.1, gamma =0.001,w_edge=1.07, w_line = 0.60,  max_iterations = 5000, convergence = 1, bc='periodic') 
snek  = seg.active_contour(gaussian(rgb2gray(img), 3), init2, alpha=0.015, beta=0.1, gamma =0.001,w_edge=1.07, w_line = 0.60,  max_iterations = 5000, convergence = 1, bc='periodic')
snek1 = seg.active_contour(gaussian(rgb2gray(img), 3), init3, alpha=0.015, beta=0.1, gamma =0.001,w_edge=1.07, w_line = 0.60,  max_iterations = 5000, convergence = 1, bc='periodic') 
snek2  = seg.active_contour(gaussian(rgb2gray(img), 3), init4, alpha=0.015, beta=0.1, gamma =0.001,w_edge=1.07, w_line = 0.60,  max_iterations = 5000, convergence = 1, bc='periodic') 
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(7, 7))
ax[0][0].imshow(img)
ax[0][0].plot(init[:, 0], init[:, 1], '--r', lw=1)
ax[0][0].plot(snake[:, 0], snake[:, 1], '-b', lw=2)
ax[0][0].set_xticks([]), ax[0][0].set_yticks([])
ax[0][0].axis([0, img.shape[1], img.shape[0], 0])
ax[0][1].imshow(img)
ax[0][1].plot(init2[:, 0], init2[:, 1], '--r', lw=1)
ax[0][1].plot(snek[:, 0], snek[:, 1], '-b', lw=2)
ax[0][1].set_xticks([]), ax[0][1].set_yticks([])
ax[0][1].axis([0, img.shape[1], img.shape[0], 0])
ax[1][0].imshow(img)
ax[1][0].plot(init3[:, 0], init3[:, 1], '--r', lw=1)
ax[1][0].plot(snek1[:, 0], snek1[:, 1], '-b', lw=2)
ax[1][0].set_xticks([]), ax[1][0].set_yticks([])
ax[1][0].axis([0, img.shape[1], img.shape[0], 0])
ax[1][1].imshow(img)
ax[1][1].plot(init4[:, 0], init4[:, 1], '--r', lw=1)
ax[1][1].plot(snek2[:, 0], snek2[:, 1], '-b', lw=2)
ax[1][1].set_xticks([]), ax[1][1].set_yticks([])
ax[1][1].axis([0, img.shape[1], img.shape[0], 0])
plt.show()
