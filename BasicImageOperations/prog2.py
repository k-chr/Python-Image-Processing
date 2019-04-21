import PySimpleGUI as sg      
import sys
import skimage as sk
from skimage import feature, color
import numpy as np
import cv2
from queue import *
def lowpass_filter(img:int, choice) -> int:
    temp_img = (img.copy())
    if choice == 1:
         temp_img = np.array((255*sk.filters.gaussian(temp_img, sigma = 1, multichannel=True)).astype(np.uint8)) #gaussian filter
         #gaussian filter
         #temp_img[:,:,2] *= 255
    elif choice == 2:
        if len(img.shape) > 2:
            for i in range(3):
                temp_img[:,:,i] = np.array(sk.filters.rank.mean(np.array(temp_img[:,:,i].copy()), np.array([[1,2,1], [2,4,2], [1,2,1]])))
                temp_img[:,:,i] = np.array(temp_img[:,:,i].astype(np.uint8), dtype=np.uint8)#mean filter
        else:
            temp_img = np.array(sk.filters.rank.mean(np.array(img), np.array([[1,1,1], [1,1,1], [1,1,1]]), mask = np.array([[1,2,1], [2,4,2], [1,2,1]]))).astype(np.uint8)
    elif choice == 3:
         if len(img.shape) > 2:
            for i in range(3):
                tmp = (np.array(sk.filters.median(temp_img[:,:,i].copy()), dtype=np.uint8).astype(np.uint8)) #median filter
       
                temp_img[:,:,i]  = tmp
         else:
            temp_img = np.array(sk.filters.median(img)) #median filter
    else: 
        if len(img.shape) > 2:
            for i in range(3):
                temp_img[:,:,i] = (np.array(sk.filters.rank.mean_bilateral(np.array(temp_img[:,:,i].copy()), np.array([[1,2,1], [2,4,2], [1,2,1]]), s0 = 100, s1 = 100)).astype(np.uint8)) #bilateral_mean filter
        else:
            temp_img = np.array(sk.filters.rank.mean_bilateral(np.array(img), np.array([[1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]]))).astype(np.uint8)
    sk.io.imsave('processedLena.png', temp_img)
    
    return temp_img
def s_and_p(layer):
    noise = layer.copy()
    for i in range(int(0.01 * noise.shape[0] * noise.shape[1])):
        j = np.random.randint(0, noise.shape[0])
        k = np.random.randint(0, noise.shape[1])
        if np.random.randint(50)%2 == 0:
            noise[j][k] = 0
        else:
            noise[j][k] = 255
    return noise
def FUN(layer, choice):
    FUNFUN = {1:np.random.normal(0.0, np.std(layer)**1.5, layer.shape)}
    return FUNFUN[choice]
def add_noise(img, choice):
    layer = img.copy()[:,:,2]
    temp_img = img.copy()
    if choice == 2:
        egami = s_and_p(temp_img)
        sk.io.imsave('processedLena.png', egami)
        return egami
    elif choice == 1:
        temp_img = color.rgb2hsv(img)
        layer = temp_img[:,:,2]
        noise = FUN(layer,choice)
        temp_img[:,:,2] = (temp_img[:,:,2] + noise)
        egami = np.clip(color.hsv2rgb(temp_img), 0, 1.0)
        sk.io.imsave('processedLena.png', egami)
        return egami

def deepFry(image):
    egami = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    temp_img = egami.copy()
    temp_img[:,:,0] += (np.std(img[:,:,0])**2).astype(np.uint8)+79
    temp_img[:,:,1] += ((np.std(img[:,:,2]))**0.2).astype(np.uint8)*300
    temp_img[:,:,2] += (np.std(img[:,:,2]**900).astype(np.uint8))*100
    egami = cv2.cvtColor(temp_img, cv2.COLOR_HSV2RGB)
    sk.io.imsave('processedLena.png', egami)
    return egami
def channel_adjust(channel, values):
    orig_size = channel.shape
    flat_channel = channel.flatten()
    adjusted = np.interp(flat_channel, np.linspace(0, 1, len(values)), values)
    return adjusted.reshape(orig_size)
def gothamFilter(image):
    image = image/255
    r = image[:, :, 0]
    b = image[:, :, 2]
    r_boost_lower = channel_adjust(r, [
        0, 0.05, 0.1, 0.2, 0.3,
        0.5, 0.7, 0.8, 0.9,
        0.95, 1.0])
    b_more = np.clip(b + 0.03, 0, 1.0)
    merged = np.stack([r_boost_lower, image[:, :, 1], b_more], axis=2)
    blurred = sk.filters.gaussian(merged, sigma=10, multichannel=True)
    final = np.clip(merged * 1.3 - blurred * 0.3, 0, 1.0)
    b = final[:, :, 2]
    b_adjusted = channel_adjust(b, [
        0, 0.047, 0.118, 0.251, 0.318,
        0.392, 0.42, 0.439, 0.475,
        0.561, 0.58, 0.627, 0.671,
        0.733, 0.847, 0.925, 1])
    final[:, :, 2] = b_adjusted
    lanif = (255*final).astype(int)
    print(lanif)
    sk.io.imsave('processedLena.png', lanif)
    return lanif
def getChannel(image, channel):
    if(channel == 'red'):
        chImg = color.gray2rgb(split_image_into_channels(image)[0])
        chImg[:,:,1] = 0
        chImg[:,:,2] = 0
    elif(channel == 'blue'):
        chImg = color.gray2rgb(split_image_into_channels(image)[2])
        chImg[:,:,1] = 0
        chImg[:,:,0] = 0
    elif(channel == 'green'):
        chImg = color.gray2rgb(split_image_into_channels(image)[1])
        chImg[:,:,2] = 0
        chImg[:,:,0] = 0
    else:
        return None
    sk.io.imsave('processedLena.png', chImg)
    return chImg
def split_image_into_channels(image):
    """Look at each image separately"""
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    return red_channel, green_channel, blue_channel
def getEdgy(type_, image):
    """Get edges using basic edge-detecting algorithms"""
    fun_dict = {"canny" : feature.canny,
                "prewitt" : sk.filters.prewitt,
                "sobel" : sk.filters.sobel}
    egami = np.array(color.gray2rgb((fun_dict[type_](color.rgb2gray(image))*255).astype(np.uint8)))
    sk.io.imsave('processedLena.png', egami)
    return egami

def sharpen(image, a, b):
    """Sharpening an image: Blur and then subtract from original"""
    blurred = sk.filters.gaussian(image, sigma=3, multichannel=True)
    sharper = (np.clip(image/255 * a - blurred * b, 0, 1.0)* 255).astype(np.uint8)
    sk.io.imsave('processedLena.png', sharper)
    return sharper
def grayScale(image):
    egami = color.gray2rgb(color.rgb2gray(image))   
    sk.io.imsave('processedLena.png', (255*egami).astype(np.uint8))
    return (255*egami).astype(np.uint8)

img = sk.io.imread("lena.png")

uq = LifoQueue(50)
rq = LifoQueue(50)
# Layout the design of the GUI      
layout = [
          [sg.Text('Modify image, clicking buttons next to the image',
           auto_size_text=True)],
          [sg.Image(filename="lena.png", key='img'),sg.Column([[sg.Button('canny')],
           [sg.Button('prewitt')],
           [sg.Button('sobel')],
           [sg.Button('grayscale')],
           [sg.Button('sharpen')],
           [sg.Button('red channel')],
           [sg.Button('blue channel')],
           [sg.Button('green channel')],
           [sg.Button('deep fry Lena')]
           ]), sg.Column([[sg.Button('gaussian noise')],
                        [sg.Button('salt and pepper noise')],
                        [sg.Button('gaussian filter')],
                        [sg.Button('mean filter')],
                        [sg.Button('median filter')],
                        [sg.Button('bilateral filter')],
                        [sg.Button('Gotham filter')],
                          ])],
          [sg.Button('UNDO'),sg.Button('REDO')],
          [sg.Button('discard changes')],
           [sg.Quit()]
          ]
               
# Show the Window to the user    
window = sg.Window('Image processing example').Layout(layout)      
currImg = img.copy()
# Event loop. Read buttons, make callbacks      
while True:      
    # Read the Window    
    event, value = window.Read()      
    # Take appropriate action based on button
    
    if event == 'canny':
        rq = LifoQueue(50)
        uq.put(currImg)
        currImg = getEdgy("canny", currImg)
    elif event == 'deep fry Lena':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = deepFry(currImg)
    elif event == 'gaussian filter':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = lowpass_filter(currImg, 1)
    elif event == 'median filter':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = lowpass_filter(currImg, 3)
    elif event == 'mean filter':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = lowpass_filter(currImg, 2)
    elif event == 'bilateral filter':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = lowpass_filter(currImg, 4)
    elif event == 'gaussian noise':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = add_noise(currImg,1)
    elif event == 'salt and pepper noise':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = add_noise(currImg,2)
    elif event == 'sobel':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = getEdgy("sobel", currImg)
    elif event == 'prewitt':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = getEdgy("prewitt", currImg)
    elif event == 'sharpen':
        uq.put(currImg)
        currImg = sharpen(currImg, 1.3, 0.3)
    elif event == 'blue channel':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = getChannel(currImg, "blue")
    elif event == 'red channel':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = getChannel(currImg, "red")
    elif event == 'green channel':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = getChannel(currImg, "green")
    elif event == 'grayscale':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = grayScale(currImg)
    elif event == 'Gotham filter':
        uq.put(currImg)
        rq = LifoQueue(50)
        currImg = gothamFilter(currImg)
    elif event == "UNDO":
        if(uq.qsize() > 0):
            rq.put(currImg)
            currImg = uq.get()
            sk.io.imsave('processedLena.png',currImg)
    elif event == "REDO":
        if(rq.qsize() > 0):
            uq.put(currImg)
            currImg = rq.get()
            sk.io.imsave('processedLena.png',currImg)
    elif event == 'discard changes':
        sk.io.imsave('processedLena.png', img)
        currImg = img.copy()
        uq = LifoQueue(50)
        rq = LifoQueue(50)
    elif event =='Quit'  or event is None:  
        window.Close()    
        break
    window.FindElement('img').Update(filename = "processedLena.png")

# All done!      
sg.PopupOK('Done')    
