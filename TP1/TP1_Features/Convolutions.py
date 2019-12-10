# coding: utf-8

import numpy as np
import cv2
import math

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

#Méthode directe
t1 = cv2.getTickCount()
img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
    for x in range(1,w-1):
        val = 5*img[y, x] - img[y-1, x] - img[y, x-1] - img[y+1, x] - img[y, x+1]
        img2[y,x] = min(max(val,0),255)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe :",time,"s")

plt.figure(1)
plt.subplot(121)
plt.imshow(img2,cmap = 'gray')
plt.title('Convolution - Méthode Directe')

#Méthode filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
img3 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")

plt.subplot(122)
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Convolution - filter2D')

#Méthode directe x
t1 = cv2.getTickCount()
img4 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
    for x in range(1,w-1):
        valx = -img[y-1,x+1]+img[y-1,x-1]-2*img[y,x+1]+2*img[y,x-1]-img[y+1,x+1]+img[y+1,x-1]
        img4[y,x] = valx
img4 = 255*(img4-np.amin(img4))/(np.amax(img4)-np.amin(img4))
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe x :",time,"s")

plt.figure(2)
plt.subplot(121)
plt.imshow(img4,cmap = 'gray')
plt.title('Convolution x - Méthode Directe')

#Méthode filter2D x
t1 = cv2.getTickCount()
kernelx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
kernelx = np.flip(kernelx,1)
img5 = cv2.filter2D(img,-1,kernelx)
img5 = 255*(img5-np.amin(img5))/(np.amax(img5)-np.amin(img5))
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D x :",time,"s")

plt.subplot(122)
plt.imshow(img5,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Convolution x - filter2D')

#Méthode directe gradient
t1 = cv2.getTickCount()
img6 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
    for x in range(1,w-1):
        valx = img[y-1,x+1]-img[y-1,x-1]+2*img[y,x+1]-2*img[y,x-1]+img[y+1,x+1]-img[y+1,x-1]
        valy = img[y+1,x-1]-img[y-1,x-1]+2*img[y+1,x]-2*img[y-1,x]+img[y+1,x+1]-img[y-1,x+1]
        val = np.sqrt(valx**2+valy**2)
        img6[y,x] = val
img4 = 255*(img6)/(np.amax(img6))
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe Gradient :",time,"s")

plt.figure(3)
plt.subplot(121)
plt.imshow(img6,cmap = 'gray')
plt.title('Convolution Gradient - Méthode Directe')

#Méthode filter2D gradient
t1 = cv2.getTickCount()
kernelx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
kernely = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
kernelx = np.flip(kernelx,1)
kernely = np.flip(kernely,0)
imgx = cv2.filter2D(img,-1,kernelx)
imgy = cv2.filter2D(img,-1,kernely)
img7 = np.sqrt(np.square(imgx)+np.square(imgy))
img7 = 255*(img7)/(np.amax(img7))
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D Gradient :",time,"s")

plt.subplot(122)
plt.imshow(img7,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Convolution Gradient - filter2D')

plt.show()
