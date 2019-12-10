# coding: utf-8

import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#Début du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
# Mettre ici le calcul de la fonction d'intérêt de Harris
alpha = 0.06 # ou 0.04
d_maxloc = 9 # nombre of pixels de la coté du noyau
BlurImg = cv2.GaussianBlur(img,(d_maxloc,d_maxloc),0,0,cv2.BORDER_REPLICATE)
kernelx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]) # filter de Sobel horizontal
kernely = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]) # filter de Sobel vertical
kernelx = np.flip(kernelx,1)
kernely = np.flip(kernely,0)
imgx = cv2.filter2D(BlurImg,-1,kernelx) # Ix Dérivé partielle horizontale
imgy = cv2.filter2D(BlurImg,-1,kernely) # Iy Dérivé partielle verticale
Harris = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(0,h):
    for x in range(0,w):
        Autocr = np.zeros(shape=(h,w,2,2))
        for j in range(-1,1):
            for i in range(-1,1):
                Autocr[y,x,0,0] += imgx[y+j, x+i]*imgx[y+j, x+i]
                Autocr[y,x,1,0] += imgx[y+j, x+i]*imgy[y+j, x+i]
                Autocr[y,x,0,1] += imgx[y+j, x+i]*imgy[y+j, x+i]
                Autocr[y,x,1,1] += imgy[y+j, x+i]*imgy[y+j, x+i]
        Harris[y,x] = np.linalg.det(Autocr[y,x]) - alpha*np.trace(Autocr[y,x])
for y in range(0,h):
	for x in range(0,w):
		Theta[y,x] = min(max((Harris[y,x] - np.min(Harris))/(np.max(Harris) - np.min(Harris)),0),255)
# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
seuil_relatif = 0.01
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)
#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0
#On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Mon calcul des points de Harris : ",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

plt.figure(1)
plt.subplot(121)
plt.imshow(Theta,cmap = 'gray')
plt.title('Theta')

plt.subplot(122)
plt.imshow(Theta_dil,cmap = 'gray')
plt.title('Theta_dil')

plt.figure(2)
plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Relecture image pour affichage couleur
Img_pts=cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

plt.show()
