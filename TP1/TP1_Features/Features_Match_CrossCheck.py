# coding: utf-8

import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import sys

def rotate(pt,degree,anchor):
    ptr = np.array([0,0])
    ptr[0] = anchor[0] + math.cos(degree) * (pt[0] - anchor[0]) + math.sin(degree) * (pt[1] - anchor[1])
    ptr[1] = anchor[1] - math.sin(degree) * (pt[0] - anchor[0]) + math.cos(degree) * (pt[1] - anchor[1])
    return ptr

if len(sys.argv) != 2:
    print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
    sys.exit(2)
if sys.argv[1].lower() == "orb":
    detector = 1
elif sys.argv[1].lower() == "kaze":
    detector = 2
else:
    print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
    sys.exit(2)

#Paramètres de rotation
angle = 180 #degrés
tol = 3 #tolérance

#Lecture de la paire d'images
img1 = cv2.imread('../Image_Pairs/torb_small1.png')
print("Dimension de l'image 1 :",img1.shape[0],"lignes x",img1.shape[1],"colonnes")
print("Type de l'image 1 :",img1.dtype)
#img2 = cv2.imread('../Image_Pairs/torb_small2.png')
#print("Dimension de l'image 2 :",img2.shape[0],"lignes x",img2.shape[1],"colonnes")
#print("Type de l'image 2 :",img2.dtype)

#Rotation
M = cv2.getRotationMatrix2D((img1.shape[0]/2, img1.shape[1]/2), angle, 1)
img2 = cv2.warpAffine(img1,M,(img1.shape[1],img1.shape[0]))
anchor = [img1.shape[0]/2,img1.shape[1]/2]

#Début du calcul
t1 = cv2.getTickCount()
#Création des objets "keypoints"
if detector == 1:
    kp1 = cv2.ORB_create(nfeatures = 500,#Par défaut : 500
                        scaleFactor = 1.2,#Par défaut : 1.2
                        nlevels = 8)#Par défaut : 8
    kp2 = cv2.ORB_create(nfeatures=500,
                        scaleFactor = 1.2,
                        nlevels = 8)
    print("Détecteur : ORB")
else:
    kp1 = cv2.KAZE_create(upright = False,#Par défaut : false
                        threshold = 0.001,#Par défaut : 0.001
                        nOctaves = 4,#Par défaut : 4
                        nOctaveLayers = 4,#Par défaut : 4
                        diffusivity = 2)#Par défaut : 2
    kp2 = cv2.KAZE_create(upright = False,#Par défaut : false
                        threshold = 0.001,#Par défaut : 0.001
                        nOctaves = 4,#Par défaut : 4
                        nOctaveLayers = 4,#Par défaut : 4
                        diffusivity = 2)#Par défaut : 2
    print("Détecteur : KAZE")
#Conversion en niveau de gris
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Détection et description des keypoints
pts1, desc1 = kp1.detectAndCompute(gray1,None)
pts2, desc2 = kp2.detectAndCompute(gray2,None)
#Les points non appariés apparaîtront en gris
img1 = cv2.drawKeypoints(gray1, pts1, None, color=(127,127,127), flags=0)
img2 = cv2.drawKeypoints(gray2, pts2, None, color=(127,127,127), flags=0)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection points et calcul descripteurs :",time,"s")
# Calcul de l'appariement
t1 = cv2.getTickCount()
if detector == 1:
    #Distance de Hamming pour descripteur BRIEF (ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
else:
    #Distance L2 pour descripteur M-SURF (KAZE)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(desc1,desc2)
# Tri des appariemements
matches = sorted(matches, key = lambda x:x.distance)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Calcul de l'appariement :",time,"s")

# Trace les N meilleurs appariements
Nbest = 200
img3 = cv2.drawMatches(img1,pts1,img2,pts2,matches[:Nbest],None,flags=2)
plt.imshow(img3)
plt.title('%i meilleurs appariements'%Nbest)
plt.show()

#Évaluation quantitative
countCorrect = 0
countAll = 0
for m in matches:
    img1 = m.queryIdx
    img2 = m.trainIdx
    (x1,y1) = pts1[img1].pt
    (x2,y2) = pts2[img2].pt
    (xr,yr) = rotate((x1,y1),math.radians(angle),anchor)
    distr = math.sqrt((xr-x2)**2+(yr-y2)**2)
    if distr<tol:
        countCorrect += 1;
    countAll += 1;
print(f'''Exactitude de l'appariement : {100*countCorrect/countAll}%''')
