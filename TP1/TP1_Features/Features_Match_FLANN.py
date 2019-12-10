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
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection points et calcul descripteurs :",time,"s")
# Calcul de l'appariement
t1 = cv2.getTickCount()
if detector == 1:
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    search_params = dict(checks=50)
else:
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(desc1,desc2,k=2)
# Application du ratio test
good = []
for m,n in matches:
  if m.distance < 0.7*n.distance:
    good.append([m])
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Calcul de l'appariement :",time,"s")

# Affichage
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = 0)

# Affichage des appariements qui respectent le ratio test
img3 = cv2.drawMatchesKnn(gray1,pts1,gray2,pts2,good,None,**draw_params)

Nb_ok = len(good)
plt.imshow(img3)
plt.title('%i appariements OK'%Nb_ok)
plt.show()


#Évaluation quantitative
countCorrect = 0
countAll = 0
for m,n in matches:
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
