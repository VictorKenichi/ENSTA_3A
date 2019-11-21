import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from argparse import ArgumentParser

#bin = 32 # nombre de bins
r = 2
q = 2
tol = 0.5

# module pour utiliser une ligne pour taper les arguments d'un fichier sur le terminal
parser = ArgumentParser()
parser.add_argument(dest="video", type=int, help="video d'entrée")
input_args = parser.parse_args()
video = int(input_args.video)

if video == 1:
    cap = cv2.VideoCapture("./Vidéos/Extrait1-Cosmos_Laundromat1(340p).m4v")
    montageTest = pd.read_csv("./Montage/Montage_1.csv", index_col=0)
elif video == 2:
    cap = cv2.VideoCapture("./Vidéos/Extrait2-ManWithAMovieCamera(216p).m4v")
    montageTest = pd.read_csv("./Montage/Montage_2.csv", index_col=0)
elif video == 3:
    cap = cv2.VideoCapture("./Vidéos/Extrait3-Vertigo-Dream_Scene(320p).m4v")
    montageTest = pd.read_csv("./Montage/Montage_3.csv", index_col=0)
elif video == 4:
    cap = cv2.VideoCapture("./Vidéos/Extrait4-Entracte-Poursuite_Corbillard(358p).m4v")
    montageTest = pd.read_csv("./Montage/Montage_4.csv", index_col=0)
elif video == 5:
    cap = cv2.VideoCapture("./Vidéos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
    montageTest = pd.read_csv("./Montage/Montage_5.csv", index_col=0)
else:
    cap = cv2.VideoCapture(0)
    montageTest = pd.read_csv("./Montage/Montage_0.csv", index_col=0)

# Paramètres du détecteur de points d'intérêt
feature_params = dict( maxCorners = 10000,
                       qualityLevel = 0.01,
                       minDistance = 5,
                       blockSize = 7 )

# Paramètres de l'algo de Lucas et Kanade
lk_params = dict( winSize  = (15,15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Extraction image initiale et détection des points d'intérêt
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
index = 1

h = old_gray.shape[0]
w = old_gray.shape[1]

while(ret):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcul du flot optique
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)


    # Sélection des points valides
    good_new = p1[st==1]
    good_old = p0[st==1]

    disp = p1 - p0

    hist = cv2.calcHist([disp], [0,1], None, [h/r,w/r], [-h/q,h/q,-w/q,w/q])
    cv2.imshow('Histogram',hist)#/(h*w))

    # Image masque pour tracer les vecteurs de flot
    mask = np.zeros_like(old_frame)

    # Affichage des vecteurs de flot
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d),(255,255,0),2)
        frame = cv2.circle(frame,(c,d),3,(255,255,0),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('Flot Optique Lucas-Kanade Pyramidal',img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('OF_PyrLk%04d.png'%index,img)

    # Mis à jour image et détection des nouveaux points
    p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    old_gray = frame_gray.copy()
    index += 1

cv2.destroyAllWindows()
cap.release()
