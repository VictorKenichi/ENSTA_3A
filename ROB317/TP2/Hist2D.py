import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("./Vidéos/Extrait1-Cosmos_Laundromat1(340p).m4v")

index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
yuv = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
hist = cv2.calcHist([yuv], [1,2], None, [256,256], [0,256,0,256])
hist_256 = (256*(hist-np.amin(hist))/(np.amax(hist)-np.amin(hist))).astype('uint8')

while(ret):
    index += 1
    yuv = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
    hist = cv2.calcHist([yuv], [1,2], None, [256,256], [0,256,0,256])
    hist_256 = (256*(hist-np.amin(hist))/(np.amax(hist)-np.amin(hist))).astype('uint8')
    plt.figure(1)
    cv2.imshow('Image et Champ de vitesses (Farnebäck)',frame2)
    plt.figure(2)
    cv2.imshow('Histogramme 2d de (u,v) ',hist_256)
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)
        cv2.imwrite('OF_hsv_%04d.png'%index,yuv)
    prvs = next
    ret, frame2 = cap.read()
    if (ret):
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

cap.release()
cv2.destroyAllWindows()
