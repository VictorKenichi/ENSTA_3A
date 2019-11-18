import cv2
import numpy as np
from matplotlib import pyplot as plt

# Paramètres de l'algorithem
bin = 32 # nombre de bins
tol = 0.2 # tolerence

#cap = cv2.VideoCapture("./Vidéos/Extrait1-Cosmos_Laundromat1(340p).m4v")
cap = cv2.VideoCapture("./Vidéos/Extrait3-Vertigo-Dream_Scene(320p).m4v")
#cap = cv2.VideoCapture("./Vidéos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")

index = 1
cut = 0
nTicks = 4
ret, frame = cap.read()
yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
hist = cv2.calcHist([yuv], [1,2], None, [bin,bin], [0,256,0,256])

plot = 0

if plot == 1:
    fig, ax = plt.subplots()
    ln, = plt.plot([],[],'ro')
    ax.set_title('Histogram 2D de (u,v)')
    ax.set_xlabel('v')
    ax.set_ylabel('u')
    ax.set_xlim(0, bin-1)
    ax.set_ylim(0, bin-1)
    plt.xticks(np.arange(0,bin,bin/nTicks),np.arange(0,256,256//nTicks))
    plt.yticks(np.arange(0,bin,bin/nTicks),np.arange(0,256,256//nTicks))
    plt.ion()
    plt.show()

while(ret):
    cv2.imshow('Image et Champ de vitesses (Farnebäck)',frame)
    if plot==1:
        plt.imshow(hist/256**2, cmap='gray',interpolation='None')
    else:
        cv2.imshow('Histogram 2D de (u,v)',hist/256**2)
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame)
        cv2.imwrite('OF_Yuv_%04d.png'%index,yuv)
    hist_old = hist.copy()
    ret, frame = cap.read()
    index += 1
    if(ret):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        hist = cv2.calcHist([yuv], [1,2], None, [bin,bin], [0,256,0,256])
        hTest = cv2.compareHist(hist_old,hist,0)
        if hTest<1-tol:
            cut += 1
            print(f'''index = {index}''')
            print(f'''Correlation = {hTest}''')
print(f'''Nombre des raccords : {cut}''')
cap.release()
cv2.destroyAllWindows()
