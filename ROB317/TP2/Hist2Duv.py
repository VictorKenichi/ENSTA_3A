import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from argparse import ArgumentParser

# Paramètres de l'algorithem
bin = 32 # nombre de bins
tol = 0.2 # tolerence

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

cutTest = montageTest["Raccord"].to_numpy()
cutHistUV = np.zeros_like(cutTest)

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
    if(ret):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        hist = cv2.calcHist([yuv], [1,2], None, [bin,bin], [0,256,0,256])
        hTest = cv2.compareHist(hist_old,hist,0)
        if hTest<1-tol:
            cut += 1
            cutHistUV[index] = 1
            print(f'''index = {index}''')
            print(f'''Correlation = {hTest}''')
        index += 1

cf = confusion_matrix(cutTest,cutHistUV)
print(f'''Nombre des raccords : {cut}''')
print('Matrice de confusion:')
print(pd.DataFrame(cf))
print(f'''Accuracy : {(100*cf[0][0]+cf[1][1])/(cf[0][0]+cf[1][0]+cf[0][1]+cf[1][1])}%''')
print(f'''Precision : {100*cf[1][1]/(cf[0][1]+cf[1][1])}%''')
print(f'''Recall : {100*cf[1][1]/(cf[1][0]+cf[1][1])}%''')


cap.release()
cv2.destroyAllWindows()
