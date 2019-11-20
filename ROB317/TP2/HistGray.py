import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from argparse import ArgumentParser

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
mse = 0
cut = 0
ret, frame = cap.read()
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [256], [0,256])
hist = hist/256**2

# Paramètres de l'algorithem
kernel = 3 # voisinage consideré
tol = 0.05 # tolerance

fig, ax = plt.subplots()
ax.set_title('Histogram 1D du Niveau de Gris')
ax.set_xlabel('Niveau de Gris')
ax.set_ylabel('Frequence')
lineGray, = ax.plot(np.arange(256), np.zeros((256,)), c='k', lw=3)
ax.set_xlim(0, 255)
ax.set_ylim(0, 1)
plt.ion()
plt.show()

while(ret):
    plt.figure(1)
    cv2.imshow('Image noir et blanc',gray)
    lineGray.set_ydata(hist)
    fig.canvas.draw()
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame)
        cv2.imwrite('OF_gray_%04d.png'%index,gray)
    hist_old = hist.copy()
    ret, frame = cap.read()
    if(ret):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = hist/256**2
        mse = 0
        for i in range(kernel//2,256-kernel//2):
            if hist[i][0] != 0:
                diff_min = (hist[i][0]-hist_old[i][0])**2
                for i2 in range(-kernel//2+1,kernel//2):
                    diff = (hist[i][0] - hist_old[i+i2][0])**2
                    if diff < diff_min:
                        diff_min = diff
                mse += diff_min
        if mse>tol:
            cut += 1
            cutHistUV[index] = 1
            print(f'''index = {index}''')
            print(f'''mse = {mse}''')
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
