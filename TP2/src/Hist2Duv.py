import cv2
import numpy  as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from argparse        import ArgumentParser

# Paramètres de l'algorithem
bin = 128 # nombre de bins

# module pour utiliser une ligne pour taper les arguments d'un fichier sur le terminal
parser = ArgumentParser()
parser.add_argument(dest="video", type=int, help="video d'entrée")
input_args = parser.parse_args()
video = int(input_args.video)

if video == 1:
    cap = cv2.VideoCapture("../Vidéos/Extrait1-Cosmos_Laundromat1(340p).m4v")
    montageTest = pd.read_csv("../Validation/Montage_1.csv", index_col=0)
    tol = 0.45
elif video == 2:
    cap = cv2.VideoCapture("../Vidéos/Extrait2-ManWithAMovieCamera(216p).m4v")
    montageTest = pd.read_csv("../Validation/Montage_2.csv", index_col=0)
    tol = 0.6
elif video == 3:
    cap = cv2.VideoCapture("../Vidéos/Extrait3-Vertigo-Dream_Scene(320p).m4v")
    montageTest = pd.read_csv("../Validation/Montage_3.csv", index_col=0)
    tol = 0.70
elif video == 4:
    cap = cv2.VideoCapture("../Vidéos/Extrait4-Entracte-Poursuite_Corbillard(358p).m4v")
    montageTest = pd.read_csv("../Validation/Montage_4.csv", index_col=0)
    tol = 0.7
elif video == 5:
    cap = cv2.VideoCapture("../Vidéos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
    montageTest = pd.read_csv("../Validation/Montage_5.csv", index_col=0)
    tol = 0.65
else:
    cap = cv2.VideoCapture(0)
    montageTest = pd.read_csv("../Validation/Montage_0.csv", index_col=0)

cutTest = montageTest["Raccord"].to_numpy()
cutHist = np.zeros_like(cutTest)

index  = 1
cut    = 0
nTicks = 4
ret, frame = cap.read()
yuv  = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
hist = cv2.calcHist([yuv], [1,2], None, [bin,bin], [0,256,0,256])

cv2.imwrite('../Images/Frame_%04d.png'%index,frame)
cv2.imwrite('../Images/Hist2Duv_%04d.png'%index,cv2.resize(255*hist/(np.amax(hist)+1),(256,256)))

h = frame.shape[0]
w = frame.shape[1]

while(ret):
    cv2.imshow('Frame',frame)
    cv2.imshow('Histogram 2D de (u,v)',cv2.resize(hist/(np.amax(hist)+1),(256,256)))
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('../Images/Frame_%04d.png'%index,frame)
        cv2.imwrite('../Images/Hist2Duv_%04d.png'%index,cv2.resize(hist/(np.amax(hist)+1),(256,256)))
    hist_old = hist.copy()
    ret, frame = cap.read()
    if(ret):
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        hist = cv2.calcHist([yuv], [1,2], None, [bin,bin], [0,256,0,256])
        hTest = cv2.compareHist(hist_old,hist,0) # métrique : corrélation
        if hTest<tol:
            cut += 1
            cutHist[index] = 1
            #print(f'Raccord {index} : {hTest:.2f}')
        index += 1

# Statistiques
cf = confusion_matrix(cutTest,cutHist)
print(f'''Tolerance           : {tol}''')
print(f'''Nombre des raccords : {cut}''')
print('Matrice de confusion   :')
print(pd.DataFrame(cf))
print(f'''Exactitude  : {(100*cf[0][0]+cf[1][1])/(cf[0][0]+cf[1][0]+cf[0][1]+cf[1][1]):.2f} %''')
print(f'''Précision : {100*cf[1][1]/(cf[0][1]+cf[1][1]):.2f} %''')
print(f'''Rappel    : {100*cf[1][1]/(cf[1][0]+cf[1][1]):.2f} %''')

cap.release()
cv2.destroyAllWindows()
