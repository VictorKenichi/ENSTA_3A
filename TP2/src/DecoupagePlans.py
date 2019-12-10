import cv2
import numpy  as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from argparse        import ArgumentParser

# Paramètres de l'algorithem
r = 4
q = 1
bin = 32 # nombre de bins

# module pour utiliser une ligne pour taper les arguments d'un fichier sur le terminal
parser = ArgumentParser()
parser.add_argument(dest="video", type=int, help="video d'entrée")
input_args = parser.parse_args()
video = int(input_args.video)

if video == 1:
    cap = cv2.VideoCapture("../Vidéos/Extrait1-Cosmos_Laundromat1(340p).m4v")
    montageTest = pd.read_csv("../Validation/Montage_1.csv", index_col=0)
    tolYuv = 0.7
    tolFO  = 0.995
    color  = True
    Polar = True
elif video == 2:
    cap = cv2.VideoCapture("../Vidéos/Extrait2-ManWithAMovieCamera(216p).m4v")
    montageTest = pd.read_csv("../Validation/Montage_2.csv", index_col=0)
    tolYuv = 0.71
    tolFO  = 0.955
    color  = False
    Polar = True
elif video == 3:
    cap = cv2.VideoCapture("../Vidéos/Extrait3-Vertigo-Dream_Scene(320p).m4v")
    montageTest = pd.read_csv("../Validation/Montage_3.csv", index_col=0)
    tolYuv = 0.999
    tolFO  = 0.645
    color  = True
    Polar = False
elif video == 4:
    cap = cv2.VideoCapture("../Vidéos/Extrait4-Entracte-Poursuite_Corbillard(358p).m4v")
    montageTest = pd.read_csv("../Validation/Montage_4.csv", index_col=0)
    tolYuv = 0.7
    tolFO  = 0.993
    color  = False
    Polar = True
elif video == 5:
    cap = cv2.VideoCapture("../Vidéos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
    montageTest = pd.read_csv("../Validation/Montage_5.csv", index_col=0)
    tolYuv = 0.85
    tolFO  = 1
    color  = True
    Polar = True
else:
    cap = cv2.VideoCapture(0)
    montageTest = pd.read_csv("../Validation/Montage_0.csv", index_col=0)
    color = True
    Polar = False

cutTest    = montageTest["Raccord"].to_numpy()
cutHistFO  = np.zeros_like(cutTest)
cutHistYuv = np.zeros_like(cutTest)
cutHist    = np.zeros_like(cutTest)

index  = 3
cut    = 0
nTicks = 4
ret, frame0 = cap.read()

h = frame0.shape[0]
w = frame0.shape[1]

bgrPolarOld = np.zeros(shape=(frame0.shape[0],frame0.shape[1],2), dtype='float32') # Image nulle de même taille que flow
bgrPolarNew = np.copy(bgrPolarOld)

prvs1 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris

ret, frame1 = cap.read() # Passe à l'image suivante

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris

ret, frame2 = cap.read()

yuv = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

if color:
    print("Film coloré")
    histYuvNew = cv2.calcHist([yuv], [1,2], None, [bin,bin], [0,256,0,256])
else:
    print("Film noir et blanc")
    histGrayNew = cv2.calcHist([next], [0], None, [bin], [0,256])

# Paramètres de l'algorithem deu flot optique
pyr_scale  = 0.5 # Taux de réduction pyramidal
levels     = 3 # Nombre de niveaux de la pyramide
winsize    = 15 # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
iterations = 3 # Nb d'itérations par niveau
poly_n     = 7 # Taille voisinage pour approximation polynomiale
poly_sigma = 1.5 # E-T Gaussienne pour calcul dérivées
flags      = 0

flowOld = cv2.calcOpticalFlowFarneback(prvs1,prvs,None,
                                    pyr_scale = pyr_scale,# Taux de réduction pyramidal
                                    levels = levels, # Nombre de niveaux de la pyramide
                                    winsize = winsize, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                    iterations = iterations, # Nb d'itérations par niveau
                                    poly_n = poly_n, # Taille voisinage pour approximation polynomiale
                                    poly_sigma = poly_sigma, # E-T Gaussienne pour calcul dérivées
                                    flags = flags)

if Polar:
    magOld, angOld = cv2.cartToPolar(flowOld[:,:,0], flowOld[:,:,1]) # Conversion cartésien vers polaire
    bgrPolarOld[:,:,0] = 180*angOld/(2*np.pi)
    bgrPolarOld[:,:,1] = 180*magOld/(np.amax(magOld)+1) # Valeur <--> Norme
    histFOOld = cv2.calcHist([bgrPolarOld], [1,0], None, [180/r,180/r], [0,180/q,0,180/q])
else:
    histFOOld = cv2.calcHist([-flowOld], [1,0], None, [2*h/r,2*w/r], [-h/q,h/q,-w/q,w/q])

while(ret):
#    cv2.imshow('Frame',frame2)
#    cv2.imshow('Histogram 2D de (Vx,Vy)', histFOOld/(np.amax(histFOOld)+1))
#    cv2.imshow('Histogram 2D de (u,v)',histYuvNew/(np.amax(histYuvNew)+1))
    k = cv2.waitKey(1) & 0xff

    flowNew = cv2.calcOpticalFlowFarneback(next,prvs,None,
                                        pyr_scale = pyr_scale,# Taux de réduction pyramidal
                                        levels = levels, # Nombre de niveaux de la pyramide
                                        winsize = winsize, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = iterations, # Nb d'itérations par niveau
                                        poly_n = poly_n, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = poly_sigma, # E-T Gaussienne pour calcul dérivées
                                        flags = flags)

    if Polar:
        magNew, angNew = cv2.cartToPolar(flowNew[:,:,0], flowNew[:,:,1]) # Conversion cartésien vers polaire
        bgrPolarNew[:,:,0] = 180*angNew/(2*np.pi)
        bgrPolarNew[:,:,1] = 180*magNew/(np.amax(magNew)+1) # Valeur <--> Norme
        histFONew = cv2.calcHist([bgrPolarNew], [1,0], None, [180/r,180/r], [0,180/q,0,180/q])
    else:
        histFONew = cv2.calcHist([-flowNew], [1,0], None, [2*h/r,2*w/r], [-h/q,h/q,-w/q,w/q])

    if color:
        histYuvOld = histYuvNew.copy()
    else:
        histGrayOld = histGrayNew.copy()

    prvs = next

    ret, frame2 = cap.read()
    if(ret):
        yuv = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        if color:
            histYuvNew = cv2.calcHist([yuv], [1,2], None, [bin,bin], [0,256,0,256])
            hTestYuv = cv2.compareHist(histYuvOld,histYuvNew,0) # métrique : corrélation
        else:
            histGrayNew = cv2.calcHist([next], [0], None, [bin], [0,256])
            hTestYuv = cv2.compareHist(histGrayOld,histGrayNew,0) # métrique : corrélation
        hTestFO = cv2.compareHist(histFONew,histFOOld,0) # métrique : corrélation
        histFOOld = histFONew
        if hTestYuv<tolYuv:
            cutHistYuv[index] = 1
            #print(f'Raccord Yuv {index} : {hTestYuv:.2f}')
        if hTestFO<tolFO:
            cutHistFO[index] = 1
            #print(f'Raccord Flot Optique {index} : {hTestFO:.2f}')
        if hTestYuv<tolYuv and hTestFO<tolFO:
            cut += 1
            cutHist[index] = 1
            #print(f'Raccord Combiné {index} : hTestYuv = {hTestYuv:.2f} et hTestFO = {hTestFO:.2f}')
        index += 1

cfYuv = confusion_matrix(cutTest,cutHistYuv)
cfFO  = confusion_matrix(cutTest,cutHistFO)
cf    = confusion_matrix(cutTest,cutHist)

# Statistiques
if color:
    print(f'Tolerance de Yuv     : {tolYuv}')
    print('Matrice de confusion de Yuv :')
    print(pd.DataFrame(cfYuv))
    print(f'Exactitude de Yuv : {(100*cfYuv[0][0]+cfYuv[1][1])/(cfYuv[0][0]+cfYuv[1][0]+cfYuv[0][1]+cfYuv[1][1]):.2f} %')
    print(f'Précision de Yuv  : {100*cfYuv[1][1]/(cfYuv[0][1]+cfYuv[1][1]):.2f} %')
    print(f'Rappel de Yuv     : {100*cfYuv[1][1]/(cfYuv[1][0]+cfYuv[1][1]):.2f} %')
else:
    print(f'Tolerance de Gris     : {tolYuv}')
    print('Matrice de confusion de Gris :')
    print(pd.DataFrame(cfYuv))
    print(f'Exactitude de Gris : {(100*cfYuv[0][0]+cfYuv[1][1])/(cfYuv[0][0]+cfYuv[1][0]+cfYuv[0][1]+cfYuv[1][1]):.2f} %')
    print(f'Précision de Gris  : {100*cfYuv[1][1]/(cfYuv[0][1]+cfYuv[1][1]):.2f} %')
    print(f'Rappel de Gris     : {100*cfYuv[1][1]/(cfYuv[1][0]+cfYuv[1][1]):.2f} %')

print(f'Tolerance de Flot Optique         : {tolFO}')
if Polar:
    print('Coordonées polaires')
else:
    print('Coordonées cartésiennes')
print('Matrice de confusion de Flot Optique    :')
print(pd.DataFrame(cfFO))
print(f'Exactitude de Flot Optique : {(100*cfFO[0][0]+cfFO[1][1])/(cfFO[0][0]+cfFO[1][0]+cfFO[0][1]+cfFO[1][1]):.2f} %')
print(f'Précision de Flot Optique  : {100*cfFO[1][1]/(cfFO[0][1]+cfFO[1][1]):.2f} %')
print(f'Rappel de Flot Optique     : {100*cfFO[1][1]/(cfFO[1][0]+cfFO[1][1]):.2f} %')

print(f'''Nombre des raccords combiné : {cut}''')

print(f'''Tolerance de Yuv              : {tolYuv}''')
print(f'''Tolerance de Flot Optique     : {tolFO}''')
print('Matrice de confusion combiné :')
print(pd.DataFrame(cf))
print(f'Exactitude combinée : {(100*cf[0][0]+cf[1][1])/(cf[0][0]+cf[1][0]+cf[0][1]+cf[1][1]):.2f} %')
print(f'Précision combinée  : {100*cf[1][1]/(cf[0][1]+cf[1][1]):.2f} %')
print(f'Rappel combiné      : {100*cf[1][1]/(cf[1][0]+cf[1][1]):.2f} %')


cap.release()
cv2.destroyAllWindows()
