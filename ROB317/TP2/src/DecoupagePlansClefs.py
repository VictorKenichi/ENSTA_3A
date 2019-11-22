import cv2
import numpy as np

from   matplotlib import pyplot as plt
from   argparse   import ArgumentParser

#bin = 32 # nombre de bins
r   = 1
q   = 1
tol = 0.4

# module pour utiliser une ligne pour taper les arguments d'un fichier sur le terminal
parser = ArgumentParser()
parser.add_argument(dest="video", type=int, help="video d'entrée")
input_args = parser.parse_args()
video = int(input_args.video)

if video == 1:
    cap         = cv2.VideoCapture("../Vidéos/Extrait1-Cosmos_Laundromat1(340p).m4v")
    montageTest = pd.read_csv("../Montage/Montage_1.csv", index_col=0)

elif video == 2:
    cap         = cv2.VideoCapture("../Vidéos/Extrait2-ManWithAMovieCamera(216p).m4v")
    montageTest = pd.read_csv("../Montage/Montage_2.csv", index_col=0)

elif video == 3:
    cap         = cv2.VideoCapture("../Vidéos/Extrait3-Vertigo-Dream_Scene(320p).m4v")
    montageTest = pd.read_csv("../Montage/Montage_3.csv", index_col=0)

elif video == 4:
    cap         = cv2.VideoCapture("../Vidéos/Extrait4-Entracte-Poursuite_Corbillard(358p).m4v")
    montageTest = pd.read_csv("../Montage/Montage_4.csv", index_col=0)

elif video == 5:
    cap         = cv2.VideoCapture("../Vidéos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
    montageTest = pd.read_csv("../Montage/Montage_5.csv", index_col=0)

else:
    cap         = cv2.VideoCapture(0)
    montageTest = pd.read_csv("../Montage/Montage_0.csv", index_col=0)

ret, frame1 = cap.read() # Passe à l'image suivante
yuv         = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)
h           = frame1.shape[0]
w           = frame1.shape[1]

ret, frame0 = cap.read() # Passe à l'image suivante
prvs1       = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris

ret, frame1 = cap.read() # Passe à l'image suivante
prvs        = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris

cut         = 0
index       = 1
ret, frame2 = cap.read()
next        = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

while(ret):
    index += 1
    flow1 = cv2.calcOpticalFlowFarneback(prvs,next,None,
                                        pyr_scale  = 0.5, # Taux de réduction pyramidal
                                        levels     = 3,   # Nombre de niveaux de la pyramide
                                        winsize    = 15,  # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3,   # Nb d'itérations par niveau
                                        poly_n     = 7,   # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées
                                        flags      = 0)

    flow2 = cv2.calcOpticalFlowFarneback(prvs1,prvs,None,
                                        pyr_scale  = 0.5,# Taux de réduction pyramidal
                                        levels     = 3, # Nombre de niveaux de la pyramide
                                        winsize    = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n     = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées
                                        flags      = 0)

    hist1 = cv2.calcHist([flow1], [1,0], None, [h/r,w/r], [-h/q,h/q,-w/q,w/q])

    hist2 = cv2.calcHist([flow2], [1,0], None, [h/r,w/r], [-h/q,h/q,-w/q,w/q])

    hTest = cv2.compareHist(hist2,hist1,0)

    cv2.imshow('Image',frame2)
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png'%index,frame2)
        cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
    prvs1 = prvs
    prvs = next
    ret, frame2 = cap.read()
    if (ret):
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        if hTest<1-tol:
            cut += 1
            print(f'''index = {index}''')
            print(f'''Correlation = {hTest}''')
print(f'''Nombre des raccords : {cut}''')

cap.release()
cv2.destroyAllWindows()
