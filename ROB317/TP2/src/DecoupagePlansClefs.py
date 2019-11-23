import cv2
import numpy  as np
import pandas as pd

from   sklearn.metrics import confusion_matrix
from   matplotlib      import pyplot          as plt
from   argparse        import ArgumentParser

# Paramètres de l'algorithem
r   = 4
q   = 1
bin = 32 # nombre de bins

# Module pour utiliser une ligne pour taper les arguments d'un fichier sur le terminal
parser     = ArgumentParser()
parser.add_argument(dest="video", type=int, help="video d'entrée")
input_args = parser.parse_args()
video      = int(input_args.video)

if video == 1:
    cap         = cv2.VideoCapture("../Vidéos/Extrait1-Cosmos_Laundromat1(340p).m4v")
    montageTest = pd.read_csv("../Montage/Montage_1.csv", index_col=0)
    tolYuv      = 0.3
    tolFO       = 0.05
    color       = 3 # si 1 c'est gris et si 3 c'est coloré

elif video == 2:
    cap         = cv2.VideoCapture("../Vidéos/Extrait2-ManWithAMovieCamera(216p).m4v")
    montageTest = pd.read_csv("../Montage/Montage_2.csv", index_col=0)
    tolYuv      = 0.3
    tolFO       = 0.2
    color       = 1

elif video == 3:
    cap         = cv2.VideoCapture("../Vidéos/Extrait3-Vertigo-Dream_Scene(320p).m4v")
    montageTest = pd.read_csv("../Montage/Montage_3.csv", index_col=0)
    tolYuv      = 0.1
    tolFO       = 0.1
    color       = 3

elif video == 4:
    cap         = cv2.VideoCapture("../Vidéos/Extrait4-Entracte-Poursuite_Corbillard(358p).m4v")
    montageTest = pd.read_csv("../Montage/Montage_4.csv", index_col=0)
    tolYuv      = 0.3
    tolFO       = 0.15
    color       = 1

elif video == 5:
    cap         = cv2.VideoCapture("../Vidéos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
    montageTest = pd.read_csv("../Montage/Montage_5.csv", index_col=0)
    tolYuv      = 0.2
    tolFO       = 0.23
    color       = 3

else:
    cap         = cv2.VideoCapture(0)
    montageTest = pd.read_csv("../Montage/Montage_0.csv", index_col=0)
    color       = 3

cutTest         = montageTest["Raccord"].to_numpy()
cutHistFO       = np.zeros_like(cutTest)
cutHistYuv      = np.zeros_like(cutTest)
cutHist         = np.zeros_like(cutTest)
     
index           = 3
cut             = 0
frames_per_plan = 1          # Compteur de frames entre le changement de plan
nTicks          = 4
ret, frame0     = cap.read()

h               = frame0.shape[0]
w               = frame0.shape[1]
    
bgrPolarOld     = np.zeros(shape=(frame0.shape[0],frame0.shape[1],2), dtype='float32') # Image nulle de même taille que flow
bgrPolarNew     = np.copy(bgrPolarOld)
    
prvs1           = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
ret, frame1     = cap.read()                              # Passe à l'image suivante
prvs            = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
ret, frame2     = cap.read()

yuv  = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

if color == 3:
    print("Film Coloré")
    histYuvNew = cv2.calcHist([yuv], [1,2], None, [bin,bin], [0,256,0,256])
else:
    print("Film Noir et Blanc")
    histGrayNew = cv2.calcHist([next], [0], None, [bin], [0,256])

# Paramètres de l'algorithem deu flot optique
pyr_scale  = 0.5 # Taux de réduction pyramidal
levels     = 3   # Nombre de niveaux de la pyramide
winsize    = 15  # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
iterations = 3   # Nb d'itérations par niveau
poly_n     = 7   # Taille voisinage pour approximation polynomiale
poly_sigma = 1.5 # E-T Gaussienne pour calcul dérivées
flags      = 0

flowOld = cv2.calcOpticalFlowFarneback(prvs1,prvs,None,
                                       pyr_scale  = pyr_scale,  # Taux de réduction pyramidal
                                       levels     = levels,     # Nombre de niveaux de la pyramide
                                       winsize    = winsize,    # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                       iterations = iterations, # Nb d'itérations par niveau
                                       poly_n     = poly_n,     # Taille voisinage pour approximation polynomiale
                                       poly_sigma = poly_sigma, # E-T Gaussienne pour calcul dérivées
                                       flags      = flags)

magOld, angOld     = cv2.cartToPolar(flowOld[:,:,0], flowOld[:,:,1]) # Conversion cartésien vers polaire
bgrPolarOld[:,:,0] = 180*angOld/(2*np.pi)
bgrPolarOld[:,:,1] = 180*magOld/np.amax(magOld) # Valeur <--> Norme
histFOOld          = cv2.calcHist([bgrPolarOld], [1,0], None, [180/r,180/r], [0,180/q,0,180/q])

# Initialisation de Histogrammes 
hist_average       = []   
hist_average_temp  = histFOOld
hist_total         = hist_average_temp

while(ret):
    cv2.imshow('Image et Champ de vitesses (Farnebäck)', frame2)
    k = cv2.waitKey(5) & 0xff

    flowNew = cv2.calcOpticalFlowFarneback(next,prvs,None,
                                           pyr_scale  = pyr_scale,  # Taux de réduction pyramidal
                                           levels     = levels,     # Nombre de niveaux de la pyramide
                                           winsize    = winsize,    # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                           iterations = iterations, # Nb d'itérations par niveau
                                           poly_n     = poly_n,     # Taille voisinage pour approximation polynomiale
                                           poly_sigma = poly_sigma, # E-T Gaussienne pour calcul dérivées
                                           flags      = flags)

    magNew, angNew     = cv2.cartToPolar(flowNew[:,:,0], flowNew[:,:,1]) # Conversion cartésien vers polaire
    bgrPolarNew[:,:,0] = 180*angNew/(2*np.pi)
    bgrPolarNew[:,:,1] = 180*magNew/np.amax(magNew) # Valeur <--> Norme

    histFONew  = cv2.calcHist([bgrPolarNew], [1,0], None, [180/r,180/r], [0,180/q,0,180/q])
    hist_total = hist_total + histFONew

    frames_per_plan += 1 
    print(frames_per_plan)

    if color == 3:
        histYuvOld   = histYuvNew.copy()
    else: 
        histGrayOld  = histGrayNew.copy()

    prvs = next

    ret, frame2 = cap.read()
    if(ret):
        yuv  = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        if color == 3:
            histYuvNew  = cv2.calcHist([yuv], [1,2], None, [bin,bin], [0,256,0,256])
            hTestYuv    = cv2.compareHist(histYuvOld, histYuvNew, 0)
        else:
            histGrayNew = cv2.calcHist([next], [0], None, [bin], [0,256])
            hTestYuv    = cv2.compareHist(histGrayOld, histGrayNew, 0)

        hTestFO   = cv2.compareHist(histFONew, histFOOld, 0)
        histFOOld = histFONew

        '''
        # Reinitialization 
        def reinitialization():
            hist_average_temp = hist_total/frames_per_plan
            hist_average.append(hist_average_temp)

            print(frames_per_plan)

            hist_total        = histFONew
            hist_average_temp = hist_total 
            frames_per_plan   = 1 
        '''

        # Détection de changement de plan 
        if hTestYuv < 1 - tolYuv:
            cutHistYuv[index] = 1
            #print(f'''cutHistYuv[{index}] : {cutHistYuv[index]}''')

        if hTestFO < 1 - tolFO:
            cutHistFO[index] = 1
            #print(f'''cutHistFO [{index}] : {cutHistFO[index]}''')

        if hTestYuv < 1 - tolYuv and hTestFO < 1 - tolFO:
            cut              += 1
            cutHist[index]    = 1
            print(f'''cutHist   [{index}] : {cutHist[index]}''')
            
            # Reinitialization
            #reinitialization()  
            hist_average_temp = hist_total/frames_per_plan
            hist_average.append(hist_average_temp)

            print(frames_per_plan)

            hist_total        = histFONew
            hist_average_temp = hist_total 
            frames_per_plan   = 1 

        index += 1

# Comparer chaque frame avec le histrogrammes moyens capturés
while(ret):

    # Trouver l'histrogramme qui se rassemble le plus avec le histogramme moyen 
    for i in hist_average:
        for j in ret: 
            ret, frame_compare = cap.read()
            yuv                = cv2.cvtColor(frame_compare, cv2.COLOR_BGR2YUV)
            next               = cv2.cvtColor(frame_compare, cv2.COLOR_BGR2GRAY)
            frame_index        = 1

            if color == 3:
                histYuvNew  = cv2.calcHist([yuv], [1,2], None, [bin,bin], [0,256,0,256])
                hTestYuv    = cv2.compareHist(i, histYuvNew, 0)

            else:
                hist_frame_gray = cv2.calcHist([next], [0], None, [bin], [0,256])
                hTestYuv        = cv2.compareHist(i, hist_frame_gray, 0)

            frame_index += 1

            # Percorrer todas as imagens e buscar a maior correlação 

            # Sauvegarder l'image
            cv2.imwrite('../Images_Plan_Clefs/Frame_%04d.png'%frame_index, frame_compare)

cfYuv = confusion_matrix(cutTest,cutHistYuv)
cfFO  = confusion_matrix(cutTest,cutHistFO)
cf    = confusion_matrix(cutTest,cutHist)

print(f''' Tamanho do vetor de histograma medio : {len(hist_average)}''')

# === Statistiques
print(f'''Tolerance de Yuv         : {tolYuv}''')
print('Matrice de confusion de Yuv :')
print(pd.DataFrame(cfYuv))
print(f'''Accuracy de Yuv  : {(100*cfYuv[0][0]+cfYuv[1][1])/(cfYuv[0][0]+cfYuv[1][0]+cfYuv[0][1]+cfYuv[1][1])} %''')
print(f'''Precision de Yuv : {100*cfYuv[1][1]/(cfYuv[0][1]+cfYuv[1][1])} %''')
print(f'''Recall de Yuv    : {100*cfYuv[1][1]/(cfYuv[1][0]+cfYuv[1][1])} %''')

print(f'''Tolerance de Flot Optique         : {tolFO}''')
print('Matrice de confusion de Flot Optique :')
print(pd.DataFrame(cfFO))
print(f'''Accuracy de Flot Optique  : {(100*cfFO[0][0]+cfFO[1][1])/(cfFO[0][0]+cfFO[1][0]+cfFO[0][1]+cfFO[1][1])} %''')
print(f'''Precision de Flot Optique : {100*cfFO[1][1]/(cfFO[0][1]+cfFO[1][1])} %''')
print(f'''Recall de Flot Optique    : {100*cfFO[1][1]/(cfFO[1][0]+cfFO[1][1])} %''')

print(f'''Nombre des raccords combiné : {cut}''')

print(f'''Tolerance de Yuv            : {tolYuv}''')
print(f'''Tolerance de Flot Optique   : {tolFO}''')
print('Matrice de confusion combiné   :')
print(pd.DataFrame(cf))
print(f'''Accuracy combiné  : {(100*cf[0][0]+cf[1][1])/(cf[0][0]+cf[1][0]+cf[0][1]+cf[1][1])} %''')
print(f'''Precision combiné : {100*cf[1][1]/(cf[0][1]+cf[1][1])} %''')
print(f'''Recall combiné    : {100*cf[1][1]/(cf[1][0]+cf[1][1])} %''')


cap.release()
cv2.destroyAllWindows()
