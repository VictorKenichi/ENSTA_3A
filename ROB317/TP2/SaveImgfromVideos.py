import cv2
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser

# module pour utiliser une ligne pour taper les arguments d'un fichier sur le terminal
parser = ArgumentParser()
parser.add_argument(dest="video", type=int, help="video d'entrée")
input_args = parser.parse_args()
video = int(input_args.video)

if video == 1:
    cap = cv2.VideoCapture("./Vidéos/Extrait1-Cosmos_Laundromat1(340p).m4v")
elif video == 2:
    cap = cv2.VideoCapture("./Vidéos/Extrait2-ManWithAMovieCamera(216p).m4v")
elif video == 3:
    cap = cv2.VideoCapture("./Vidéos/Extrait3-Vertigo-Dream_Scene(320p).m4v")
elif video == 4:
    cap = cv2.VideoCapture("./Vidéos/Extrait4-Entracte-Poursuite_Corbillard(358p).m4v")
elif video == 5:
    cap = cv2.VideoCapture("./Vidéos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
else:
    cap = cv2.VideoCapture(0)

index = 1
ret, frame = cap.read() # Passe à l'image suivante

while(ret):
    cv2.imwrite('./Images/Frame_%04d.png'%index,frame)
    cv2.imshow('Images',frame)
    k = cv2.waitKey(15) & 0xff
    ret, frame = cap.read()
    index += 1

cap.release()
cv2.destroyAllWindows()
