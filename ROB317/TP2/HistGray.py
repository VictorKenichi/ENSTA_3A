import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("./Vidéos/Extrait1-Cosmos_Laundromat1(340p).m4v")
#cap = cv2.VideoCapture("./Vidéos/Extrait4-Entracte-Poursuite_Corbillard(358p).m4v")

index = 1
mse = 0
cut = 0
ret, frame = cap.read()
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [256], [0,256])
hist = hist/256**2

# Paramètres de l'algorithem
kernel = 3 # voisinage consideré
tol = 0.2 # tolerance

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
    cv2.imshow('Image et Champ de vitesses (Farnebäck)',gray)
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
    index += 1
    if(ret):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = hist/256**2
        mse = 0
        for i in range(kernel-kernel):
            if hist[i][0] != 0:
                diff_min = (hist[i][0]-hist_old[i][0])**2
                for i2 in range(-kernel//2+1,kernel//2):
                    diff = (hist[i][0] - hist_old[i+i2][0])**2
                    if diff < diff_min:
                        diff_min = diff
                mse += diff_min
        if mse>tol:
            cut += 1
            print(f'''index = {index}''')
            print(f'''mse = {mse}''')
print(f'''Nombre des raccords : {cut}''')
cap.release()
cv2.destroyAllWindows()
