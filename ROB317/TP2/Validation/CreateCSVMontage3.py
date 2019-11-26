import numpy as np
import pandas as pd

data = np.zeros(shape=(2375,2))

# Afficher les raccords
for i in range(1,40):
    data[i+100][0] = 1
data[284][0] = 1
for i in range(75,91):
    data[i+700][0] = 1
for i in range(1,12):
    data[i+800][0] = 1
data[1054][0] = 1
data[1184][0] = 1
data[1351][0] = 1
data[1660][0] = 1
data[1838][0] = 1
data[2025][0] = 1
data[2211][0] = 1
for i in range(88,128):
    data[i+2200][0] = 1

df = pd.DataFrame(data, columns=["Raccord","Mouvement de plan"])

df.to_csv('../Validation/Montage_3.csv')
