import numpy as np
import pandas as pd

data = np.zeros(shape=(3168,2))

# Afficher les raccords
data[250][0] = 1
data[479][0] = 1
data[511][0] = 1
data[600][0] = 1
data[653][0] = 1
data[691][0] = 1
data[1114][0] = 1
data[1181][0] = 1
data[1310][0] = 1
data[1415][0] = 1
data[1517][0] = 1
data[1565][0] = 1
data[1712][0] = 1
data[1781][0] = 1
data[1864][0] = 1
data[1989][0] = 1
data[2047][0] = 1
data[2166][0] = 1
data[2216][0] = 1
data[2278][0] = 1
data[2442][0] = 1
data[2512][0] = 1
data[2559][0] = 1
data[2637][0] = 1
data[2714][0] = 1
data[2765][0] = 1
data[2838][0] = 1
data[3020][0] = 1
data[3094][0] = 1
data[3131][0] = 1
data[3162][0] = 1

df = pd.DataFrame(data, columns=["Raccord","Mouvement de plan"])

df.to_csv('./Montage_1.csv')
