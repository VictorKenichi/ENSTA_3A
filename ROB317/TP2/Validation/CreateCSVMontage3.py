import numpy as np
import pandas as pd

data = np.zeros(shape=(2375,2))

# Afficher les raccords
data[110][0] = 1
data[284][0] = 1
data[788][0] = 1
data[803][0] = 1
data[1054][0] = 1
data[1184][0] = 1
data[1351][0] = 1
data[1660][0] = 1
data[1838][0] = 1
data[2025][0] = 1
data[2211][0] = 1
data[2307][0] = 1

df = pd.DataFrame(data, columns=["Raccord","Mouvement de plan"])

df.to_csv('../Montage/Montage_3.csv')
