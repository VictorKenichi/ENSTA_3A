import numpy as np
import pandas as pd

data = np.zeros(shape=(768,2))

# Afficher les raccords
data[62][0] = 1
data[103][0] = 1
data[146][0] = 1
data[208][0] = 1
data[253][0] = 1
data[296][0] = 1
data[361][0] = 1
data[409][0] = 1
data[447][0] = 1
data[512][0] = 1
data[621][0] = 1
data[679][0] = 1
data[745][0] = 1

df = pd.DataFrame(data, columns=["Raccord","Mouvement de plan"])

df.to_csv('./Montage/Montage_5.csv')
