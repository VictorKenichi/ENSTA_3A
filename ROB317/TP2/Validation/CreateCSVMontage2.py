import numpy as np
import pandas as pd

data = np.zeros(shape=(1649, 2))

# Afficher les raccords
data[157][0] = 1
data[158][0] = 1
data[203][0] = 1
data[204][0] = 1
data[387][0] = 1
data[518][0] = 1
data[568][0] = 1
data[667][0] = 1
data[668][0] = 1
data[958][0] = 1
data[959][0] = 1
data[966][0] = 1
data[967][0] = 1
data[968][0] = 1
data[994][0] = 1
data[995][0] = 1
data[1000][0] = 1
data[1001][0] = 1
data[1027][0] = 1
data[1028][0] = 1
data[1029][0] = 1
data[1044][0] = 1
data[1045][0] = 1
data[1104][0] = 1
data[1105][0] = 1
data[1134][0] = 1
data[1135][0] = 1
data[1253][0] = 1
data[1276][0] = 1
data[1310][0] = 1
data[1311][0] = 1
data[1312][0] = 1
data[1381][0] = 1
data[1382][0] = 1
data[1590][0] = 1
data[1591][0] = 1

df = pd.DataFrame(data, columns=["Raccord","Mouvement de plan"])

df.to_csv('../Montage/Montage_2.csv')
