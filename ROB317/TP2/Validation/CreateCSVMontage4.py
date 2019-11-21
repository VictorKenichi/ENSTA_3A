import numpy as np
import pandas as pd

data = np.zeros(shape=(5065,2))

# Afficher les raccords
data[177] [0] = 1
data[178] [0] = 1
data[179] [0] = 1
data[242] [0] = 1
data[243] [0] = 1
data[244] [0] = 1
data[294] [0] = 1
data[295] [0] = 1
data[296] [0] = 1
data[297] [0] = 1
data[298] [0] = 1
data[296] [0] = 1
data[440] [0] = 1
data[441] [0] = 1
data[442] [0] = 1
data[494] [0] = 1
data[495] [0] = 1
data[496] [0] = 1
data[623] [0] = 1
data[624] [0] = 1
data[625] [0] = 1
data[659] [0] = 1
data[661] [0] = 1
data[660] [0] = 1
data[975] [0] = 1
data[976] [0] = 1
data[1053][0] = 1
data[1153][0] = 1
data[1215][0] = 1
data[1407][0] = 1
data[1408][0] = 1
data[1409][0] = 1
data[1460][0] = 1
data[1461][0] = 1
data[1462][0] = 1
data[1646][0] = 1
data[1919][0] = 1
data[2086][0] = 1
data[2087][0] = 1
data[2167][0] = 1
data[2168][0] = 1
data[2237][0] = 1
data[2238][0] = 1
data[2239][0] = 1
data[2351][0] = 1
data[2399][0] = 1
data[2400][0] = 1
data[2571][0] = 1
data[2572][0] = 1
data[2573][0] = 1
data[2599][0] = 1
data[2658][0] = 1
data[2659][0] = 1
data[2777][0] = 1
data[2778][0] = 1
data[2826][0] = 1
data[2876][0] = 1
data[2926][0] = 1
data[2984][0] = 1
data[3056][0] = 1
data[3081][0] = 1
data[3193][0] = 1
data[3237][0] = 1
data[3238][0] = 1
data[3352][0] = 1
data[3353][0] = 1
data[3405][0] = 1
data[3513][0] = 1
data[3514][0] = 1

for i in range(0, 81):
          data[3556 + i][0]

for i in range(0, 22):
          data[3663 + i][0]

for i in range(0, 15):
          data[3701 + i][0]

for i in range(0, 15):
          data[3743 + i][0]

data[3797][0] = 1
data[3828][0] = 1

for i in range(0, 19):
          data[3836 + i][0]

for i in range(0, 8):
          data[3883 + i][0]

data[3882][0] = 1
data[3890][0] = 1
data[3933][0] = 1
data[3934][0] = 1

for i in range(0, 24):
          data[3945 + i][0]

data[3995][0] = 1
data[4099][0] = 1
data[4174][0] = 1
data[4270][0] = 1
data[4347][0] = 1
data[4382][0] = 1
data[4447][0] = 1
data[4520][0] = 1
data[4620][0] = 1
data[4664][0] = 1
data[4713][0] = 1
data[4799][0] = 1

for i in range(0, 23):
          data[4847 + i][0]

data[4882][0] = 1
data[4900][0] = 1
data[4919][0] = 1
data[4984][0] = 1

df = pd.DataFrame(data, columns=["Raccord","Mouvement de plan"])

df.to_csv('../Montage/Montage_4.csv')
