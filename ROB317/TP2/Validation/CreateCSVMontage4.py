import numpy as np
import pandas as pd

data = np.zeros(shape=(5065,2))

# Afficher les raccords
data[179][0] = 1
data[243][0] = 1
data[294][0] = 1
data[296][0] = 1
data[440][0] = 1
data[494][0] = 1
data[624][0] = 1
data[660][0] = 1
data[975][0] = 1
data[1053][0] = 1
data[1153][0] = 1
data[1215][0] = 1
data[1409][0] = 1
data[1460][0] = 1
data[1646][0] = 1
data[1919][0] = 1
data[2087][0] = 1
data[2167][0] = 1
data[2238][0] = 1
data[2351][0] = 1
data[2399][0] = 1
data[2572][0] = 1
data[2599][0] = 1
data[2659][0] = 1
data[2778][0] = 1
data[2826][0] = 1
data[2876][0] = 1
data[2926][0] = 1
data[2984][0] = 1
data[3056][0] = 1
data[3081][0] = 1
data[3193][0] = 1
data[3238][0] = 1
data[3352][0] = 1
data[3405][0] = 1
data[3514][0] = 1
data[3576][0] = 1
data[3636][0] = 1
data[3677][0] = 1
data[3700][0] = 1
data[3797][0] = 1
data[3828][0] = 1
data[3854][0] = 1
data[3882][0] = 1
data[3890][0] = 1
data[3933][0] = 1
data[3944][0] = 1
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
data[4854][0] = 1
data[4882][0] = 1
data[4900][0] = 1
data[4919][0] = 1
data[4984][0] = 1

df = pd.DataFrame(data, columns=["Raccord","Mouvement de plan"])

df.to_csv('./Montage_4.csv')
