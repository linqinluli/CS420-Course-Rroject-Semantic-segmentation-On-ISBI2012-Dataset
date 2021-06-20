from os import read
import numpy as np
import pandas as pd
from pandas.core.indexes.range import RangeIndex

def read_data(filename):
    data = pd.read_csv(filename, header=None)
    data = data.values
    x = data[:,1]
    rand = data[:,3]
    info = data[:,5]
    loss = data[:,7]
    return x, rand, info, loss
#%%
x1, vrand1, info1, loss1 = read_data('origin.csv')
x2, vrand2, info2, loss2 = read_data('aug_log.csv')
x3, vrand3, info3, loss3 = read_data('model_modify_log.csv')
#%%
import matplotlib.pyplot as plt

plt.xlabel('Epoch')
plt.ylabel('Vrand')

plt.plot(x1, vrand1, label = 'Origin Unet')
plt.plot(x2, vrand2, label = 'Data Augmentation')
plt.plot(x3, vrand3, label = 'Modify Model')

plt.legend()
# %%
plt.xlabel('Epoch')
plt.ylabel('Vinfo')

plt.plot(x1, info1, label = 'Origin Unet')
plt.plot(x2, info2, label = 'Data Augmentation')
plt.plot(x3, info3, label = 'Modify Model')

plt.legend()
# %%
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.plot(x1, loss1, label = 'Origin Unet')
plt.plot(x2, loss2, label = 'Data Augmentation')
plt.plot(x3, loss3, label = 'Modify Model')

plt.legend()
# %%
