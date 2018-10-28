from process import process
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

target = ['vx0', 'vy0', 'vz0']          # 2 or 3 feature

# data processing
df = pd.read_csv('./datasets/data_noah.csv')[target]
df = (df - df.mean()) / (df.max() - df.min())
data = np.array([[df[_str].values[i] for _str in target ]
                 for i in range(df[target[0]].values.shape[0])])

result = process.Kmeans(data, k=2)
