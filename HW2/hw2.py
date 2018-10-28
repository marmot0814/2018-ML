from process import process
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
# data processing
df = pd.read_csv('./datasets/data_noah.csv')[['x', 'y']]
df = (df - df.mean()) / (df.max() - df.min())
data = np.array([[df['x'].values[i], df['y'].values[i]]
                 for i in range(df['x'].values.shape[0])])

result = process.Kmeans(data, k=3)
