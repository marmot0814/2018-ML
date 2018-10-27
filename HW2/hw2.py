from process import test
import numpy as np
import pandas as pd

# data processing
df = pd.read_csv('./datasets/data_noah.csv')[['x', 'y', 'pitch_type']]
data = np.array([[df['x'].values[i], df['y'].values[i]]
                 for i in range(df['x'].values.shape[0])])

result = test.Kmeans(data, k = 20)
