import numpy as np
import pandas as pd
from process import process
import matplotlib.pyplot as plt
import time

# data processing
df = pd.read_csv("./datasets/data_noah.csv")
df = df[['x', 'y', 'pitch_type']]
data = np.array([[df['x'].values[i], df['y'].values[i]]
                 for i in range(df['x'].values.shape[0])])
# kmeans
k = 3
cModel = process.Kmeans(k)  # clustering model , k = 3
sizes_clusters = cModel.data_label(data, np.unique(df['pitch_type'].values))
# sizes_clusters = [[1, 2, 3], [[1, 2, 3], [4, 5, 6]]]
plt.ion()
for i in range(5000):
    sizes_clusters = cModel.iter(sizes_clusters)
    if (sizes_clusters == None):
        time.sleep(2)
        break
    cModel.plot(sizes_clusters)
