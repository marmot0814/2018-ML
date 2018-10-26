import numpy as np
import pandas as pd
from process import process
from matplotlib import pyplot as plt

# data processing
df = pd.read_csv("./datasets/data_noah.csv")
df = df[['x', 'y', 'pitch_type']]
groups = df.groupby('pitch_type')
x = df['x'].values.tolist()
y = df['y'].values.tolist()
# plot 
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
ax.legend()
plt.show()
