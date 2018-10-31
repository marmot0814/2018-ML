from process import process
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
# validation

#target = ['vx0', 'vy0', 'vz0']  # 2 or 3 feature
target = ['x', 'y']
# data processing
df = pd.read_csv('./datasets/data_noah.csv')

# df = (df - df.mean()) / (df.max() - df.min())
data = np.array([[df[_str].values[i]
                  for _str in target]
                 for i in range(df[target[0]].values.shape[0])])
x, y = [], []
# for t in range(1, 10):
for t in range(3, 4):
	print("Cluster:",t)
	result = process.Kmeans(data, k=t)
	# validation
	data_type = [ df['pitch_type'].values[i] for i in range(df[target[0]].values.shape[0])]
	accuracy = result.accuracy(data,data_type);
	print("Accuracy:", accuracy, "%")
	# cost
	cost = result.cost(data)
	print("Cost:", cost)
	# x.append(t)
	# y.append(cost/t)

# plt.plot(x ,y)
# plt.show()