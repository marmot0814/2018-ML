import csv
import numpy as np
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

csv_file = open('data/new_googleplaystore.csv')
reader = csv.reader(csv_file)
headers = next(reader)

maxV = [-np.inf for x in range(len(headers) - 1)]
minV = [np.inf for x in range(len(headers) - 1)]

data = []
label = []
label_dict = {}

for row in reader:
    for index in range(len(row) - 1):
        maxV[index] = max(maxV[index], float(row[index]))
        minV[index] = min(minV[index], float(row[index]))
    
        data_row = []
    for index in range(len(row) - 1):
        data_row.append(float(row[index]))

    data.append(data_row) 

    if label_dict.get(row[-1]) == None:
        label_dict[row[-1]] = len(label_dict)
    label.append(label_dict[row[-1]])

for row_index in range(len(data)):
    for col_index in range(len(data[row_index])):
        data[row_index][col_index] = (float(data[row_index][col_index]) - minV[col_index]) / (maxV[col_index] - minV[col_index])

label_name = [0 for x in range(len(label_dict))]
for item in label_dict:
    label_name[label_dict[item]] = item

label = np.asarray(label)
pca=PCA(n_components=3)
newData=pca.fit_transform(data)
ax = Axes3D(plt.figure())

color = ['aqua', 'black', 'crimson', 'darkseagreen', 'gold']

for i in range(len(label_name)):
    ax.scatter(newData[label == i, 0], newData[label == i, 1], newData[label == i, 2], s = 0.5, c = color[i], alpha = 0.8, label = label_name[i])
plt.legend()
plt.show()
