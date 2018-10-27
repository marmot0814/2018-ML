import numpy as np
from sklearn.neighbors import KDTree
import os
import matplotlib.pyplot as plt

# load and preprocessing data
# data is a 2-dim numpy array
with open(os.path.join('datasets', 'points.txt'), 'r') as f:
    data = f.read().split('\n')
for i in range(len(data)):
    temp = data[i].split(' ')
    data[i] = [float(temp[0]), float(temp[1])]
data = np.array(data)

# show kd-tree
## calculate the start cut
variance = np.var(data, axis=0)
if variance[0] > variance[1]:
    cut_axis = 0
else:
    cut_axis = 1


## def Cut function for every iteration
## min_x, max_x means x plot boundary
def Cut(pts, axis, min_x, max_x, min_y, max_y):
    pts = sorted(pts, key=lambda x: x[axis])
    median = int((1 + len(pts)) / 2) - 1
    if axis == 0:
        plt.plot([pts[median][0], pts[median][0]], [min_y, max_y],
                 color='red',
                 linewidth=1)
        if median != 0:
            Cut(pts[:median], 1, min_x, pts[median][0], min_y, max_y)
            Cut(pts[median + 1:], 1, pts[median][0], max_x, min_y, max_y)
        if len(pts) == 2:
            Cut(pts[median + 1:], 1, pts[median][0], max_x, min_y, max_y)
    else:
        plt.plot([min_x, max_x], [pts[median][1], pts[median][1]],
                 color='blue',
                 linewidth=1)
        if median != 0:
            Cut(pts[:median], 0, min_x, max_x, min_y, pts[median][1])
            Cut(pts[median + 1:], 0, min_x, max_x, pts[median][1], max_y)
        if len(pts) == 2:
            Cut(pts[median + 1:], 0, min_x, max_x, pts[median][1], max_y)


## run cut iteration
max_data = np.amax(data, axis=0)
min_data = np.amin(data, axis=0)
Cut(data, cut_axis, min_data[0] - 2, max_data[0] + 2, min_data[1] - 2,
    max_data[1] + 2)
plt.scatter(data[:, 0], data[:, 1], c='black', s=15)
plt.xlim(min_data[0] - 2, max_data[0] + 2)
plt.ylim(min_data[1] - 2, max_data[1] + 2)
plt.show()