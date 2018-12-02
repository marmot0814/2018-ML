import numpy as np
import pandas as pd
import time
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sn


class Kmeans:

    def __init__(self, data, k=3):
        self.i = 0
        self.n = data.shape[0]
        self.k = k
        self.color = [
            'purple', 'g', 'b', 'y', 'pink', 'm', 'c', 'black', 'lime',
            'orange', 'brown', 'grey', '#abd03d', '#298834'
        ]

        # shuffle data
        indices = np.random.permutation(self.n)
        self.data = np.array(data[indices])

        # initial plt
        self.fig = plt.figure()
        plt.ion()
        plt.show()
        if data.shape[1] == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)
        self.ax.axis("equal")
        # initial centers
        self.centers = np.array([self.data[i] for i in range(self.k)])

        # inital clusters
        self.clusters = [np.array([]) for x in range(self.k)]

        # iterator
        iter_cnt = 0
        while True:
            iter_cnt += 1
            stop = self.iterator()
            #time.sleep(0.5)
            if self.data.shape[1] == 2:
                self.display2D()
            else:
                self.display3D()
            plt.title("iterator " + str(iter_cnt))
            plt.pause(0.01)

            if stop:
                break
        plt.pause(1)

    def iterator(self):
        self.i += 1
        # initial
        clusters = [[] for x in range(self.k)]

        # find the cluster of the data for each one
        #print(self.distance(self.centers, self.data[0]))
        #print(self.centers)
        for i in range(self.n):
            clusters[np.argmin(self.distance(
                self.centers, self.data[i]))].append(self.data[i])

        # trans list to numpy array
        clusters = [np.array(clusters[i]) for i in range(self.k)]

        # update centers
        self.centers = np.array(
            [np.mean(clusters[i], axis=0) for i in range(self.k)])
        #print(self.centers)
        # swap old and new clusters
        self.clusters, clusters = clusters, self.clusters

        return all(
            [np.array_equal(x, y) for x, y in zip(self.clusters, clusters)])

    def distance(self, a, b):
        return np.sum((a - b)**2, axis=1)

    def display3D(self):
        self.ax.cla()
        for i in range(self.k):
            self.ax.plot([x[0] for x in self.clusters[i]],
                         [x[1] for x in self.clusters[i]],
                         [x[2] for x in self.clusters[i]],
                         marker='o',
                         linestyle='',
                         ms=4,
                         color=self.color[i])
        for i in range(self.k):
            self.ax.plot([self.centers[i][0]], [self.centers[i][1]],
                         [self.centers[i][2]],
                         marker='o',
                         linestyle='',
                         ms=4,
                         color='r')

        self.fig.canvas.draw()

    def display2D(self):
        self.ax.cla()
        for i in range(self.k):
            self.ax.plot([x[0] for x in self.clusters[i]],
                         [x[1] for x in self.clusters[i]],
                         marker='o',
                         linestyle='',
                         ms=4,
                         color=self.color[i])
        for i in range(self.k):
            self.ax.plot([self.centers[i][0]], [self.centers[i][1]],
                         marker='o',
                         linestyle='',
                         ms=4,
                         color='r')

        self.fig.canvas.draw()

    def cost(self, data):
        cost = 0
        for _data in data:
            cost += min(self.distance(self.centers, _data))
        return cost

    def accuracy(self,data,data_type):
        clusters = [[] for x in range(self.k)]
        for i in range(self.n):
            clusters[np.argmin(self.distance(
                self.centers, data[i]))].append(data_type[i])

        # confusion validation
        c_type, c_pred = [], []
        for line in clusters:
            tmp = Counter(line).most_common(3)
            c_type += line
            c_pred += [tmp[0][0] for t in range(len(line))]

        type_name = list(set(c_type))
        df_cm = pd.DataFrame(confusion_matrix(c_type,c_pred), index=type_name, columns=type_name)
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True,  fmt='g', cmap='Blues')
        plt.show(1) 

        return sum(a==b for a, b in zip(c_type,c_pred))/len(c_type)
