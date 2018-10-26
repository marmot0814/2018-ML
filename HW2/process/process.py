import numpy as np
import math
from matplotlib import pyplot as plt


class Kmeans:

    def __init__(self, k=3):
        self.k = k

    # data: (all data), label: (unique label only)
    def data_label(self, data, label):
        self.n = data.shape[0]
        self.fsize = data.shape[1]  # feature size
        self.label = label
        self.clusters = np.zeros(shape=(self.k,
                                        self.n))  # create array of [k][n]

        # shuffle data
        indices = np.random.permutation(self.n)
        self.data = np.array(data[indices])

        # initial clustering
        clusters = np.zeros(shape=(self.k, self.n,
                                   self.fsize))  # empty clusters
        centers = np.array(
            [self.data[i] for i in range(self.k)])  # initial center
        sizes = np.zeros(self.k, dtype=np.int)  # sizes of centers
        for i in range(1, self.n):
            # decide which cluster it belongs to
            category = 0
            cost = math.inf
            for cat_i in range(self.k):
                cost_i = self.distance(centers[cat_i], self.data[i])
                if (cost_i < cost):
                    cost = cost_i
                    category = cat_i
            # put it in the cluster
            clusters[category][sizes[category]] = self.data[i]
            sizes[category] += 1

        return [sizes, clusters]

    def iter(self, sizes_clusters):
        sizes = sizes_clusters[0]
        clusters = sizes_clusters[1]
        # print(clusters)
        # calculat means
        means = np.array(
            [np.mean(clusters[i][0:sizes[i]], axis=0) for i in range(self.k)])
        #print(means)
        new_clusters = np.zeros(shape=(self.k, self.n,
                                       self.fsize))  # empty clusters
        new_sizes = np.zeros(self.k, dtype=np.int)  # sizes of centers
        for i in range(self.n):
            # decide which cluster it belongs to
            category = 0
            cost = math.inf
            for cat_i in range(self.k):
                cost_i = self.distance(means[cat_i], self.data[i])
                if (cost_i < cost):
                    cost = cost_i
                    category = cat_i
            # put it in the cluster
            new_clusters[category][new_sizes[category]] = self.data[i]
            new_sizes[category] += 1

        # compare difference
        if (not np.array_equal(new_clusters, clusters)):
            return [new_sizes, new_clusters]
        else:
            return None

    def loss(self,):
        pass

    def distance(self, a, b):
        return np.sum((a - b)**2)

    def plot(self, sizes_clusters):
        sizes = sizes_clusters[0]
        clusters = sizes_clusters[1]
        fig, ax = plt.subplots()
        for i in range(self.k):
            ax.plot([clusters[i][j][0] for j in range(sizes[i])],
                    [clusters[i][j][1] for j in range(sizes[i])],
                    marker='o',
                    linestyle='',
                    ms=4,
                    label=['A', 'B', 'C'][i])
        ax.legend()
        fig.canvas.draw()
        plt.pause(1)  # give the gui time to process the draw events
