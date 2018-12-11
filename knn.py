#!/usr/bin/env python
# coding=utf-8
import numpy as np

class KNN:
    def __init__(self, dataset, labels, k):
        self.topK = k
        self.ylabels = labels
        self.dataset = dataset

    def classfy(self, x):
        dataSetSize = self.dataset.shape[0]
        XRepresentationindatasetdimension = np.tile(x, (dataSetSize,1))
        coordinatedifference = XRepresentationindatasetdimension - self.dataset
        differencesquare = coordinatedifference**2
        absolutedistance = differencesquare.sum(axis=1)**0.5
        sorteddistance = absolutedistance.argsort() ##元素从小到大排列，提取索引
        vote2num = {}
        for i in range(self.topK):
            voteLabel = self.ylabels[sorteddistance[i]]
            vote2num[voteLabel] = vote2num.get(voteLabel,0) + 1
        sortlabel =  sorted( vote2num.items(), key=lambda x:x[1], reverse=True)
        return sortlabel[0][0]


def show():
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    simulatedata = np.array([[1,2], [2,3],[3,3],[5,1], [7,2],[19,3], [1,32], [2,7],[3,6], [5,5],[24,34],[6,8]])
    ax.scatter(simulatedata[:,0], simulatedata[:,1])
    plt.show()

def test():
    simulatedata = np.array([[1,2], [2,3],[3,3],[5,1], [7,2],[19,3], [1,32], [2,7],[3,6], [5,5],[24,34],[6,8]])
    dlabels = [1,1,1,2,2,2,3,3,3,4,4,4]
    knn = KNN(simulatedata, dlabels, 3)
    print(knn.classfy([4,4]))
