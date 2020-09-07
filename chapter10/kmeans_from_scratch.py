'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 10  Discovering Underlying Topics in the Newsgroups Dataset with Clustering and Topic Modeling
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''



from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

import numpy as np
from matplotlib import pyplot as plt
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

k = 3
random_index = np.random.choice(range(len(X)), k)
centroids = X[random_index]


def visualize_centroids(X, centroids):
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
    plt.show()


visualize_centroids(X, centroids)


def dist(a, b):
    return np.linalg.norm(a - b, axis=1)

def assign_cluster(x, centroids):
    distances = dist(x, centroids)
    cluster = np.argmin(distances)
    return cluster

def update_centroids(X, centroids, clusters):
    for i in range(k):
        cluster_i = np.where(clusters == i)
        centroids[i] = np.mean(X[cluster_i], axis=0)


clusters = np.zeros(len(X))

tol = 0.0001
max_iter = 100

iter = 0
centroids_diff = 100000

from copy import deepcopy
while iter < max_iter and centroids_diff > tol:
    for i in range(len(X)):
        clusters[i] = assign_cluster(X[i], centroids)
    centroids_prev = deepcopy(centroids)
    update_centroids(X, centroids, clusters)
    iter += 1
    centroids_diff = np.linalg.norm(centroids - centroids_prev)
    print('Iteration:', str(iter))
    print('Centroids:\n', centroids)
    print('Centroids move: {:5.4f}'.format(centroids_diff))
    visualize_centroids(X, centroids)


for i in range(k):
    cluster_i = np.where(clusters == i)
    plt.scatter(X[cluster_i, 0], X[cluster_i, 1])
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
plt.show()



