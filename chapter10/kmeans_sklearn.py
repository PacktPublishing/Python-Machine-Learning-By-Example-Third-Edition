'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 10  Discovering Underlying Topics in the Newsgroups Dataset with Clustering and Topic Modeling
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''



from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

from matplotlib import pyplot as plt

k = 3
from sklearn.cluster import KMeans
kmeans_sk = KMeans(n_clusters=3, random_state=42)
kmeans_sk.fit(X)
clusters_sk = kmeans_sk.labels_
centroids_sk = kmeans_sk.cluster_centers_


plt.scatter(X[:, 0], X[:, 1], c=clusters_sk)
plt.scatter(centroids_sk[:, 0], centroids_sk[:, 1], marker='*', s=200, c='#050505')
plt.show()
