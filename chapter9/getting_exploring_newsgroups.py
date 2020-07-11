'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 9  Mining the 20 Newsgroups Dataset with Text Analysis Techniques
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''


from sklearn.datasets import fetch_20newsgroups


groups = fetch_20newsgroups()
groups.keys()
groups['target_names']
groups.target


import numpy as np
np.unique(groups.target)



import seaborn as sns
sns.distplot(groups.target)
import matplotlib.pyplot as plt
plt.show()


groups.data[0]
groups.target[0]
groups.target_names[groups.target[0]]



