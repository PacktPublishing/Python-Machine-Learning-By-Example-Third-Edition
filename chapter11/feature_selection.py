'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 11  Machine Learning Best Practices
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''


import numpy as np
from sklearn.datasets import load_digits
dataset = load_digits()
X, y = dataset.data, dataset.target
print(X.shape)

# Estimate accuracy on the original data set
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
classifier = SVC(gamma=0.005, random_state=42)
score = cross_val_score(classifier, X, y).mean()
print(f'Score with the original data set: {score:.2f}')


# Feature selection with random forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', n_jobs=-1, random_state=42)
random_forest.fit(X, y)

# Sort features based on their importancies
feature_sorted = np.argsort(random_forest.feature_importances_)

# Select different number of top features
K = [10, 15, 25, 35, 45]
for k in K:
    top_K_features = feature_sorted[-k:]
    X_k_selected = X[:, top_K_features]
    # Estimate accuracy on the data set with k selected features
    classifier = SVC(gamma=0.005)
    score_k_features = cross_val_score(classifier, X_k_selected, y).mean()
    print(f'Score with the data set of top {k} features: {score_k_features:.2f}')

