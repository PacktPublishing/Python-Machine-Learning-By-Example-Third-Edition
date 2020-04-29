'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 5 Predicting Online Ads Click-through with Logistic Regression
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''


import numpy as np
import pandas as pd
n_rows = 100000
df = pd.read_csv("train", nrows=n_rows)

X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values


X_train = X
Y_train = Y


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)


# Feature selection with random forest

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1)
random_forest.fit(X_train_enc.toarray(), Y_train)




feature_imp = random_forest.feature_importances_
print(feature_imp)

# bottom 10 weights and the corresponding 10 least important features
feature_names = enc.get_feature_names()
print(np.sort(feature_imp)[:10])
bottom_10 = np.argsort(feature_imp)[:10]
print('10 least important features are:\n', feature_names[bottom_10])

# top 10 weights and the corresponding 10 most important features
print(np.sort(feature_imp)[-10:])
top_10 = np.argsort(feature_imp)[-10:]
print('10 most important features are:\n', feature_names[top_10])

