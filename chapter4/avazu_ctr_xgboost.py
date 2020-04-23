'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 4 Predicting Online Ads Click-through with Tree-Based Algorithms
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

import pandas as pd
n_rows = 300000
df = pd.read_csv("train", nrows=n_rows)


X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values
Y = df['click'].values

n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)


import xgboost as xgb
model = xgb.XGBClassifier(learning_rate=0.1, max_depth=10, n_estimators=1000)

model.fit(X_train_enc, Y_train)
pos_prob = model.predict_proba(X_test_enc)[:, 1]


from sklearn.metrics import roc_auc_score
print(f'The ROC AUC on testing set is: {roc_auc_score(Y_test, pos_prob):.3f}')

