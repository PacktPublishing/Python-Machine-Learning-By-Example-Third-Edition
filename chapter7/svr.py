'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 7 Predicting Stock Price with Regression Algorithms
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

from sklearn import datasets
boston = datasets.load_boston()

num_test = 10    # the last 10 samples as testing set
X_train = boston.data[:-num_test, :]
y_train = boston.target[:-num_test]
X_test = boston.data[-num_test:, :]
y_test = boston.target[-num_test:]

from sklearn.svm import SVR
regressor = SVR(C=0.1, epsilon=0.02, kernel='linear')

regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(predictions)
