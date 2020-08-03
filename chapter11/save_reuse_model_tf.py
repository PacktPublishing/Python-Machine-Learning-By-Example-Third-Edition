'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 11  Machine Learning Best Practices
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

import tensorflow as tf
from tensorflow import keras

from sklearn import datasets
cancer_data = datasets.load_breast_cancer()
X = cancer_data.data
Y = cancer_data.target


learning_rate = 0.005
n_iter = 10

tf.random.set_seed(42)

model = keras.Sequential([
    keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate))


model.fit(X, Y, epochs=n_iter)

model.summary()


path = './model_tf'
model.save(path)


new_model = tf.keras.models.load_model(path)

new_model.summary()

