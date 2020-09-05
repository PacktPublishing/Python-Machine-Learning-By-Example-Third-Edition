'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 12 Categorizing Images of Clothing with Convolutional Neural Networks
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

import tensorflow as tf
import matplotlib.pyplot as plt


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0
test_images = test_images / 255.0


from tensorflow.keras.preprocessing.image import load_img

def generate_plot_pics(datagen, original_img, save_prefix):
    folder = 'aug_images'
    i = 0
    for batch in datagen.flow(original_img.reshape((-1, 28, 28, 1)),
                              batch_size=1,
                              save_to_dir=folder,
                              save_prefix=save_prefix,
                              save_format='jpeg'):
        i += 1
        if i > 2:
            break
    plt.subplot(2, 2, 1, xticks=[],yticks=[])
    plt.imshow(original_img)
    plt.title("Original")
    i = 1
    for file in os.listdir(folder):
        if file.startswith(save_prefix):
            plt.subplot(2, 2, i + 1, xticks=[],yticks=[])
            aug_img = load_img(folder + "/" + file)
            plt.imshow(aug_img)
            plt.title(f"Augmented {i}")
            i += 1
    plt.show()





from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(horizontal_flip=True)
generate_plot_pics(datagen, train_images[0], 'horizontal_flip')

datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True)
generate_plot_pics(datagen, train_images[0], 'hv_flip')

datagen = ImageDataGenerator(rotation_range=30)
generate_plot_pics(datagen, train_images[0], 'rotation')

datagen = ImageDataGenerator(width_shift_range=8)
generate_plot_pics(datagen, train_images[0], 'width_shift')

datagen = ImageDataGenerator(width_shift_range=8,
                             height_shift_range=8)
generate_plot_pics(datagen, train_images[0], 'width_height_shift')




X_train = train_images.reshape((train_images.shape[0], 28, 28, 1))
X_test = test_images.reshape((test_images.shape[0], 28, 28, 1))

n_small = 500
X_train = X_train[:n_small]
train_labels = train_labels[:n_small]

print(X_train.shape)

tf.random.set_seed(42)


from tensorflow.keras import datasets, layers, models, losses
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])


model.fit(X_train, train_labels, validation_data=(X_test, test_labels), epochs=20, batch_size=40)

test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=2)
print('Accuracy on test set:', test_acc)




datagen = ImageDataGenerator(height_shift_range=3,
                             horizontal_flip=True
                             )

model_aug = tf.keras.models.clone_model(model)

model_aug.compile(optimizer='adam',
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

train_generator = datagen.flow(X_train, train_labels, seed=42, batch_size=40)
model_aug.fit(train_generator, epochs=50, validation_data=(X_test, test_labels))

test_loss, test_acc = model_aug.evaluate(X_test, test_labels, verbose=2)
print('Accuracy on test set:', test_acc)


