'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 13 Making Predictions with Sequences using Recurrent Neural Networks
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers

import numpy as np

from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping


training_file = 'warpeace_input.txt'

raw_text = open(training_file, 'r').read()
raw_text = raw_text.lower()
print(raw_text[:200])

all_words = raw_text.split()
unique_words = list(set(all_words))
print(f'Number of unique words: {len(unique_words)}')

n_chars = len(raw_text)
print(f'Total characters: {n_chars}')
chars = sorted(list(set(raw_text)))
n_vocab = len(chars)
print(f'Total vocabulary (unique characters): {n_vocab}')
print(chars)



index_to_char = dict((i, c) for i, c in enumerate(chars))
char_to_index = dict((c, i) for i, c in enumerate(chars))
print(char_to_index)


seq_length = 160
n_seq = int(n_chars / seq_length)

X = np.zeros((n_seq, seq_length, n_vocab))
Y = np.zeros((n_seq, seq_length, n_vocab))

for i in range(n_seq):
	x_sequence = raw_text[i * seq_length : (i + 1) * seq_length]
	x_sequence_ohe = np.zeros((seq_length, n_vocab))
	for j in range(seq_length):
		char = x_sequence[j]
		index = char_to_index[char]
		x_sequence_ohe[j][index] = 1.
	X[i] = x_sequence_ohe
	y_sequence = raw_text[i * seq_length + 1 : (i + 1) * seq_length + 1]
	y_sequence_ohe = np.zeros((seq_length, n_vocab))
	for j in range(seq_length):
		char = y_sequence[j]
		index = char_to_index[char]
		y_sequence_ohe[j][index] = 1.
	Y[i] = y_sequence_ohe

print(X.shape)
print(Y.shape)



batch_size = 100
hidden_units = 700
n_epoch= 300
dropout = 0.4

tf.random.set_seed(42)

model = models.Sequential()
model.add(layers.LSTM(hidden_units, input_shape=(None, n_vocab), return_sequences=True, dropout=dropout))
model.add(layers.LSTM(hidden_units, return_sequences=True, dropout=dropout))

model.add(layers.TimeDistributed(layers.Dense(n_vocab, activation='softmax')))

optimizer = optimizers.RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)


print(model.summary())




def generate_text(model, gen_length, n_vocab, index_to_char):
    """
    Generating text using the RNN model
    @param model: current RNN model
    @param gen_length: number of characters we want to generate
    @param n_vocab: number of unique characters
    @param index_to_char: index to character mapping
    @return:
    """
    # Start with a randomly picked character
    index = np.random.randint(n_vocab)
    y_char = [index_to_char[index]]
    X = np.zeros((1, gen_length, n_vocab))
    for i in range(gen_length):
        X[0, i, index] = 1.
        indices = np.argmax(model.predict(X[:, max(0, i - 99):i + 1, :])[0], 1)
        index = indices[-1]
        y_char.append(index_to_char[index])
    return ''.join(y_char)


class ResultChecker(Callback):
    def __init__(self, model, N, gen_length):
        self.model = model
        self.N = N
        self.gen_length = gen_length

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.N == 0:
            result = generate_text(self.model, self.gen_length, n_vocab, index_to_char)
            print('\nMy War and Peace:\n' + result)


filepath="weights/weights_epoch_{epoch:03d}_loss_{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=50, verbose=1, mode='min')

result_checker = ResultChecker(model, 10, 500)

model.fit(X, Y, batch_size=batch_size, verbose=1, epochs=n_epoch,
                 callbacks=[result_checker, checkpoint, early_stop])

