'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 11  Machine Learning Best Practices
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

import gensim.downloader as api

model = api.load("glove-twitter-25")


vector = model.wv['computer']
print('Word computer is embedded into:\n', vector)

similar_words = model.most_similar("computer")
print('Top ten words most contextually relevant to computer:\n', similar_words)



doc_sample = ['i', 'love', 'reading', 'python', 'machine', 'learning', 'by', 'example']

import numpy as np
doc_vector = np.mean([model.wv[word] for word in doc_sample], axis=0)
print('The document sample is embedded into:\n', doc_vector)
