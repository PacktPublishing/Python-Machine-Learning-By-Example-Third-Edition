'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 9  Mining the 20 Newsgroups Dataset with Text Analysis Techniques
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


categories_3 = ['talk.religion.misc', 'comp.graphics', 'sci.space']

groups_3 = fetch_20newsgroups(categories=categories_3)


from nltk.corpus import names
all_names = set(names.words())


count_vector_sw = CountVectorizer(stop_words="english", max_features=500)


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

data_cleaned = []

for doc in groups_3.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)


data_cleaned_count_3 = count_vector_sw.fit_transform(data_cleaned)




from sklearn.manifold import TSNE


tsne_model = TSNE(n_components=2,  perplexity=40, random_state=42, learning_rate=500)


data_tsne = tsne_model.fit_transform(data_cleaned_count_3.toarray())


import matplotlib.pyplot as plt
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=groups_3.target)

plt.show()






categories_5 = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                'comp.windows.x']
groups_5 = fetch_20newsgroups(categories=categories_5)

count_vector_sw = CountVectorizer(stop_words="english", max_features=500)

data_cleaned = []

for doc in groups_5.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)

data_cleaned_count_5 = count_vector_sw.fit_transform(data_cleaned)

data_tsne = tsne_model.fit_transform(data_cleaned_count_5.toarray())

plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=groups_5.target)

plt.show()