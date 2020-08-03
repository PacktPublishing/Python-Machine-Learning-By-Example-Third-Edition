'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 10  Discovering Underlying Topics in the Newsgroups Dataset with Clustering and Topic Modeling
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''


from sklearn.datasets import fetch_20newsgroups

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]


groups = fetch_20newsgroups(subset='all', categories=categories)

from nltk.corpus import names
all_names = set(names.words())



from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

data_cleaned = []

for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)


from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words="english", max_features=None, max_df=0.5, min_df=2)

data = count_vector.fit_transform(data_cleaned)


from sklearn.decomposition import NMF

t = 20
nmf = NMF(n_components=t, random_state=42)

nmf.fit(data)

print(nmf.components_)

terms = count_vector.get_feature_names()


for topic_idx, topic in enumerate(nmf.components_):
        print("Topic {}:" .format(topic_idx))
        print(" ".join([terms[i] for i in topic.argsort()[-10:]]))


