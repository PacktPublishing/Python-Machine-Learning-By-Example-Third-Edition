'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 9  Mining the 20 Newsgroups Dataset with Text Analysis Techniques
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''

from sklearn.datasets import fetch_20newsgroups


groups = fetch_20newsgroups()



from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer(stop_words="english",max_features=500)
data_count = count_vector.fit_transform(groups.data)

print(count_vector.get_feature_names())

data_count.toarray()[0]




data_cleaned = []
for doc in groups.data:
    doc_cleaned = ' '.join(word for word in doc.split() if word.isalpha())
    data_cleaned.append(doc_cleaned)


from sklearn.feature_extraction import stop_words
print(stop_words.ENGLISH_STOP_WORDS)


from nltk.corpus import names
all_names = set(names.words())


count_vector_sw = CountVectorizer(stop_words="english", max_features=500)


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

data_cleaned = []

for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)


data_cleaned_count = count_vector_sw.fit_transform(data_cleaned)

print(count_vector_sw.get_feature_names())





