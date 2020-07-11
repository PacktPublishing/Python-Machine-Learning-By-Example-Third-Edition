'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 9  Mining the 20 Newsgroups Dataset with Text Analysis Techniques
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''


from nltk.corpus import names

print(names.words()[:10])

print(len(names.words()))



from nltk.tokenize import word_tokenize
sent = '''I am reading a book.
          It is Python Machine Learning By Example,
          2nd edition.'''

print(word_tokenize(sent))


sent2 = 'I have been to U.K. and U.S.A.'
print(word_tokenize(sent2))



import spacy

nlp = spacy.load('en_core_web_sm')
tokens2 = nlp(sent2)

print([token.text for token in tokens2])


from nltk.tokenize import sent_tokenize
print(sent_tokenize(sent))


import nltk
tokens = word_tokenize(sent)
print(nltk.pos_tag(tokens))
nltk.help.upenn_tagset('PRP')
nltk.help.upenn_tagset('VBP')



print([(token.text, token.pos_) for token in tokens2])



tokens3 = nlp('The book written by Hayden Liu in 2020 was sold at $30 in America')
print([(token_ent.text, token_ent.label_) for token_ent in tokens3.ents])



from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
porter_stemmer.stem('machines')
porter_stemmer.stem('learning')


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('machines')