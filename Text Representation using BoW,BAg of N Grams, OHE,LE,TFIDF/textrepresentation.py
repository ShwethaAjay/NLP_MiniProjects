"""
One-hot encoding using scikit -learn
We encode our corpus as a one-hot numeric array using scikit-learn's OneHotEncoder. We will demostrate:
1. One Hot Encoding: In one-hot encoding, each word w in corpus vocabulary is given a unique integer id wid that is between 1 and |V|, where V is the set of corpus vocab. 
                     Each word is then represented by a V-dimensional binary vector of 0s and 1s.
2. Label Encoding: In Label Encoding, each word w in our corpus is converted into a numeric value between 0 and n-1 (where n refers to number of unique words in our corpus).
"""

S1 = 'dog bites man'
S2 = 'man bites dog'
S3 = 'dog eats meat'
S4 = 'man eats food'
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data = [S1.split(), S2.split(), S3.split(), S4.split()]
values = data[0]+data[1]+data[2]+data[3]
print(values)
print("The data: ",values)

#Label Encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print("Label Encoded:",integer_encoded)

#One-Hot Encoding
onehot_encoder = OneHotEncoder()
onehot_encoded = onehot_encoder.fit_transform(data).toarray()
print("Onehot Encoded Matrix:\n",onehot_encoded)

Bag of Words (BOW)
documents = ["Dog bites man.", "Man bites dog.", "Dog eats meat.", "Man eats food."] 
processed_docs = [doc.lower().replace(".","") for doc in documents]
processed_docs

from sklearn.feature_extraction.text import CountVectorizer
#look at the documents list
print("Our corpus: ", processed_docs)
count_vect = CountVectorizer()
#Build a BOW representation for the corpus
bow_rep = count_vect.fit_transform(processed_docs)

#Look at the vocabulary mapping
print("Our vocabulary: ", count_vect.vocabulary_)

#see the BOW rep for first 2 documents
print("BoW representation for 'dog bites man': ", bow_rep[0].toarray())
print("BoW representation for 'man bites dog: ",bow_rep[1].toarray())

#Get the representation using this vocabulary, for a new text
temp = count_vect.transform(["dog and dog are friends"])
print("Bow representation for 'dog and dog are friends':", temp.toarray())

"""
Bag of N-Grams
One hot encoding, BoW and TF-IDF treat words as independent units. There is no notion of phrases or word ordering. Bag of Ngrams (BoN) approach tries to remedy this. It does so by breaking text into chunks of n countigous words/tokens. This can help us capture some context, which earlier approaches could not do. Let us see how it works using the same toy corpus we used in earlier examples.
CountVectorizer, which we used for BoW, can be used for getting a Bag of N-grams representation as well, using its ngram_range argument.
"""

from sklearn.feature_extraction.text import CountVectorizer

TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
bow_rep_tfidf = tfidf.fit_transform(processed_docs)

#IDF for all words in the vocabulary
print("IDF for all words in the vocabulary",tfidf.idf_)
print("-"*10)
#All words in the vocabulary.
print("All words in the vocabulary",tfidf.get_feature_names())
print("-"*10)

#TFIDF representation for all documents in our corpus 
print("TFIDF representation for all documents in our corpus\n",bow_rep_tfidf.toarray()) 
print("-"*10)

temp = tfidf.transform(["dog and man are friends"])
print("Tfidf representation for 'dog and man are friends':\n", temp.toarray())

#Ngram vectorization example with count vectorizer and uni, bi, trigrams
count_vect = CountVectorizer(ngram_range=(1,3))

#Build a BOW representation for the corpus
bow_rep = count_vect.fit_transform(processed_docs)

#Look at the vocabulary mapping
print("Our vocabulary: ", count_vect.vocabulary_)

#see the BOW rep for first 2 documents
print("BoW representation for 'dog bites man': ", bow_rep[0].toarray())
print("BoW representation for 'man bites dog: ",bow_rep[1].toarray())

#Get the representation using this vocabulary, for a new text
temp = count_vect.transform(["dog and dog are friends"])

print("Bow representation for 'dog and dog are friends':", temp.toarray())

"""
Tasks: 

1. Take raw text to create your corpus
2. Do suitable data cleaning and preprocessing
3. Perform feature representation using OHE,LE,BOW,BON-grams and TF-IDF.
4. Conclude your result.
"""

#One-hot encoding using scikit-learn
s1 = 'Prospero uses magic to conjure a storm'
s2 = 'Prospero’s slave Caliban plots to rid himself of his master'
s3 = 'The King’s young son Ferdinand falls in love with Prospero’s daughter Miranda'
s4 = 'Their celebrations are cut short'
#S5 = 'The families are reunited'
#S6 = 'Prospero grants Ariel his freedom'
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np


data = [s1.split(), s2.split(), s3.split(), s4.split()]
values = data[0]+data[1]+data[2]+data[3]
print(values)
print("The data: ",values)

#Label Encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print("Label Encoded:",integer_encoded)
onehot_encoded = []
#One-Hot Encoding
onehot_encoder = OneHotEncoder()
for i in data:
  i = np.array(i)
  i= i.reshape(-1,1)
  onehot_encoded.append(onehot_encoder.fit_transform(i).toarray())
print("Onehot Encoded Matrix:\n",onehot_encoded)

#Bag of Words (BOW)
documents = ["Prospero uses magic to conjure a storm.", 
             "Prospero’s slave Caliban plots to rid himself of his master.", 
             "The King’s young son Ferdinand falls in love with Prospero’s daughter Miranda.", 
             "Their celebrations are cut short."] 
processed_docs = [doc.lower().replace(".","") for doc in documents]
processed_docs
from sklearn.feature_extraction.text import CountVectorizer
#look at the documents list
print("Our corpus: ", processed_docs)
count_vect = CountVectorizer()
#Build a BOW representation for the corpus
bow_rep = count_vect.fit_transform(processed_docs)

#Look at the vocabulary mapping
print("Our vocabulary: ", count_vect.vocabulary_)

s1 = 'Prospero uses magic to conjure a storm'
s2 = 'Prospero’s slave Caliban plots to rid himself of his master'
s3 = 'The King’s young son Ferdinand falls in love with Prospero’s daughter Miranda'
s4 = 'Their celebrations are cut short'
#S5 = 'The families are reunited'
#S6 = 'Prospero grants Ariel his freedom'

#see the BOW rep for first 2 documents
print("BoW representation for 'Prospero uses magic to conjure a storm': ", bow_rep[0].toarray())
print("BoW representation for 'Prospero’s slave Caliban plots to rid himself of his master': ",bow_rep[1].toarray())

#Get the representation using this vocabulary, for a new text
temp = count_vect.transform(["Prospero and Prospero are friends"])
print("Bow representation for 'Prospero and Prospero are friends':", temp.toarray())

#Bag of N-Grams
from sklearn.feature_extraction.text import CountVectorizer

#Ngram vectorization example with count vectorizer and uni, bi, trigrams
count_vect = CountVectorizer(ngram_range=(1,3))

#Build a BOW representation for the corpus
bow_rep = count_vect.fit_transform(processed_docs)

#Look at the vocabulary mapping
print("Our vocabulary: ", count_vect.vocabulary_)

#see the BOW rep for first 2 documents
print("BoW representation for 'dog bites man': ", bow_rep[0].toarray())
print("BoW representation for 'man bites dog: ",bow_rep[1].toarray())

#Get the representation using this vocabulary, for a new text
temp = count_vect.transform(["dog and dog are friends"])

print("Bow representation for 'dog and dog are friends':", temp.toarray())

#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
bow_rep_tfidf = tfidf.fit_transform(processed_docs)

#IDF for all words in the vocabulary
print("IDF for all words in the vocabulary",tfidf.idf_)
print("-"*10)
#All words in the vocabulary.
print("All words in the vocabulary",tfidf.get_feature_names())
print("-"*10)

#TFIDF representation for all documents in our corpus 
print("TFIDF representation for all documents in our corpus\n",bow_rep_tfidf.toarray()) 
print("-"*10)

temp = tfidf.transform(["dog and man are friends"])
print("Tfidf representation for 'dog and man are friends':\n", temp.toarray())

