# Text Feature Representation with Scikit-Learn

This repository demonstrates various text feature representation techniques using the Scikit-Learn library in Python. The techniques include One-Hot Encoding, Label Encoding, Bag of Words (BoW), N-Grams, and TF-IDF. The provided code illustrates how to preprocess text data and convert it into numerical representations suitable for machine learning models.

## Requirements

Make sure you have Python installed on your machine. You can install the required libraries using pip:

```bash
pip install scikit-learn numpy
```

# Usage

The code is organized into several sections demonstrating different text processing techniques. Below is a brief overview of each section:

## 1. One-Hot Encoding and Label Encoding
We encode our corpus as a one-hot numeric array using Scikit-Learn's OneHotEncoder and demonstrate label encoding.

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

S1 = 'dog bites man'
S2 = 'man bites dog'
S3 = 'dog eats meat'
S4 = 'man eats food'

data = [S1.split(), S2.split(), S3.split(), S4.split()]
values = data[0] + data[1] + data[2] + data[3]
print("The data: ", values)

# Label Encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print("Label Encoded:", integer_encoded)

# One-Hot Encoding
onehot_encoder = OneHotEncoder()
onehot_encoded = onehot_encoder.fit_transform(data).toarray()
print("Onehot Encoded Matrix:\n", onehot_encoded)
```
## 2. Bag of Words (BoW)
The Bag of Words model converts a collection of documents into a matrix of token counts.

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = ["Dog bites man.", "Man bites dog.", "Dog eats meat.", "Man eats food."]
processed_docs = [doc.lower().replace(".", "") for doc in documents]

# Build a BOW representation for the corpus
count_vect = CountVectorizer()
bow_rep = count_vect.fit_transform(processed_docs)

print("Our vocabulary: ", count_vect.vocabulary_)
print("BoW representation for 'dog bites man': ", bow_rep[0].toarray())
```

## 3. Bag of N-Grams
N-Grams extend the Bag of Words approach by considering sequences of words.

```python
count_vect = CountVectorizer(ngram_range=(1, 3))
bow_rep = count_vect.fit_transform(processed_docs)

print("Our vocabulary: ", count_vect.vocabulary_)
```

## 4. TF-IDF
The TF-IDF (Term Frequency-Inverse Document Frequency) model helps to evaluate how important a word is to a document in a collection.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
bow_rep_tfidf = tfidf.fit_transform(processed_docs)

print("IDF for all words in the vocabulary", tfidf.idf_)
print("TFIDF representation for all documents in our corpus\n", bow_rep_tfidf.toarray())
```
# Conclusion

This project demonstrates various techniques for text feature representation, showcasing how to preprocess text and convert it into numerical formats suitable for machine learning applications.
