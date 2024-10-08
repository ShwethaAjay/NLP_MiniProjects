# Text Pre-processing

This repository demonstrates various text processing and analysis techniques using the NLTK and SpaCy libraries. The main objective is to preprocess a given text corpus for further analysis.

## Requirements

Make sure you have Python installed on your machine. You can install the necessary libraries using pip:

```bash
pip install nltk spacy matplotlib wordcloud
```

# Usage

The code is organized into several sections demonstrating different text processing techniques. Below is a brief overview of each section:

## 1. Initial Setup
Import the required libraries and set up the corpus.

```python
import nltk
import string
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from spacy.lang.en import English
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')

corpus = '''Your text corpus goes here.'''
```

## 2. Text Normalization
Convert the text to lowercase and remove punctuation.

```python
corpus = corpus.lower()
corpus1 = corpus.translate(str.maketrans("", "", string.punctuation))
```
## 3. Removing Stop Words
Filter out common English stop words.

```python
stop_words1 = set(stopwords.words('english'))
corpus_tokenized = corpus1.split()
corpus_tokenized_no_stopwords = [i for i in corpus_tokenized if i not in stop_words1]
```

## 4. Stemming
Apply stemming to reduce words to their base form.

```python
stemmer = PorterStemmer()
for i in corpus_tokenized_no_stopwords:
    print(stemmer.stem(i), end=" ")
```

## 5. Lemmatization
Apply lemmatization to convert words to their lemma.

```python
lem = WordNetLemmatizer()
for word in corpus_tokenized_no_stopwords:
    print(lem.lemmatize(word), end=" ")
```

6. POS Tagging
Perform part-of-speech tagging using both NLTK and SpaCy.

```python
print("POS Tagging using NLTK:")
print(nltk.pos_tag(word_tokenize(corpus)))

sp = English()
tokenizer = sp.tokenizer
sentence = sp(corpus)
for word in sentence:
    print(word.text, word.pos_)
```
