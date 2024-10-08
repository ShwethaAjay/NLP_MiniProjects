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


corpus = '''The thing that's great about this job is the time sourcing the items involves no traveling. I just look online to buy it. It's really as simple as that. While everyone else is searching for what they can sell, I sit in front of my computer and buy better stuff for less money and spend a fraction of the time doing it.
He sat across from her trying to imagine it was the first time. It wasn't. Had it been a hundred? It quite possibly could have been. Two hundred? Probably not. His mind wandered until he caught himself and again tried to imagine it was the first time.
She didn't like the food. She never did. She made the usual complaints and started the tantrum he knew was coming. But this time was different. Instead of trying to placate her and her unreasonable demands, he just stared at her and watched her meltdown without saying a word.
There weren't supposed to be dragons flying in the sky. First and foremost, dragons didn't exist. They were mythical creatures from fantasy books like unicorns. This was something that Pete knew in his heart to be true so he was having a difficult time acknowledging that there were actually fire-breathing dragons flying in the sky above him.
It had been her dream for years but Dana had failed to take any action toward making it come true. There had always been a good excuse to delay or prioritize another project. As she woke, she realized she was once again at a crossroads. Would it be another excuse or would she finally find the courage to pursue her dream? Dana rose and took her first step.
Don't forget that gifts often come with costs that go beyond their purchase price. When you purchase a child the latest smartphone, you're also committing to a monthly phone bill. When you purchase the latest gaming system, you're likely not going to be satisfied with the games that come with it for long and want to purchase new titles to play. When you buy gifts it's important to remember that some come with additional costs down the road that can be much more expensive than the initial gift itself.
'''

len(corpus)

print(corpus)

# convert the corpus into lowercase

corpus = corpus.lower()

print(corpus)

# removing punctuation marks
corpus1 = corpus.translate(str.maketrans("", "", string.punctuation))
print(corpus1)

# removing all the white spaces from the text
corpus = corpus.strip()

corpus1 = corpus.replace('\n','')

corpus1

import nltk
nltk.download('stopwords')

# stop words
stop_words1 = set(stopwords.words('english'))
print(stop_words1)

corpus_tokenized = corpus1.split()

print(corpus_tokenized)

corpus_tokenized_no_stopwords = [i for i in corpus_tokenized if i not in stop_words1]

print(corpus_tokenized_no_stopwords)

stemmer = PorterStemmer()

# stemming the corpus
for i in corpus_tokenized_no_stopwords:
    print(stemmer.stem(i),end=" ")

import nltk
nltk.download('omw-1.4')

# lemmetization on the corpus
# Lemmetizing the text
lem = WordNetLemmatizer()
for word in corpus_tokenized_no_stopwords:
    print(lem.lemmatize(word),end=" ")

import nltk
nltk.download('punkt')

# POS tagging
nltk.download('averaged_perceptron_tagger')
print("POS Tagging using NLTK:")
print(nltk.pos_tag(word_tokenize(corpus)))

#### USING SPACY

sp = English()
all_stopwords = sp.Defaults.stop_words

# Tokenizing the corpus
tokenizer = sp.tokenizer
corpus = s = re.sub(r'[^\w\s]','',corpus)
sentence = sp(corpus)
text_tokens = tokenizer(corpus)
for word in sentence:
    print(word.text)

# POS Tagging
for word in sentence:
    print(word.text,  word.pos_)

# Stemming using NLTK as SpaCy does not have a stemmer
spacy_stemmer= PorterStemmer()

print("Stemming:")

for word in corpus_tokenized_no_stopwords:
    print(stemmer.stem(word),end=" ")

# Lemmatization
for word in sentence:
    print(word.text,  word.lemma_)
