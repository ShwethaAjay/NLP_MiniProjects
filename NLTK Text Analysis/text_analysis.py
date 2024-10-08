import nltk
nltk.download()

nltk.download('gutenberg')
nltk.download('genesis')
nltk.download('inaugural')
nltk.download('nps_chat')
nltk.download('webtext')
nltk.download('treebank')
nltk.download('punkt')

from nltk.book import*

#Task - 1
from nltk.collocations import BigramCollocationFinder,BigramAssocMeasures
biagram_collocation = BigramCollocationFinder.from_words(text6)
biagram_collocation.nbest(BigramAssocMeasures.likelihood_ratio, 15)

from nltk.collocations import BigramCollocationFinder,BigramAssocMeasures
biagram_collocation = BigramCollocationFinder.from_words(text7)
biagram_collocation.nbest(BigramAssocMeasures.likelihood_ratio, 15)

#Task - 2
string = "Sky is blue."
print(string)

print(string + string)
print(string*3)

#Task - 3
my_sent = ["Sky","is","blue"]
sent = " ".join(my_sent)
print(sent)
print(sent.split())

#Task - 4
words = []
for i in text7:
  if i[0] == 'T':
    words.append(i)
print(sorted(words))

#Task - 5
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
words = dict()
for i in text5:
  if len(i) == 4 and i in words:
    words[i] += 1
  elif len(i) == 4 and i not in words:
    words[i] = 1
words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1],reverse=True)}
data = pd.DataFrame()
data['Word'] = words.keys()
data['Frequency'] = words.values()
data

plt.figure(figsize=(10,10))
sns.countplot(x='Frequency',data=data)

#Task - 6
for i in text6:
  if i.islower():
    print(i)
  else:
    continue
