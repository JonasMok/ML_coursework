# ML_cousework
This is a repository for Machine learning coursework from Cardiff University. 

### Getting started
This repository contains a machine learning code able to do a sentiment analysis 
of movie reviews. This document outlines how to run this code. 

### Dataset 

The core dataset contains 25,000 reviews split into train, development
and test sets. The overall distribution of labels is roughly balanced.

### Files

There are two files: 
- Data set (Each folder contains files with negative and positive reviews one review per line)
- One Python code and;

Reference paper
```

@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```

### Installing packages

'''

import numpy as np
import pandas as pd
import re
import nltk
import sklearn
import nltk
'''

#Data Preprocessing and Feature Engineering
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn import svm
from nltk import sent_tokenize, word_tokenize
from sklearn.svm import LinearSVC
from sklearn.svm import SVC



df_pos = pd.read_csv('imdb_train_pos.txt', sep='\n', header = None)
df_neg = pd.read_csv('imdb_train_neg.txt', sep='\n', header = None)


df_pos[1] = 1
df_neg[1] = 0
#print(df_pos.head())
#print(df_neg.head())

df_train = pd.concat([df_pos,df_neg])
df_train = pd.concat([df_pos,df_neg])
df_train.columns = ['text','label']
#print(df_train.head())
#print(df_train.tail())

df_pos_test = pd.read_csv('imdb_test_pos.txt', sep='\n', header = None)
df_neg_test = pd.read_csv('imdb_test_neg.txt', sep='\n', header = None)


df_pos_test[1] = 1
df_neg_test[1] = 0
#print(df_pos.head())
#print(df_neg.head())

df_test = pd.concat([df_pos,df_neg])
df_test = pd.concat([df_pos,df_neg])
df_test.columns = ['text','label']

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
class LemmaTokenizer:
 def __init__(self):
     self.wnl = WordNetLemmatizer()
 def __call__(self, doc):
     return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


my_stopwords = ENGLISH_STOP_WORDS.union(['@','<br />'])


vect = CountVectorizer(max_features=1000, ngram_range = (1,3), stop_words = my_stopwords, tokenizer=LemmaTokenizer())
X = vect.fit_transform(df_train.text)
X_test = vect.fit_transform(df_test.text)

#Transform to an array
my_array = X.toarray()
my_array_test = X_test.toarray()
#Transform back to a dataframe, assign column names
X_df = pd.DataFrame(my_array, columns=vect.get_feature_names())
X_df_test = pd.DataFrame(my_array_test, columns=vect.get_feature_names())


svm_review = sklearn.svm.SVC(kernel="linear",gamma='auto')
model = svm_review.fit(X_df,df_train.label)
predictions  = model.predict(X_df_test)


print(confusion_matrix(df_test.label,predictions))
print(classification_report(df_test.label,predictions))
print(accuracy_score(df_test.label, predictions))

'''
