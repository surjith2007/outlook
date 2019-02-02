#!/usr/bin/python
# -*- coding: utf-8 -*-

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
import tweets_reader as tr

def clean(model,index) :
  index = index + 1
  dirty = ["+",".","*"]
  while model.find("0.") != -1 :
    prob = model[model.find("0."):model.find("0.")+6]
    model = string.replace(model,prob,"")
  for element in dirty :    
    model = string.replace(model,element,"")
  model = "Topic #"+str(index)+": "+model
  return model

all_tweets = tr.get_all_tweets()  
doc_set = []
count = 0
for index in all_tweets :
  doc_set.append(all_tweets[index]["clean_tweet"])
  count = count + 1

tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()  
texts = []

# loop through
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    stopped_tokens = [i for i in tokens if not i.lower() in en_stop]
    

    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    texts.append(stemmed_tokens)


dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

num_topics = 10
num_words = 10
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary)

i = 0
while i < num_topics:
  print(clean(ldamodel.print_topics(num_topics=num_topics,num_words=num_words)[i][1],i))
  i = i+1