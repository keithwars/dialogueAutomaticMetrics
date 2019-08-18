
# coding: utf-8

# In[18]:

import logging
# logging.basicConfig(level=logging.INFO)
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import csv
import nltk.corpus
import nltk.data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

posts = nltk.corpus.nps_chat.xml_posts()[:10000]
featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# In[27]:

class checkQuestion():

    def checkQuestionStatement(response):
        responseUtterances = tokenizer.tokenize(response)

        classifiedUtterances = {}    
        classifiedUtterances[response] = (classifier.classify(dialogue_act_features(response)))

        classifiedUtterances_df = pd.DataFrame.from_dict(classifiedUtterances, orient='index', columns=['Category'])
        return(classifiedUtterances[response])

