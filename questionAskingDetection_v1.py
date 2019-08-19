
# coding: utf-8

# # Question Asking Detection

# ### Synopsis

# This is the Question Asking Detection module. It is an external script contibuting to the main class of the combined component. It loads 1 main model, mainly the NaiveBayesClassifier which uses a dialogue act festure extractor trained on nps chats. The imports necessary for the correct execution of this script are listed under the Imports header block. The following parts load the models, consecutively. The final part of the code, listed under the Main Function header, is responsible for checkQuestionStatement function. This funcion takes only a response as input. It is then passed onto a nested function, the dialogue act feature extractor which extracts the features and returns the features for each response. These features are then passed through the classifier and the a data frame of classificatied responses is created, where each sentence has a type.

# ### Imports

# In[3]:

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


# ### Main Function - Loading

# In[6]:

#**************************************************************************************/
#    Title: Natural Language Processing with Python
#    Author: Steven Bird, Ewan Klein and Edward Loper
#    Date: 2019
#    Code version: 3.0
#    Availability: http://www.nltk.org/book/ch06.html
#**************************************************************************************/

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


# In[7]:

class checkQuestion():
    
    def checkQuestionStatement(response):
        responseUtterances = tokenizer.tokenize(response)

        classifiedUtterances = {}    
        classifiedUtterances[response] = (classifier.classify(dialogue_act_features(response)))

        classifiedUtterances_df = pd.DataFrame.from_dict(classifiedUtterances, orient='index', columns=['Category'])
        return(classifiedUtterances[response])

