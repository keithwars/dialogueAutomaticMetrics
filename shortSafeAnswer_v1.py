
# coding: utf-8

# # Short Safe Answer

# ### Synopsis

# This is the Short Safe Answer module. It is an external script contibuting to the main class of the combined component. It loads 1 main model, mainly the nlp model from the en_core_web_sm which is a language package import for various nlp models. The imports necessary for the correct execution of this script are listed under the Imports header block. The following parts load the model, and initialise the functions. The final part of the code, listed under the Main Function header, is responsible for checking for short-safe answers. This function takes a statement and a response as input. It is then passed onto a nested function, the named entity recognition checker, which extracts the NER features and returns scores for each response-statement combination. This score is backpropogated to the main function and outputted to the main component.

# ### Imports

# In[2]:

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from tqdm import tqdm_notebook as tqdm
from spacy import displacy
from collections import Counter
import logging
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
import nltk.corpus
import nltk.data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import nltk
import spacy
import pprint
import en_core_web_sm
nlp = en_core_web_sm.load()


# ### Hyperparameters

# In[3]:

short_sent_threshold = 6
ner_status = ''
ner_score = 0


# ### Named Entity Recognition

# In[4]:

def check_NER(statement, response):
    
    statement_token = nlp(statement)
    statement_token = ([(X.text, X.label_) for X in statement_token.ents])
    statement_ner = []
    
    for token in statement_token:
        statement_ner.append(token[1])
        
    response_token = nlp(response)
    response_token = ([(X.text, X.label_) for X in response_token.ents])
    response_ner = []
    
    for token in response_token:
        response_ner.append(token[1])
        
    if not (response_ner):
#             ner_status = ("Response is not informative. Does not contain any NER.")
            ner_score = 0
            if not (statement_ner):
#                 ner_status = ("No Named Entity Recognition found in statement.")
                ner_score = 1
            return(ner_score)
            
    for token in response_token:
        if (token[1]) in statement_ner:
            ner_status = ('Response is informative. Contains NER.')
            ner_score = 1
        elif (token[1]) not in statement_ner:
#             ner_status = ('Response is not informative. Does not contain NER.')
            if not (statement_ner):
                ner_score = 1
#                 ner_status = ("No Named Entity Recognition found in statement.")
                
    return(ner_score)


# ### Main Function

# In[5]:

### Hyperparameters
class checkShortSafeResponse():
    
    def checkShortSafe(statement, response):
        ner_score=1
        response_len = len(response.split())
        statement_len = len(statement.split())
        if (statement_len > short_sent_threshold):
            if (response_len <= short_sent_threshold):
                return (check_NER(statement,response))
            else: 
    #             ner_score = 100
    #             return ("Chatbot's response is above the threshold.", ner_score)
                return (ner_score)

        else:
    #         ner_score = 60
    #         return ("User's statement is below the threshold.", ner_score)
            return (ner_score)

