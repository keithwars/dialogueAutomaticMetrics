
# coding: utf-8

# In[4]:

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
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
import pprint


# In[5]:

short_sent_threshold = 3
ner_status = ''
ner_score = 0


# In[6]:

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
                ner_score = 0
            return(ner_score)
            
    for token in response_token:
        if (token[1]) in statement_ner:
#            ner_status = ('Response is informative. Contains NER.')
            ner_score = 1
        elif (token[1]) not in statement_ner:
            ner_score = 0
#             ner_status = ('Response is not informative. Does not contain NER.')
            if not (statement_ner):
                ner_score = 1
#                 ner_status = ("No Named Entity Recognition found in statement.")
                
    return(ner_score)


# In[7]:

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

