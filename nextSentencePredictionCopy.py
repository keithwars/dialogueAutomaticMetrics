
# coding: utf-8

# In[1]:

import logging
# logging.basicConfig(level=logging.INFO)

import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import csv

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from allennlp.predictors import Predictor
predictor = Predictor.from_path("https://allennlp.s3.amazonaws.com/models/decomposable-attention-elmo-2018.02.19.tar.gz")
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForNextSentencePrediction

# In[2]:

CACHE_DIR='cache/'
BERT_MODEL = 'model.tar.gz' 

# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# model = BertForNextSentencePrediction.from_pretrained('bert-large-uncased')

tokenizer = BertTokenizer.from_pretrained('vocab.txt')
model = BertForNextSentencePrediction.from_pretrained(CACHE_DIR + BERT_MODEL, cache_dir=CACHE_DIR)

model.eval()


# In[21]:

class checkNextSentenceCopy():
    
    def checkNextSentence(statement, response):
        # Load fine-tuned vocabulary from model tokenizer, using the original pre-trained bert model combined with chat data


        statement = "[CLS] " + statement + " [SEP]"

        tokenized_statement = tokenizer.tokenize(statement)

        response = response + " [SEP]"
        tokenized_response = tokenizer.tokenize(response)

        # Combining both statement and response into a single input for the model
        combinedInput = statement + response

        # Converting tokens to vocabulary indices
        tokens_as_indexes = tokenizer.convert_tokens_to_ids(tokenized_statement + tokenized_response)
        segments_ids = [0] * len(tokenized_statement) + [1] * len(tokenized_response)

        # Converting inputs to tensors
        tokens_tensor = torch.tensor([tokens_as_indexes])
        segments_tensors = torch.tensor([segments_ids])


        CONFIG_NAME = "config.json"
        WEIGHTS_NAME = "pytorch_model.bin"


        # Load the weights from the pre-trained model
        try:
            with torch.no_grad():
                # Carry out prediction whether response 'IsNextSent'
                predictions = model(tokens_tensor, segments_tensors)
                predictionScore = torch.nn.functional.softmax(predictions, dim=1)[:, 0]
    #           return(float(predictionScore))
    #             print("Prediction Probability: ", predictionScore)
                if (predictionScore > 0.5):
                    return('1')
                else:
                    return('0')

    #             print(predictionOutput)
        except Exception as exec:
            print('Error Handler: ', exec)
    
    def checkNLI(_hypothesis, _premise):
        results = predictor.predict(
            hypothesis=(_hypothesis),
            premise=(_premise)
        )
        entailment = round((results['label_probs'][0])*100,1)
        contradiction = round((results['label_probs'][1])*100,1)
        neutral = round((results['label_probs'][2])*100,1)
        return entailment, contradiction, neutral
