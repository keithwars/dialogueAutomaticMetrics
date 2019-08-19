
# coding: utf-8

# # BLEU Score

# ### Synopsis

# This is the Baseline - BLEU module. It is an external script contibuting to the main class of the combined component by providing results for the baseline model. It loads the main model, from NLTK. The imports necessary for the correct execution of this script are listed under the Imports header block. The following parts load the model, consecutively. The output can be either a smoothed flaot value based on the n-grams overlap.

# ### Imports

# In[3]:

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm_notebook as tqdm
smoother = SmoothingFunction()

import logging
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import pandas as pd
import nltk
import re


# ### Single-Turn BLEU Score

# In[1]:

reader = csv.reader(open('data/parlAI_data.txt', encoding = "ISO-8859-1"))
header = next(reader)
lines = list(reader)

pbar = tqdm(total=len(lines))
for row in lines:
    output = check_short_safe(row[9],row[10])
    lines[int(row[0])][18] = output
    pbar.update(1)
pbar.close()

# writer = csv.writer(open('data/MultiTurnOutputFinal_Safe.csv', 'w'))
# writer.writerow(header)
# writer.writerows(lines)


# ### Multi-Turn BLEU Score

# In[214]:

df = pd.read_json ('data/summer_wild_evaluation_dialogs.json')
dfObj = pd.DataFrame(columns = ['id','statement', 'response', 'bleu'])

pbar = tqdm(total=len(df))
for i in range(0, len(df)):
    if (df['eval_score'][i] != 'null'):
        if (len(df['dialog'][i])) > 2:
            if ((df['dialog'][i][0]['sender_class']) == 'Bot'):
                    skip_first = True
            for j in range (0, len(df['dialog'][i])):
                if skip_first:
                    skip_first = False
                else:
                    if ((df['dialog'][i][j]['sender_class']) == (df['dialog'][i][j-1]['sender_class'])):
                            continue
                    else:
                        if ((df['dialog'][i][j]['sender_class']) == 'Bot'):
                            if ((df['dialog'][i][j]['text']) != 'Text is not given. Please try to type /end and /test to reset the state and get text.'):
                                
                                statement = (df['dialog'][i][j-1]['text'])
                                statement = re.sub(r'[^\w\s]','',statement)
                                statement = statement.split()
                                statement = [[x.lower() for x in statement]]
                                
                                response = (df['dialog'][i][j]['text'])
                                response = re.sub(r'[^\w\s]','',response)
                                response = response.split()
                                response = [x.lower() for x in response]
                                
                                score = nltk.translate.bleu_score.sentence_bleu(statement, response,smoothing_function=smoother.method1)
#                                 score = sentence_bleu(statement, response)
                                
                                dfObj = dfObj.append({'id': i, 'statement': df['dialog'][i][j-1]['text'], 'response': df['dialog'][i][j]['text'], 'bleu': score }, ignore_index=True)
                        else:
                            continue
    pbar.update(1)
pbar.close()    
dfObj.to_csv('data/summer_wild_evaluation_dialogs.csv', encoding='utf-8', index=False)
# print(dfObj)

