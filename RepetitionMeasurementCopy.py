
# coding: utf-8

# In[76]:

import pandas as pd
from tqdm import tqdm_notebook as tqdm
from nltk import ngrams, collections
import re
import string
import ipython_genutils
import csv

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer


# In[112]:

# class checkRepetitionInternal():
    
#     def check_internalRepetition(response):

#     #     print('-------------------INTERNAL------------------')


#         text_txt = re.sub('<.*>','',response)
#         punctutationNoPeriod='[' + re.sub('\.','',string.punctuation) + ']'
#         tex_txt = re.sub(punctutationNoPeriod,'',text_txt)

#         tokenized = text_txt.split()
#         tokenized_lower = []

#         for token in tokenized:
#             tokenized_lower.append(token.lower())

#         unigram_repetitions = ngrams(tokenized_lower, 1)
#         unigram_repetitions = [ ' '.join(grams) for grams in unigram_repetitions]
#         unigram_repetitions_len = (len(unigram_repetitions))
#         unigram_repetitions_freq = collections.Counter(unigram_repetitions)

#         unigram_count = 0
#         unigram_len = 0
#         from collections import Counter
#         for k,v in unigram_repetitions_freq.items():
#             if v >= 1:
#                 unigram_len = unigram_len + v
#                 if v > 1:
#                     unigram_count = unigram_count + v

#         bigram_repetitions = ngrams(tokenized_lower, 2)
#         bigram_repetitions = [ ', '.join(grams) for grams in bigram_repetitions]
#         bigram_repetitions_len = (len(bigram_repetitions))
#         bigram_repetitions_freq = collections.Counter(bigram_repetitions)

#         bigram_count = 0
#         bigram_len = 0
#         from collections import Counter
#         for k,v in bigram_repetitions_freq.items():
#             if v >= 1:
#                 bigram_len = bigram_len + v
#                 if v > 1:
#                     bigram_count = bigram_count + v

#         trigram_repetitions = ngrams(tokenized_lower, 3)
#         trigram_repetitions = [ ', '.join(grams) for grams in trigram_repetitions]
#         trigram_repetitions_len = (len(trigram_repetitions))
#         trigram_repetitions_freq = collections.Counter(trigram_repetitions)

#         trigram_count = 0
#         trigram_len = 0
#         from collections import Counter
#         for k,v in trigram_repetitions_freq.items():
#             if v >= 1:
#                 trigram_len = trigram_len + v
#                 if v > 1:
#                     trigram_count = trigram_count + v

#         if unigram_count < 1:
#             unigram_count = 0

#         if unigram_len < 1:
#             unigram_len = 1
            
#         if bigram_count < 1:
#             bigram_count = 0

#         if bigram_len < 1:
#             bigram_len = 1
            
#         if trigram_count < 1:
#             trigram_count = 0

#         if trigram_len < 1:
#             trigram_len = 1

#         print((round(((unigram_count/unigram_len)*100),1)))
#         return round(((round(((unigram_count/unigram_len)*100),1))+
#                 round(((bigram_count/bigram_len)*100),1)+
#                 round(((trigram_count/trigram_len)*100),1))/3,1)

        
#     #     df_merge_col = pd.concat([a1, a2, a3], axis=1)
#     #     return df_merge_col


# In[114]:

# class checkRepetitionPartner():
    
#     def check_partnerRepetition(statement, response):

#         repeated_words = []

#         punctutationNoPeriod='[' + re.sub('\.','',string.punctuation) + ']'

#         #Statement
#         text_statement = re.sub('<.*','', statement)
#         text_statement = re.sub(punctutationNoPeriod,'',text_statement)

#         tokenized_statement = text_statement.split()
#         tokenized_statement_lower = []

#         for token in tokenized_statement:
#             tokenized_statement_lower.append(token.lower())

#         statement_unigram_repetitions = ngrams(tokenized_statement_lower, 1)
#         statement_unigram_repetitions = [ ' '.join(grams) for grams in statement_unigram_repetitions]
#         statement_unigram_repetitions_len = (len(statement_unigram_repetitions))
#         statement_unigram_repetitions_freq = collections.Counter(statement_unigram_repetitions)

#         statement_bigram_repetitions = ngrams(tokenized_statement_lower, 2)
#         statement_bigram_repetitions = [ ', '.join(grams) for grams in statement_bigram_repetitions]
#         statement_bigram_repetitions_len = (len(statement_bigram_repetitions))
#         statement_bigram_repetitions_freq = collections.Counter(statement_bigram_repetitions)

#         statement_trigram_repetitions = ngrams(tokenized_statement_lower, 3)
#         statement_trigram_repetitions = [ ', '.join(grams) for grams in statement_trigram_repetitions]
#         statement_trigram_repetitions_len = (len(statement_trigram_repetitions))
#         statement_trigram_repetitions_freq = collections.Counter(statement_trigram_repetitions)

#         #Response
#         text_response = re.sub('<.*>','',response)
#         text_response = re.sub(punctutationNoPeriod,'',text_response)

#         tokenized_response = text_response.split()
#         tokenized_response_lower = []
#         for token in tokenized_response:
#             tokenized_response_lower.append(token.lower())

#         response_unigram_repetitions = ngrams(tokenized_response_lower, 1)
#         response_unigram_repetitions = [ ' '.join(grams) for grams in response_unigram_repetitions]
#         response_unigram_repetitions_freq = collections.Counter(response_unigram_repetitions)

#         for word in tokenized_statement_lower:
#             if word in tokenized_response_lower:
#                 repeated_words.append(word)

#         response_bigram_repetitions = ngrams(tokenized_response_lower, 2)
#         response_bigram_repetitions = [ ', '.join(grams) for grams in response_bigram_repetitions]
#         response_bigram_repetitions_freq = collections.Counter(response_bigram_repetitions)

#         repeated_bigrams = []
#         for bigram in statement_bigram_repetitions_freq:
#             if bigram in response_bigram_repetitions_freq:
#                 repeated_bigrams.append(bigram)
       
#         response_trigram_repetitions = ngrams(tokenized_response_lower, 3)
#         response_trigram_repetitions = [ ', '.join(grams) for grams in response_trigram_repetitions]
#         response_trigram_repetitions_freq = collections.Counter(response_trigram_repetitions)

#         repeated_trigrams = []
#         for trigram in statement_trigram_repetitions_freq:
#             if trigram in response_trigram_repetitions_freq:
#                 repeated_trigrams.append(trigram)
                
#         if len(response_unigram_repetitions) < 1:
#             response_unigram_repetitions = 'word'

#         if len(repeated_words) < 1:
#             repeated_words = ''
            
#         if len(response_bigram_repetitions) < 1:
#             response_bigram_repetitions = 'word'

#         if len(repeated_bigrams) < 1:
#             repeated_bigrams = ''
            
#         if len(response_trigram_repetitions) < 1:
#             response_trigram_repetitions = 'word'

#         if len(repeated_trigrams) < 1:
#             repeated_trigrams = ''
      
#         return round(((round((len(repeated_words)/len(response_unigram_repetitions)*100),1))+
#                 round((len(repeated_bigrams)/len(response_bigram_repetitions)*100),1)+
#                 round((len(repeated_trigrams)/len(response_trigram_repetitions)*100),1))/3,1)


# In[122]:

# class checkRepetitionExternal():

#     def check_externalRepetition(history, response):

#         history_unigram_repetitions = []
#         history_bigram_repetitions = []
#         history_trigram_repetitions = []

#         repeated_words = []

#         punctutationNoPeriod='[' + re.sub('\.','',string.punctuation) + ']'

#         #History
#         text_history = re.sub('<.*','', history)
#         text_history = re.sub(punctutationNoPeriod,'',text_history)

#         tokenized_history = text_history.split()
#         tokenized_history_lower = []

#         for token in tokenized_history:
#             tokenized_history_lower.append(token.lower())

#         history_unigram_repetitions = ngrams(tokenized_history_lower, 1)
#         history_unigram_repetitions = [ ' '.join(grams) for grams in history_unigram_repetitions]
#         history_unigram_repetitions_freq = collections.Counter(history_unigram_repetitions)

#         history_bigram_repetitions = ngrams(tokenized_history_lower, 2)
#         history_bigram_repetitions = [ ', '.join(grams) for grams in history_bigram_repetitions]
#         history_bigram_repetitions_freq = collections.Counter(history_bigram_repetitions)

#         history_trigram_repetitions = ngrams(tokenized_history_lower, 3)
#         history_trigram_repetitions = [ ', '.join(grams) for grams in history_trigram_repetitions]
#         history_trigram_repetitions_freq = collections.Counter(history_trigram_repetitions)

#         #Response
#         text_response = re.sub('<.*>','',response)
#         text_response = re.sub(punctutationNoPeriod,'',text_response)

#         tokenized_response = text_response.split()
#         tokenized_response_lower = []
#         for token in tokenized_response:
#             tokenized_response_lower.append(token.lower())

#         response_unigram_repetitions = ngrams(tokenized_response_lower, 1)
#         response_unigram_repetitions = [ ' '.join(grams) for grams in response_unigram_repetitions]
#         response_unigram_repetitions_freq = collections.Counter(response_unigram_repetitions)

#         for word in history_unigram_repetitions_freq:
#             if word in response_unigram_repetitions:
#                 repeated_words.append(word)
        
#         response_bigram_repetitions = ngrams(tokenized_response_lower, 2)
#         response_bigram_repetitions = [ ', '.join(grams) for grams in response_bigram_repetitions]
#         response_bigram_repetitions_freq = collections.Counter(response_bigram_repetitions)

#         repeated_bigrams = []
#         for bigram in history_bigram_repetitions_freq:
#             if bigram in response_bigram_repetitions:
#                 repeated_bigrams.append(bigram)
        
#         response_trigram_repetitions = ngrams(tokenized_response_lower, 3)
#         response_trigram_repetitions = [ ', '.join(grams) for grams in response_trigram_repetitions]
#         response_trigram_repetitions_freq = collections.Counter(response_trigram_repetitions)

#         repeated_trigrams = []
#         for trigram in history_trigram_repetitions_freq:
#             if trigram in response_trigram_repetitions:
#                 repeated_trigrams.append(trigram)
                
#         if len(response_unigram_repetitions) < 1:
#             response_unigram_repetitions = 'word'

#         if len(repeated_words) < 1:
#             repeated_words = ''
            
#         if len(response_bigram_repetitions) < 1:
#             response_bigram_repetitions = 'word'

#         if len(repeated_bigrams) < 1:
#             repeated_bigrams = ''
            
#         if len(response_trigram_repetitions) < 1:
#             response_trigram_repetitions = 'word'

#         if len(repeated_trigrams) < 1:
#             repeated_trigrams = ''
            
#         print()
       
#         return round(((round((len(repeated_words)/len(response_unigram_repetitions)*100),1))+
#                 round((len(repeated_bigrams)/len(response_bigram_repetitions)*100),1)+
#                 round((len(repeated_trigrams)/len(response_trigram_repetitions)*100),1))/3,1)


# In[91]:

class checkRepetitionInternal():
    
    def check_internalRepetition(response):
        """Returns the fraction of items in the list that are repeated"""
        if len(response) == 0:
            return 0.0
        num_rep = 0
        for idx in range(len(response)):
            if response[idx] in response[:idx]:
                num_rep += 1
        return round(num_rep / len(response),1)
    
    def get_ngrams(text, n):
        """Returns all ngrams that are in the text.
        Inputs:
            text: string
            n: int
        Returns:
            list of strings (each is a ngram)
        """
        text = text.lower()
        tokens = text.split()
        return [
            " ".join(tokens[i : i + n]) for i in range(len(tokens) - (n - 1))
        ]  # list of str
    
    def intrep_repeated_ngram_frac(response, n):
        """
        Sentence-level attribute function. See explanation above.
        Returns the fraction of n-grams in utt that are repeated.
        Additional inputs:
          n: int, the size of the n-grams considered.
        """
        if re.search(r"^\s+$",response):
            ngrams = ""
        elif not response:
            ngrams = ""
        else:
            assert response.strip() != ""
            ngrams = checkRepetitionExternal.get_ngrams(response, n)
        
        return checkRepetitionInternal.check_internalRepetition(ngrams)


# In[89]:

class checkRepetitionPartner():
    
    def check_partnerRepetition(statement, response):
        """Returns the fraction of items in lst1 that are in lst2"""
        if len(response) == 0:
            return 0.0
        num_rep = len([x for x in response if x in statement])
        return round(num_rep / len(response),1)
    
    def get_ngrams(text, n):
        """Returns all ngrams that are in the text.
        Inputs:
            text: string
            n: int
        Returns:
            list of strings (each is a ngram)
        """
        text = text.lower()
        tokens = text.split()
        return [
            " ".join(tokens[i : i + n]) for i in range(len(tokens) - (n - 1))
        ]  # list of str
    
    def partnerrep_repeated_ngram_frac(statement, response, n):
        """
        Sentence-level attribute function. See explanation above.
        Returns the fraction of n-grams in utt that are repeated.
        Additional inputs:
          n: int, the size of the n-grams considered.
        """
        if re.search(r"^\s+$",statement):
            ngrams_statement = ""
        elif not statement:
            ngrams_statement = ""
        else:
            assert statement.strip() != ""
            ngrams_statement = checkRepetitionExternal.get_ngrams(statement, n)
            
        if re.search(r"^\s+$",response):
            ngrams_response = ""
        elif not response:
            ngrams_response = ""
        else:
            assert response.strip() != ""
            ngrams_response = checkRepetitionExternal.get_ngrams(response, n)
        
        return checkRepetitionPartner.check_partnerRepetition(ngrams_response, ngrams_statement)


# In[83]:

class checkRepetitionExternal():
    
    def check_externalRepetition(history, response):
        """Returns the fraction of items in lst1 that are in lst2"""
        if len(response) == 0:
            return 0.0
        num_rep = len([x for x in response if x in history])
        return round(num_rep / len(response),1)
    
    def get_ngrams(text, n):
        """Returns all ngrams that are in the text.
        Inputs:
            text: string
            n: int
        Returns:
            list of strings (each is a ngram)
        """
        text = text.lower()
        tokens = text.split()
        return [
            " ".join(tokens[i : i + n]) for i in range(len(tokens) - (n - 1))
        ]  # list of str
    
    def extrep_repeated_ngram_frac(history, response, n):
        """
        Sentence-level attribute function. See explanation above.
        Returns the fraction of n-grams in utt that are repeated.
        Additional inputs:
          n: int, the size of the n-grams considered.
        """
        
        if re.search(r"^\s+$",history):
            ngrams_history = ""
        elif not history:
            ngrams_history = ""
        else:
            assert history.strip() != ""
            ngrams_history = checkRepetitionExternal.get_ngrams(history, n)
            
        if re.search(r"^\s+$",response):
            ngrams_response = ""
        elif not response:
            ngrams_response = ""
        else:
            assert response.strip() != ""
            ngrams_response = checkRepetitionExternal.get_ngrams(response, n)
            
        
        return checkRepetitionExternal.check_externalRepetition(ngrams_response, ngrams_history)

