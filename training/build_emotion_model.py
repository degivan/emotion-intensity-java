from os import path, listdir
import numpy as np
import pandas as pd
import gensim
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import resource
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense
from keras.layers import GaussianNoise, SimpleRNN, LSTM, Reshape, Embedding, SpatialDropout1D, GaussianDropout, Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn.linear_model import SGDRegressor
from collections import namedtuple
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import pickle
import sys
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
import json
import random as rn

import os

os.mkdir('networks')

GPU = True

if GPU:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=4,\
        inter_op_parallelism_threads=4, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)


position_from_emotion = {}
position_from_emotion['anger'] = [1, 0, 0, 0]
position_from_emotion['sadness'] = [0, 1, 0, 0]
position_from_emotion['joy'] = [0, 0, 1, 0]
position_from_emotion['fear'] = [0, 0, 0, 1]

cl_from_emotion = {}
cl_from_emotion['anger'] = 0
cl_from_emotion['sadness'] = 1
cl_from_emotion['joy'] = 2
cl_from_emotion['fear'] = 3


class Tweet(object):
    def __init__(self, message, res, common_class):
        self.cl = cl_from_emotion[common_class]
        self.message = message
        self.res = [x * res for x in position_from_emotion[common_class]]

    def __str__(self):
        return str(self.message) + " " + str(self.res)


def get_tweet(str_tweet, res_acc=1):
    num, message, common_class, res = str_tweet.split('\t')
    if res == 'NONE':
        res = '1.000'
    return Tweet(message, float(res[0:res_acc]), common_class)
        


def get_tweets(str_tweets, res_acc=1):
    return [get_tweet(line, res_acc) for line in str_tweets.split('\n') if len(line) > 0]

EMOTIONS = ['anger', 'joy', 'sadness', 'fear']

def run_competition_files(path_pattern):
    em_tweets = {}
    for emotion in EMOTIONS:
        filename = path.join(path_pattern % emotion)
        file = open(filename, 'r')
        em_tweets[emotion] = get_tweets(file.read(), res_acc=5)
        file.close()
    return em_tweets
    
train_tweets = run_competition_files('train_data/EI-reg-en_%s_train.txt')
test_tweets = run_competition_files('train_data/EI-reg-en_%s_train.txt') # no reason to include test data here

dirty_tweets =[]

directory = path.join('dirty_data/labeled')
for filename in os.listdir(directory):
    file = open(path.join(directory,filename), 'r')
    dirty_tweets += get_tweets(file.read(), res_acc=5)
    file.close()

EMOTION = sys.argv[1]

tweets = np.array(list(dirty_tweets) + list(train_tweets[EMOTION]) + list(test_tweets[EMOTION]))
    
tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

def normalize_text(text):
    stripped = re.sub(combined_pat, '', text)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip() 

LOAD_TEXT = False

dirty_texts, train_texts, test_texts, texts = [], [], [], []
if not LOAD_TEXT:
    texts = [normalize_text(t.message) for t in tweets]
    dirty_texts = texts[0:len(dirty_tweets)]
    train_texts = texts[len(dirty_tweets): len(dirty_tweets) + len(train_tweets[EMOTION])]
    test_texts = texts[len(dirty_tweets) + len(train_tweets[EMOTION]):]
    
    assert (len(train_texts) == len(train_tweets[EMOTION]))
    assert (len(test_texts) == len(test_tweets[EMOTION]))
else:
    dirty_texts = list(np.loadtxt('features/dirty_texts.txt', dtype='str', delimiter='\n'))
    train_texts = list(np.loadtxt('features/train_texts_%s_.txt' % EMOTION, dtype='str', delimiter='\n'))
    test_texts = list(np.loadtxt('features/test_texts_%s_.txt' % EMOTION, dtype='str', delimiter='\n'))
    texts = dirty_texts + train_texts + test_texts
    
    assert (len(train_texts) == len(train_tweets[EMOTION]))
    assert (len(test_texts) == len(test_tweets[EMOTION])) 
    
s = len(dirty_tweets)
f = s + len(train_tweets[EMOTION])
e = f + len(test_tweets[EMOTION])

max_features = 20000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X)

dirty_X = X[0:s]
train_X = X[s:f]
test_X = X[f:e]

Y = np.array([t.res for t in tweets])

dirty_Y = Y[0:s]
train_Y = Y[s:f]
test_Y = Y[f:e]

if EMOTION == 'anger' or EMOTION == 'fear':
    updated_dirty_Y = []
    for row in dirty_Y.tolist():
        updated_dirty_Y.append(row)
        if row[cl_from_emotion['sadness']] == 1:
            row[cl_from_emotion['sadness']] = 0.33
            row[cl_from_emotion['anger']] = 0.33
            row[cl_from_emotion['fear']] = 0.33
    dirty_Y = np.array(updated_dirty_Y)

Params = namedtuple('Params', 'layers loss optimizer dirty_e dirty_bs train_e train_bs')

def create_params(dirty_e, dirty_bs, train_e, train_bs, layers, optimizer='adam'):
    return Params(layers, 'mean_squared_error', optimizer, dirty_e, dirty_bs, train_e, train_bs)

def create_model(params):
    nm = Sequential()
    for layer in params.layers:
        nm.add(layer())
    nm.compile(loss='mean_squared_error', optimizer=params.optimizer)
    return nm

embeddings_dim = 300
embeddings = dict()
embeddings = KeyedVectors.load_word2vec_format( "twitter_sgns_subset.txt.gz" , binary=False ) 

embedding_weights = np.zeros((max_features , embeddings_dim ) )
for word,index in tokenizer.word_index.items():
    if index < max_features:
        try: embedding_weights[index,:] = embeddings[word]
        except: embedding_weights[index,:] = np.random.rand( 1 , embeddings_dim )
            
np.random.seed(27)

params_list = []

lstm_layers = []
lstm_layers.append(lambda: Embedding(max_features, embeddings_dim, input_length = 52, weights=[embedding_weights]))
lstm_layers.append(lambda: Dropout(0.5))
lstm_layers.append(lambda: Conv1D(embeddings_dim, 3, activation='relu', padding='valid', strides=1))
lstm_layers.append(lambda: MaxPooling1D(pool_size=2))
lstm_layers.append(lambda: LSTM(embeddings_dim, dropout=0.5, recurrent_dropout=0.5))
lstm_layers.append(lambda: Dense(4, activation='sigmoid'))

params_list.append(create_params(15, 5000, 15, 16, lstm_layers))


for p in params_list:
        neural_model = create_model(p)
        neural_model.fit(np.vstack((dirty_X)), \
                         np.vstack((dirty_Y)), \
                         epochs=p.dirty_e,\
                         batch_size=p.dirty_bs)
        neural_model.fit(train_X, train_Y, epochs=p.train_e, batch_size=p.train_bs)
        model_name = 'networks/best_model_%s.h5' % EMOTION
        with open('networks/tokenizer_%s.pickle' % EMOTION, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('networks/word_index_%s.json' % EMOTION, 'w') as outfile:
            json.dump(tokenizer.word_index, outfile)
        neural_model.save(model_name)