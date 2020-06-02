import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn import model_selection
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from gensim.models import KeyedVectors

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNLSTM, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model,Sequential
from keras import initializers, regularizers, constraints, optimizers, layers

tqdm.pandas()
np.random.seed(2020)

embedding_dims = 300 
max_features = 20000
maxlen = 200
test_percent = 0.1 	# 0.2 for testing
data_extra = True   # Use extra training-data

## Load the train data 
print(">> Read data...")
if data_extra:
    train_path = "subtask-1/train-extra.csv"
else:
    train_path = "subtask-1/train.csv"
test_path = "subtask-1/subtask1_test.csv"
train = pd.read_csv(train_path, encoding = 'utf-8')
test = pd.read_csv(test_path, encoding = 'utf-8')

print("File: %s" % train_path)
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)



## Load the pre-trained embedding
## To use the pretrained embedding, you need to download the embedding file: https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
## Unzip and place it in the directory of this project

EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2v1ec_format(EMBEDDING_FILE, binary=True)

embeddings = np.stack(list(embeddings_index[word] for word in embeddings_index.vocab))

print("Google-news Doc2Vec Word Embeddings's shape:")
print(embeddings.shape)

emb_mean, emb_std = embeddings.mean(), embeddings.std()



##data pre-processing

train_df, validation_df = model_selection.train_test_split(train, test_size= test_percent, random_state = 2020)

## fill up the missing values
train_X = train_df["sentence"].fillna("_na_").values
validation_X = validation_df["sentence"].fillna("_na_").values
test_X = test["sentence"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
validation_X = tokenizer.texts_to_sequences(validation_X)
test_X = tokenizer.texts_to_sequences(test_X)
## Pad the sentences
train_X = pad_sequences(train_X, maxlen=maxlen)
validation_X = pad_sequences(validation_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_Y = train_df['gold_label'].values
validation_Y = validation_df['gold_label'].values


## Get processed pre-trained embedding matrix
 
word_index = tokenizer.word_index

embeddings_matrix = np.random.normal(emb_mean, emb_std, (max_features, embedding_dims))

for word, index in word_index.items():
    
    if index >= max_features:
        continue
    try:
        embeddings_vector = embeddings_index[word]
        embeddings_matrix[index] = embedding_vector
    except:
        pass
    
    
print("embeddings_matrix's shape:")    
print(embeddings_matrix.shape)

## Build Biodirectional-LSTM Model

model = Sequential()
model.add(Embedding(max_features, embedding_dims, embeddings_initializer = initializers.Constant(embeddings_matrix), input_length = maxlen, trainable = False))
model.add(Bidirectional(LSTM(64, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(16, activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(1, activation = "sigmoid"))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()

## train the model
model.fit(train_X, train_Y, batch_size = 256, epochs = 15, validation_data = (validation_X, validation_Y))

_ , acc = model.test_on_batch(validation_X, validation_Y, sample_weight=None)
print("Test accuracy : " + str(acc))

## Evaluate The Model
Y_pred = model.predict([validation_X],batch_size = 256, verbose = 1)

F1_score = f1_score(validation_Y, (Y_pred>0.5).astype(int))

print("The F1_score is :" + str(F1_score))
