import numpy as np

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import re

# Import cleaned_data CSV
data = pd.read_csv(
    "C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/cleaned_data.csv",
    nrows=40000,
    encoding='latin-1')
data['clean_text'] = data.clean_text.astype(str)

# integer encode the documents
vocab_size = 1000
data['textInInt'] = [one_hot(d, vocab_size) for d in data['clean_text']]

# Convert text to vector
X = pad_sequences(data['textInInt'])

# Model description
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(vocab_size, embed_dim, input_length=28))

# model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.1, recurrent_dropout=0.1)))

model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# encode the sentiments using Label encoder

# this seems to fail because only 0 in array
# from sklearn.preprocessing import LabelEncoder
# Le = LabelEncoder()
# y = Le.fit_transform(data['polarity'])

y = data['polarity'].to_numpy()

# Split training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Evaluate the model performance
model.evaluate(X_test, y_test)
