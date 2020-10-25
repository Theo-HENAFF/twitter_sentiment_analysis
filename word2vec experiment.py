import pandas as pd
import numpy as np
from gensim.models import Word2Vec

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# ------------------------------------------------------
# Import cleaned_data CSV
# ------------------------------------------------------
df = pd.read_csv(
    # "C:/Users/Théo/Documents/twitter_sentiment_analysis/data/cleaned_data.csv",
    "C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/cleaned_data.csv",
    # nrows=20000,
    encoding='latin-1')
df['clean_text'] = df.clean_text.astype(str)

dict_word = {}
sentences = []
for items in df['clean_text'].iteritems():
    words = items[1].split(" ")
    sentences.append(words)
    for word in words:
        if word in dict_word:
            dict_word[word] += 1
        else:
            dict_word[word] = 1  # dictionary['UNK']

# keep tokens with a min occurrence
min_occurance = 5
cleaned_dict_word = [k for k,c in dict_word.items() if c >= min_occurance]

# sentences = df['clean_text'].tolist()
print('Total training sentences: %d' % len(sentences))

# train word2vec model
model = Word2Vec(sentences, size=20, min_count=20, workers=10)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

# save model in ASCII (word2vec) format
filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)

print(model.wv.most_similar('good'))

def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename,'r')
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return embedding


def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, 100))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return weight_matrix


# # pad sequences
# max_length = max([len(s.split()) for s in df['clean_text']])
#
# # fit the tokenizer on the documents
# tk = Tokenizer(lower=True)
# tk.fit_on_texts(df['clean_text'].values)
# X_seq = tk.texts_to_sequences(df['clean_text'].values)  # X)
# X_pad = pad_sequences(X_seq, maxlen=max_length, padding='post')  #maxlen must be equal to maxword
#
# X_train, X_test, y_train, y_test = train_test_split(X_pad, df['polarity'].values, test_size=0.25, random_state=1, shuffle= True)
#
# batch_size = 256
# X_train1 = X_train[batch_size:]
# y_train1 = y_train[batch_size:]
# X_valid = X_train[:batch_size]
# y_valid = y_train[:batch_size]
#
# # define vocabulary size (largest integer value)
# vocab_size = len(tk.word_index) + 1
#
# # load embedding from file
# raw_embedding = load_embedding('embedding_word2vec.txt')
# # get vectors in the right order
# embedding_vectors = get_weight_matrix(raw_embedding, tk.word_index)
# # create the embedding layer
# embedding_layer = Embedding(vocab_size, 100, input_length=max_length, trainable=False, weights=[embedding_vectors])
#
# # define model
# model = Sequential()
# model.add(embedding_layer)
# model.add(LSTM(256, return_sequences=True))
# model.add(LSTM(128))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# history = model.fit(X_train1, y_train1, shuffle=True, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=20)
#
# # ------------------------------------------------------
# # Evaluate the model performance
# # ------------------------------------------------------
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Test accuracy : ", scores[1])
#
#
# loss = history.history['loss']
# loss_val = history.history['val_loss']
# accuracy = history.history['accuracy']
# accuracy_val = history.history['val_accuracy']
# epochs = range(1, len(loss)+1)
#
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, loss_val, 'b--', label='validation loss')
# plt.plot(epochs, accuracy, 'r', label='Training accuracy')
# plt.plot(epochs, accuracy_val, 'r--', label='validation accuracy')
#
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.legend()
# plt.grid()
# plt.show()
