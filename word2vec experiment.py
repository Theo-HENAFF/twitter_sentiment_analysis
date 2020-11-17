import pandas as pd
import numpy as np
from gensim.models import Word2Vec

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

from sklearn.model_selection import train_test_split

import tensorflow as tf
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Setup GPU
# ------------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6172)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.001, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)

# ------------------------------------------------------
# Parameters
# ------------------------------------------------------
# keep tokens with a min occurrence
min_occurance = 10
# Word2Vec parameters
size_word2vec = 200
min_count_word2vec = 10
# NN s general parameters
batch_size = 512
epochs = 50
test_size=0.25
# LSTM s parameters
lstm_size = 512
lstm_dropout = 0.5
lstm_recurrent_dropout = 0.0

# ------------------------------------------------------
# Import cleaned_data CSV
# ------------------------------------------------------
df = pd.read_csv(
    # "C:/Users/Théo/Documents/twitter_sentiment_analysis/data/cleaned_data.csv",
    "C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/mid_cleaned_data.csv",
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


cleaned_dict_word = [k for k,c in dict_word.items() if c >= min_occurance]

# ------------------------------------------------------
# Train Word2Vec model
# ------------------------------------------------------
model_Word2Vec = Word2Vec(sentences, size=size_word2vec, min_count=min_count_word2vec, workers=10)
# summarize vocabulary size in model
words = list(model_Word2Vec.wv.vocab)
print('Vocabulary size: %d' % len(words))

print(model_Word2Vec.wv.most_similar('good'))

size_word2vec = model_Word2Vec.vector_size


# ------------------------------------------------------
# Define functions to create the word2vec weight matrix
# ------------------------------------------------------
def get_weight_matrix(vocab):
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, size_word2vec))

    for w, i in vocab.items():
        # The word_index contains a token for all words of the training data so we need to limit that
        if i < vocab_size:
            try:
                vect = model_Word2Vec.wv.get_vector(w)
                weight_matrix[i] = vect
            # Check if the word from the training data occurs in the GloVe word embeddings
            # Otherwise the vector is kept with only zeros
            except:
                pass
        else:
            break
    return weight_matrix


# ------------------------------------------------------
# Convert text to vector
# ------------------------------------------------------
max_length = max([len(s.split()) for s in df['clean_text']])
# fit the tokenizer on the documents
tk = Tokenizer(lower=True)
tk.fit_on_texts(df['clean_text'].values)

X_seq = tk.texts_to_sequences(df['clean_text'].values)
X_pad = pad_sequences(X_seq, maxlen=max_length, padding='post')  # maxlen must be equal to maxword

# define vocabulary size (largest integer value)
vocab_size = len(tk.word_index) + 1

# embedding_vectors = get_weight_matrix(raw_embedding, tk.word_index)
embedding_vectors = get_weight_matrix(tk.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, size_word2vec, input_length=max_length, trainable=False, weights=[embedding_vectors])

# ------------------------------------------------------
# Split test/train data
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_pad, df['polarity'].values, test_size=test_size, random_state=1, shuffle=True)

X_train1 = X_train[batch_size:]
y_train1 = y_train[batch_size:]
X_valid = X_train[:batch_size]
y_valid = y_train[:batch_size]

# ------------------------------------------------------
# Model description
# ------------------------------------------------------
model = Sequential()

model.add(embedding_layer)
model.add(LSTM(lstm_size, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, return_sequences=True))
model.add(LSTM(lstm_size, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train1, y_train1, shuffle=True, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=epochs)

# ------------------------------------------------------
# Evaluate the model performance
# ------------------------------------------------------
scores = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy : ", scores[1])


loss = history.history['loss']
loss_val = history.history['val_loss']
accuracy = history.history['accuracy']
accuracy_val = history.history['val_accuracy']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, loss_val, 'b--', label='validation loss')
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, accuracy_val, 'r--', label='validation accuracy')

plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid()
plt.show()
