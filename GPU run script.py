import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional

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
    monitor='loss', min_delta=0.001, patience=20, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)


# ------------------------------------------------------
# Import cleaned_data CSV
# ------------------------------------------------------
data = pd.read_csv(
    "C:/Users/Théo/Documents/twitter_sentiment_analysis/data/cleaned_data.csv",
    # "C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/mid_cleaned_data.csv",
    # nrows=20000,
    encoding='latin-1')
data['clean_text'] = data.clean_text.astype(str)

# ------------------------------------------------------
# Convert text to vector
# ------------------------------------------------------
# X, y = (data['clean_text'].values, data['polarity'].values)
tk = Tokenizer(lower=True)
tk.fit_on_texts(data['clean_text'].values)
X_seq = tk.texts_to_sequences(data['clean_text'].values)  # X)
X_pad = pad_sequences(X_seq, maxlen=30, padding='post')  #maxlen must be equal to maxword

X_train, X_test, y_train, y_test = train_test_split(X_pad, data['polarity'].values, test_size=0.25, random_state=1, shuffle= True)

batch_size = 256
X_train1 = X_train[batch_size:]
y_train1 = y_train[batch_size:]
X_valid = X_train[:batch_size]
y_valid = y_train[:batch_size]

# ------------------------------------------------------
# Model parameters
# ------------------------------------------------------
vocabulary_size = len(tk.word_counts.keys()) + 1
max_words = 30 #100 #maxlen must be equal to maxword
embedding_size = 32

# ------------------------------------------------------
# Model description
# ------------------------------------------------------
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))

model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Bidirectional(LSTM(256)))

# model.add(LSTM(200, return_sequences=True))
# model.add(LSTM(200, return_sequences=True))
# model.add(LSTM(200))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train1, y_train1, shuffle=True, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=30)

# ------------------------------------------------------
# Evaluate the model performance
# ------------------------------------------------------
scores = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy : ", scores[1])


loss = history.history['loss']
loss_val = history.history['val_loss']
accuracy = history.history['accuracy']
accuracy_val = history.history['val_accuracy']
epochs = range(1,len(loss)+1)

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, loss_val, 'b--', label='validation loss')
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, accuracy_val, 'r--', label='validation accuracy')

plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid()
plt.show()