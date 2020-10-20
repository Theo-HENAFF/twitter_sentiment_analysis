import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional

from sklearn.model_selection import train_test_split

# Import cleaned_data CSV
data = pd.read_csv(
    "C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/mid_cleaned_data.csv",
    # nrows=20000,
    encoding='latin-1')
data['clean_text'] = data.clean_text.astype(str)

# X, y = (data['clean_text'].values, data['polarity'].values)
tk = Tokenizer(lower=True)
tk.fit_on_texts(data['clean_text'].values)
X_seq = tk.texts_to_sequences(data['clean_text'].values)  # X)
X_pad = pad_sequences(X_seq, maxlen=30, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X_pad, data['polarity'].values, test_size=0.25, random_state=1, shuffle= True)

batch_size = 50
X_train1 = X_train[batch_size:]
y_train1 = y_train[batch_size:]
X_valid = X_train[:batch_size]
y_valid = y_train[:batch_size]

# Model parameters
vocabulary_size = len(tk.word_counts.keys()) + 1
max_words = 100
embedding_size = 32

# Model description
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))

# model.add(Bidirectional(LSTM(200, return_sequences=True)))
# model.add(Bidirectional(LSTM(200, return_sequences=True)))
# model.add(Bidirectional(LSTM(200)))

model.add(LSTM(200, return_sequences=True))
model.add(LSTM(200, return_sequences=True))
model.add(LSTM(200))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train1, y_train1, shuffle=True, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=20)

# Evaluate the model performance
scores = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy : ", scores[1])

# data2 = pd.read_csv(
#     "C:/Users/HENAFF/Documents/Cours Polytech/S9 en Roumanie/Machine Learning - ML/data/cleaned_data.csv",
#     nrows=400000,
#     skiprows=41000,
#     encoding='latin-1')
# data2.columns = ['polarity', 'text', 'clean_text']
# data2['clean_text'] = data2.clean_text.astype(str)
#
# X_verif, y_verif = (data['clean_text'].values, data['polarity'].values)
#
# tk = Tokenizer(lower = True)
# tk.fit_on_texts(X_verif)
# X_verif_seq = tk.texts_to_sequences(X_verif)
# X_verif_pad = pad_sequences(X_verif_seq, maxlen=100, padding='post')
#
# # Evaluate the model performance
# scores_verif = model.evaluate(X_verif_pad, y_verif, verbose=0)
# print("Test verif accuracy : ", scores_verif[1])
