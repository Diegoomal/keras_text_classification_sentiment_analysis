import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Flatten, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer

# consts

PATH_ENC = "src/artefatos/encoded_sentences_9.pkl"
PATH_VECT = "src/artefatos/count_vectorizer_9.pkl"
PATH_MODEL = "src/artefatos/modelo_treinado_9.h5"

# Create Dataset

filepath_dict = {
    'imdb':   'src/assets/imdb_labelled.txt',
    'yelp':   'src/assets/yelp_labelled.txt',
    'amazon': 'src/assets/amazon_cells_labelled.txt'
}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source
    df_list.append(df)

df_dataset = pd.concat(df_list)
df_dataset.iloc[0]

# print("\n=== show df_dataset (raw) ===\n")
# print(f"\ndf_dataset.head()\n{df_dataset.head()}\n")
# print(f"\ndf_dataset.tail()\n{df_dataset.tail()}\n")

df_dataset.sort_values(by='label', inplace=True)

# print("\n=== show df_dataset (sorted) ===\n")
# print(f"\ndf_dataset.head()\n{df_dataset.head()}\n")
# print(f"\ndf_dataset.tail()\n{df_dataset.tail()}\n")

#

# print('str.len.max:', df_dataset['sentence'].str.len().max())

#

vocab_size = 8000
encoded_sentences = [one_hot(d, vocab_size) for d in df_dataset['sentence']]
print(f'\nEncoded sentences:\n{encoded_sentences}\n')

# salvamento
with open(PATH_ENC, 'wb') as file:
    pickle.dump(encoded_sentences, file)

# # load
# encoded_sentences = None
# with open(PATH_ENC, 'rb') as file:
#     encoded_sentences = pickle.load(file)

#

max_length = 4

padded_reviews = pad_sequences(encoded_sentences,
                               maxlen=max_length,
                               padding='post')

model = Sequential()

embedding_layer = Embedding(input_dim=vocab_size,
                            output_dim=8,
                            input_length=max_length)

model.add(embedding_layer)
model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

print(model.summary())

model.fit(padded_reviews, df_dataset['label'], epochs=100, verbose=0)

model.save(PATH_MODEL)

print("\nEmbedding_layer.get_weights()[0].shape:",
      embedding_layer.get_weights()[0].shape, "\n")

# prediction

# model = None

# model = load_model(PATH_MODEL)

sentences = [
    'rashmi loves candies',
    'rashmi likes ice cream',
    'rashmi hates chocolate.'
]

encoded_sentences = [one_hot(sentence, vocab_size) for sentence in sentences]

padded_sentences = pad_sequences(encoded_sentences,
                                 maxlen=max_length,
                                 padding='post')

predicted = model.predict(padded_sentences)

print(f"\nPredict: {predicted[0][0]} - Sentense: '{sentences[0]}'")
print(f"\nPredict: {predicted[1][0]} - Sentense: '{sentences[1]}'")
print(f"\nPredict: {predicted[2][0]} - Sentense: '{sentences[2]}'")
print("\n")
