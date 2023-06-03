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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer

# consts

PATH_VECT = "src/artefatos/count_vectorizer_6.pkl"
PATH_MODEL = "src/artefatos/modelo_treinado_6.h5"

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

print("\n=== show dataset (df_dataset) ===")

print(f"\ndf_dataset.head()\n{df_dataset.head()}")

print(f"\ndf_dataset.tail()\n{df_dataset.tail()}\n")

#



#

df_dataset_X = df_dataset['sentence'].values
df_dataset_y = df_dataset['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    df_dataset_X, df_dataset_y, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(X_train)

with open(PATH_VECT, 'wb') as file:
    pickle.dump(vectorizer, file)

X_train = vectorizer.transform(X_train)
X_test  = vectorizer.transform(X_test)

#

model = Sequential()
model.add(layers.Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    batch_size=10)

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=False)
print(f"\nevaluate -> Train -> Loss: {train_loss} - Accuracy: {train_acc}")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=False)
print(f"\nevaluate -> Test -> Loss: {test_loss} - Accuracy: {test_acc}")

model.save(PATH_MODEL)

# Predict

sentences = [
    'Rashmi likes ice cream',
    'Rashmi hates chocolate.'
]

print(f"\nSentence:{sentences}")

vectorizer = None
with open('src/artefatos/count_vectorizer_4.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

vectorized_sentences = vectorizer.transform(sentences).toarray()

model = load_model('src/artefatos/modelo_treinado_4.h5')

predicted = model.predict(vectorized_sentences)

print(f"\nPredict: {predicted[0][0]} - Sentense: '{sentences[0]}'")
print(f"\nPredict: {predicted[1][0]} - Sentense: '{sentences[1]}'")

print("\n")
