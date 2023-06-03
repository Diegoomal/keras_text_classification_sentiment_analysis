# https://medium.com/analytics-vidhya/understanding-embedding-layer-in-keras-bbe3ff1327ce

# Understanding Embedding Layer in Keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import numpy as np

model = Sequential()

model.add(Embedding(input_dim=10,output_dim=4,input_length=2))

model.compile('adam','mse')

input_data = np.array([[1,2]])
pred = model.predict(input_data)

print("input_data.shape:", input_data.shape)
print("pred:", pred[0][0])
