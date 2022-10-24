# NOTE tested with TF 2.8.0
import tensorflow as tf
import numpy as np

from tensorflow import keras

model = keras.Sequential()
shape = (4, 4)

model.add(keras.layers.InputLayer(input_shape=shape, batch_size=1))
lstmf = keras.layers.LSTM(2, return_sequences=True)
lstmb = keras.layers.LSTM(2, return_sequences=True, go_backwards=True)
model.add(keras.layers.Bidirectional(lstmf, backward_layer=lstmb, input_shape=shape))
