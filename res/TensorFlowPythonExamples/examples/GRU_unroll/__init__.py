# NOTE tested with TF 2.8.0
from tensorflow import keras

model = keras.Sequential()
shape = (4, 4)

model.add(keras.layers.InputLayer(input_shape=shape, batch_size=1))
model.add(keras.layers.GRU(2, input_shape=shape, unroll=True))
