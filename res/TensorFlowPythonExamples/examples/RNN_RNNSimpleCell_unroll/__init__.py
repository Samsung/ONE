# NOTE tested with TF 2.8.0
from tensorflow import keras

model = keras.Sequential()
shape = (4, 4)

model.add(keras.layers.InputLayer(input_shape=shape, batch_size=1))
rnncell = keras.layers.SimpleRNNCell(2)
model.add(keras.layers.RNN(rnncell, input_shape=shape, unroll=True))
