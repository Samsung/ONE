# NOTE tested with TF 2.8.0
from tensorflow import keras

model = keras.Sequential()
shape = (4, 4)

model.add(keras.layers.InputLayer(input_shape=shape, batch_size=1))
lstmcell = keras.layers.LSTMCell(2)
model.add(keras.layers.RNN(lstmcell, input_shape=shape, unroll=True))

# NOTE refer https://github.com/Samsung/ONE/issues/9895#issuecomment-1289820894
