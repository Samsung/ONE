# NOTE tested with TF 2.8.0
from tensorflow import keras

model = keras.Sequential()
shape = (4, 4)

model.add(keras.layers.InputLayer(input_shape=shape, batch_size=1))
model.add(keras.layers.LSTM(2, input_shape=shape))

# NOTE refer https://github.com/Samsung/ONE/issues/9895#issuecomment-1289766546
