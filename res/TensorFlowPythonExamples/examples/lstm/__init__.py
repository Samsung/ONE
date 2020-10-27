import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()
shape = (4, 4)
model.add(keras.layers.LSTM(2, input_shape=shape))

# Note that this code will generate pb model only with TF 1.x.x
#
# to save model in TF 2.x.x use
# - to dump keras model: model.save("lstm.h5")
# - to dump saved model: tf.saved_model.save(model, "lstm")
