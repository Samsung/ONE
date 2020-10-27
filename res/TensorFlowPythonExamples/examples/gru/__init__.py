import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.GRU(2, input_shape=(4,4,)))

# Note that this code will generate pb model only with TF 1.x.x
#
# to save model in TF 2.x.x use
# - to dump keras model: model.save("gru.h5")
# - to dump saved model: tf.saved_model.save(model, "gru")
