import tensorflow as tf
from tensorflow import keras

tf.compat.v1.disable_eager_execution()

model = keras.Sequential()
shape = (4, 4)
model.add(keras.layers.GRU(2, input_shape=shape))

# Note that this code will generate pb model only with TF 1.x.x
#
# to save model in TF 2.x.x use
# - to dump keras model: model.save("gru.h5")
# - to dump saved model: tf.saved_model.save(model, "gru")
