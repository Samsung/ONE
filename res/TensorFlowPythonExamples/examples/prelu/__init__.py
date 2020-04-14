import tensorflow as tf
from tensorflow.compat.v1.keras import layers

model = tf.compat.v1.keras.Sequential()
model.add(layers.PReLU())
# TODO Find a way to freeze Keras model for inference
model.build((1, 1))
