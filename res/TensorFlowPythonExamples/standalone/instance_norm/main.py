import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers

model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=(32, 32), name="input_layer"))
model.add(tfa.layers.InstanceNormalization())

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.allow_custom_ops = True
tflite_model = converter.convert()
tflite_file = './instance_norm.tflite'
open(tflite_file, "wb").write(tflite_model)
