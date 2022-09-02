import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import subprocess

FILENAME_TFLITE_MODEL = 'gru_model_float.tflite'
FILENAME_CIRCLE_MODEL = 'gru_model_float.circle'
FILENAME_HEADER_MODEL = 'gru_model_float.h'

def generate_model():
    model = keras.Sequential()
    model.add(layers.GRU(64, input_shape=(8, 8), unroll=True,time_major=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10))
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer="sgd", metrics=["accuracy"],)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(FILENAME_TFLITE_MODEL, 'wb') as f:
        f.write(tflite_model)

def convert_tflite2circle():
    subprocess.run("../../../build/compiler/tflite2circle/tflite2circle", FILENAME_TFLITE_MODEL, FILENAME_CIRCLE_MODEL)

def convert_circle2header():
    subprocess.run("../../../../bin2c/bin2c", FILENAME_CIRCLE_MODEL, FILENAME_HEADER_MODEL, "circle_model_raw")
