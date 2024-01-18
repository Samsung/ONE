#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
import onecc


# https://github.com/keras-team/keras/blob/v2.15.0/keras/applications/mobilenet_v2.py#L520C1-L524C13
# https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py#L413
def correct_pad(input_shape, kernel_size):
    img_dim = 1
    input_size = input_shape[img_dim : (img_dim + 2)]
    kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )


# https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
def createModel(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.ZeroPadding2D(
            padding=correct_pad(input_shape, 3)
        ),
    ])
    return model


def saveTfliteModel(out_dir, name):
    converter = tf.lite.TFLiteConverter.from_saved_model(out_dir)
    tflite_model = converter.convert()
    tflite_path = os.path.join(out_dir, name + '.tflite')
    with open(tflite_path, 'wb') as f:
          f.write(tflite_model)


def saveCircleModel(out_dir, name):
    tflite_path = os.path.join(out_dir, name + '.tflite')
    circle = onecc.import_tflite(tflite_path)
    circle_path = os.path.join(out_dir, name + '.circle')
    circle.save(circle_path)


if __name__ == "__main__":
    name = 'pad'

    input_shape = (1, 28, 28, 1)
    model = createModel(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.experimental.SGD(0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )

    # Just use `fit` as usual
    model.fit(
        x = np.random.random((1, 28, 28, 1)),
        y = np.random.random((1, 1)),
        epochs=1)

    model.summary()

    out_dir = 'op.' + name
    model.save(out_dir)
    saveTfliteModel(out_dir, name)
    saveCircleModel(out_dir, name)
