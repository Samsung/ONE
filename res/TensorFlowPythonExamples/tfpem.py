#!/usr/bin/env python

# TensorFlow Python Example Manager

import tensorflow as tf
import importlib
import argparse

from pathlib import Path
from tensorflow import keras

parser = argparse.ArgumentParser(description='Process TensorFlow Python Examples')

parser.add_argument('examples', metavar='EXAMPLES', nargs='+')

args = parser.parse_args()

output_folder = "./output/"

Path(output_folder).mkdir(parents=True, exist_ok=True)

for example in args.examples:
    print("Generate '" + example + ".pbtxt'")

    tf.compat.v1.reset_default_graph()
    # https://stackoverflow.com/questions/37808866/proper-way-to-dynamically-import-a-module-with-relative-imports
    m = importlib.import_module("examples." + example)

    with open(output_folder + example + ".pbtxt", "w") as f:
        f.write(str(tf.compat.v1.get_default_graph().as_graph_def(add_shapes=True)))

    print("Generate '" + example + ".pbtxt' - Done")

    # keras sequential?
    if hasattr(m, 'model') and isinstance(m.model, keras.Sequential):
        print("Generate '" + example + ".h5'")
        m.model.save(output_folder + example + ".h5")
        print("Generate '" + example + ".h5' - Done")

        # tflite export for experiments
        converter = tf.lite.TFLiteConverter.from_keras_model(m.model)
        converter.allow_custom_ops = True
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter._experimental_lower_tensor_list_ops = False

        tflite_model = converter.convert()
        with open(output_folder + example + ".tflite", "wb") as f:
            f.write(tflite_model)
        print("Generate '" + example + ".tflite' - Done")
