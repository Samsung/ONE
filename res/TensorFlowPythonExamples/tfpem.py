#!/usr/bin/env python

# TensorFlow Python Example Manager

import tensorflow as tf
import importlib
import argparse

parser = argparse.ArgumentParser(description='Process TensorFlow Python Examples')

parser.add_argument('examples', metavar='EXAMPLES', nargs='+')

args = parser.parse_args()

output_folder = "./output/"

Path(output_folder).mkdir(parents=True, exist_ok=True)

for example in args.examples:
    print("Generate '" + example + ".pbtxt'")

    tf.compat.v1.reset_default_graph()
    # https://stackoverflow.com/questions/37808866/proper-way-to-dynamically-import-a-module-with-relative-imports
    importlib.import_module("examples." + example)

    with open(output_folder + example + ".pbtxt", "w") as f:
        f.write(str(tf.compat.v1.get_default_graph().as_graph_def(add_shapes=True)))

    print("Generate '" + example + ".pbtxt' - Done")
