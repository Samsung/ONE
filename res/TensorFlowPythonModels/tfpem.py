#!/usr/bin/env python

# TensorFlow Python Example Manager

import tensorflow as tf
import importlib
import argparse

parser = argparse.ArgumentParser(description='Process TensorFlow Python Models')

parser.add_argument('--mode', metavar='MODE', choices=['pbtxt'], default='pbtxt')
parser.add_argument('examples', metavar='EXAMPLES', nargs='+')

args = parser.parse_args()

if args.mode == 'pbtxt':
    for example in args.examples:
        print("Generate '" + example + ".pbtxt'")

        tf.compat.v1.reset_default_graph()
        # https://stackoverflow.com/questions/37808866/proper-way-to-dynamically-import-a-module-with-relative-imports
        importlib.import_module("examples." + example)

        with open(example + ".pbtxt", "w") as f:
            f.write(str(tf.compat.v1.get_default_graph().as_graph_def(add_shapes=True)))

        print("Generate '" + example + ".pbtxt' - Done")
