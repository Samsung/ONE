# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
# Copyright (C) 2018 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import argparse
import sys


def wrap_frozen_graph(graph_def, **inouts):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    inputs = inouts["inputs"]
    outputs = inouts["outputs"]

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def main():
    if tf.__version__.find("2.") != 0:
        print("imported TensorFlow should have version >= 2.0 but " + tf.__version__)
        exit()

    parser = argparse.ArgumentParser(
        description=
        "Converting a pb file from TensorFlow v1.x to tflite file in TensorFlow 2.x")

    parser.add_argument(
        "--pb", required=True, type=str, help="path of pb file from TensorFlow 1.x")
    parser.add_argument(
        "--i",
        required=True,
        nargs='+',
        type=str,
        help="input tensor names, e.g., \"input_1:0\" \"input_2:0\"")
    parser.add_argument(
        "--o",
        required=True,
        nargs='+',
        type=str,
        help="output tensor name, e.g., \"output_1:0\" \"output_2:0\"")

    args = parser.parse_args()

    frozen_file = args.pb
    inputs = args.i
    outputs = args.o

    # Load frozen model
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(open(frozen_file, 'rb').read())

    wrap_func = wrap_frozen_graph(graph_def, inputs=inputs, outputs=outputs)

    # let's generate tflite file
    tflite_filename = frozen_file + ".tflite"

    converter = tf.lite.TFLiteConverter.from_concrete_functions([wrap_func])
    converter.experimental_new_converter = True
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    open(tflite_filename, "wb").write(tflite_model)

    print(tflite_filename + " is generated\n")


if __name__ == "__main__":
    main()
