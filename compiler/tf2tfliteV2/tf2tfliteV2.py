#!/usr/bin/env python3

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

from google.protobuf.message import DecodeError
from google.protobuf import text_format as _text_format


def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def _get_parser():
    """
  Returns an ArgumentParser for TensorFlow Lite Converter.
  """
    parser = argparse.ArgumentParser(
        description=("Command line tool to run TensorFlow Lite Converter."))

    # Converter version.
    converter_version = parser.add_mutually_exclusive_group(required=True)
    converter_version.add_argument(
        "--v1", action="store_true", help="Use TensorFlow Lite Converter 1.x")
    converter_version.add_argument(
        "--v2", action="store_true", help="Use TensorFlow Lite Converter 2.x")

    # Input model format
    model_format_arg = parser.add_mutually_exclusive_group()
    model_format_arg.add_argument(
        "--graph_def",
        action="store_const",
        dest="model_format",
        const="graph_def",
        help="Use graph def file(default)")
    model_format_arg.add_argument(
        "--saved_model",
        action="store_const",
        dest="model_format",
        const="saved_model",
        help="Use saved model")
    model_format_arg.add_argument(
        "--keras_model",
        action="store_const",
        dest="model_format",
        const="keras_model",
        help="Use keras model")

    # Input and output path.
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help="Full filepath of the input file.",
        required=True)
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="Full filepath of the output file.",
        required=True)

    # Input and output arrays.
    parser.add_argument(
        "-I",
        "--input_arrays",
        type=str,
        help="Names of the input arrays, comma-separated.",
        required=True)
    parser.add_argument(
        "-s",
        "--input_shapes",
        type=str,
        help=
        "Shapes corresponding to --input_arrays, colon-separated.(ex:\"1,4,4,3:1,20,20,3\")"
    )
    parser.add_argument(
        "-O",
        "--output_arrays",
        type=str,
        help="Names of the output arrays, comma-separated.",
        required=True)

    # Set default value
    parser.set_defaults(model_format="graph_def")
    return parser


def _check_flags(flags):
    """
  Checks the parsed flags to ensure they are valid.
  """
    if flags.v1:
        invalid = ""
        # To be filled

        if invalid:
            raise ValueError(invalid + " options must be used with v2")

    if flags.v2:
        if tf.__version__.find("2.") != 0:
            raise ValueError(
                "Imported TensorFlow should have version >= 2.0 but you have " +
                tf.__version__)

        invalid = ""
        # To be filled

        if invalid:
            raise ValueError(invalid + " options must be used with v1")

    if flags.input_shapes:
        if not flags.input_arrays:
            raise ValueError("--input_shapes must be used with --input_arrays")
        if flags.input_shapes.count(":") != flags.input_arrays.count(","):
            raise ValueError("--input_shapes and --input_arrays must have the same "
                             "number of items")


def _parse_array(arrays, type_fn=str):
    return list(map(type_fn, arrays.split(",")))


def _v1_convert(flags):
    if flags.model_format == "graph_def":
        input_shapes = None
        if flags.input_shapes:
            input_arrays = _parse_array(flags.input_arrays)
            input_shapes_list = [
                _parse_array(shape, type_fn=int)
                for shape in flags.input_shapes.split(":")
            ]
            input_shapes = dict(list(zip(input_arrays, input_shapes_list)))

        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            flags.input_path, _parse_array(flags.input_arrays),
            _parse_array(flags.output_arrays), input_shapes)

    if flags.model_format == "saved_model":
        converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(flags.input_path)

    if flags.model_format == "keras_model":
        converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(
            flags.input_path)

    converter.allow_custom_ops = True

    tflite_model = converter.convert()
    open(flags.output_path, "wb").write(tflite_model)


def _v2_convert(flags):
    if flags.model_format == "graph_def":
        file_content = open(flags.input_path, 'rb').read()
        try:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file_content)
        except (_text_format.ParseError, DecodeError):
            try:
                _text_format.Merge(file_content, graph_def)
            except (_text_format.ParseError, DecodeError):
                raise IOError("Unable to parse input file '{}'.".format(flags.input_path))

        wrap_func = wrap_frozen_graph(
            graph_def,
            inputs=[
                _str + ":0" if len(_str.split(":")) == 1 else _str
                for _str in _parse_array(flags.input_arrays)
            ],
            outputs=[
                _str + ":0" if len(_str.split(":")) == 1 else _str
                for _str in _parse_array(flags.output_arrays)
            ])
        converter = tf.lite.TFLiteConverter.from_concrete_functions([wrap_func])

    if flags.model_format == "saved_model":
        converter = tf.lite.TFLiteConverter.from_saved_model(flags.input_path)

    if flags.model_format == "keras_model":
        keras_model = tf.keras.models.load_model(flags.input_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    converter.allow_custom_ops = True
    converter.experimental_new_converter = True

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    tflite_model = converter.convert()
    open(flags.output_path, "wb").write(tflite_model)


def _convert(flags):
    if (flags.v1):
        _v1_convert(flags)
    else:
        _v2_convert(flags)


"""
Input frozen graph must be from TensorFlow 1.13.1
"""


def main():
    # Parse argument.
    parser = _get_parser()

    # Check if the flags are valid.
    flags = parser.parse_known_args(args=sys.argv[1:])
    _check_flags(flags[0])

    # Convert
    _convert(flags[0])


if __name__ == "__main__":
    main()
