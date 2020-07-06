#!/usr/bin/env python

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

import argparse
import os
import subprocess
import sys


# NOTE Please sync options with tf2tfliteV2.py script
def _get_parser():
    """
    Returns an ArgumentParser for tf2tflite.
    """
    parser = argparse.ArgumentParser(
        description=("Command line tool to compile TF model to circle model"))

    # Converter version.
    converter_version = parser.add_mutually_exclusive_group(required=True)
    converter_version.add_argument(
        "--v1", action="store_true", help="Use TensorFlow Lite Converter 1.x")
    converter_version.add_argument(
        "--v2", action="store_true", help="Use TensorFlow Lite Converter 2.x")

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
        help="Shapes corresponding to --input_arrays, colon-separated.")
    parser.add_argument(
        "-O",
        "--output_arrays",
        type=str,
        help="Names of the output arrays, comma-separated.",
        required=True)

    return parser


def invoke_tf2tflite(args, logstream):
    cwd = os.path.dirname(os.path.abspath(__file__))
    tf2tflite_path = os.path.join(cwd, "tf2tfliteV2.py")

    input_path = args.input_path
    output_path = args.input_path + ".tflite"

    shell_args = ["python", tf2tflite_path]

    cmd_line = tf2tflite_path
    if (args.v1):
        shell_args.append("--v1")
    elif (args.v2):
        shell_args.append("--v2")

    shell_args.extend(["-i", input_path])
    shell_args.extend(["-I", args.input_arrays])
    shell_args.extend(["-s", args.input_shapes])
    shell_args.extend(["-o", output_path])
    shell_args.extend(["-O", args.output_arrays])

    result = subprocess.run(shell_args, stdout=logstream, stderr=logstream)
    if (result.returncode != 0):
        raise Exception("python tf2tfliteV2.py failed to execute")


def invoke_tflite2circle(args, logstream):
    cwd = os.path.dirname(os.path.abspath(__file__))
    tflite2circle_path = os.path.join(cwd, "tflite2circle")

    # intermediate file path
    input_path = args.input_path + ".tflite"
    output_path = args.input_path + ".circle"

    if not os.path.exists(input_path):
        raise Exception("tf2tflite conversion failed")

    shell_args = [tflite2circle_path, input_path, output_path]

    result = subprocess.run(shell_args, stdout=logstream, stderr=logstream)
    if (result.returncode != 0):
        raise Exception("tflite2circle failed to execute")

    # remove intermediate .tflite file
    os.remove(input_path)


def invoke_circle2circle(args, logstream):
    cwd = os.path.dirname(os.path.abspath(__file__))
    circle2circle_path = os.path.join(cwd, "circle2circle")

    # intermediate file path
    input_path = args.input_path + ".circle"
    output_path = args.output_path

    if not os.path.exists(input_path):
        raise Exception("tflite2circle conversion failed")

    shell_args = [circle2circle_path, "--all", input_path, output_path]

    result = subprocess.run(shell_args, stdout=logstream, stderr=logstream)
    if (result.returncode != 0):
        raise Exception("circle2circle failed to execute")

    # remove intermediate .circle file
    os.remove(input_path)


def main():
    # Parse argument.
    parser = _get_parser()

    # Check if the flags are valid.
    flags = parser.parse_known_args(args=sys.argv[1:])
    args = flags[0]

    if not os.path.exists(args.input_path):
        raise Exception("File not found: " + args.input_path)

    log_file = args.input_path + ".log"
    logstream = open(log_file, 'w')

    invoke_tf2tflite(args, logstream)
    invoke_tflite2circle(args, logstream)
    invoke_circle2circle(args, logstream)

    logstream.close()


if __name__ == "__main__":
    main()
