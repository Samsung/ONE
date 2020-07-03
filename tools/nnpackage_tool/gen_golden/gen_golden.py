#!/usr/bin/env python3

# Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import argparse
import numpy as np


# cmd arguments parsing
def usage():
    script = os.path.basename(os.path.basename(__file__))
    print("Usage: {} path_to_pb".format(script))
    sys.exit(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'modelfile',
        type=str,
        help='path to modelfile in either graph_def (.pb) or tflite (.tflite)')
    parser.add_argument(
        '-o', '--output', action='store', dest="out_dir", help="output directory")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.parse_args()
        sys.exit(1)

    filename = args.modelfile

    if args.out_dir:
        out_dir = args.out_dir + '/'
    else:
        out_dir = "./"

    _, extension = os.path.splitext(filename)

    input_names = []
    output_names = []
    input_dtypes = []
    output_dtypes = []
    input_values = []
    output_values = []

    if extension == ".pb":
        # import graph_def (pb)
        graph = tf.compat.v1.get_default_graph()
        graph_def = tf.compat.v1.GraphDef()

        with tf.io.gfile.GFile(filename, 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        # identify input namess and output names
        ops = graph.get_operations()
        input_names = [op.outputs[0].name for op in ops if op.type == "Placeholder"]
        output_names = [tensor.name for op in ops for tensor in op.outputs]
        for op in ops:
            for t in op.inputs:
                if t.name in output_names:
                    output_names.remove(t.name)

        # identify input dtypes and output dtypes
        input_dtypes = [graph.get_tensor_by_name(name).dtype for name in input_names]
        output_dtypes = [graph.get_tensor_by_name(name).dtype for name in output_names]

        # gen random input values
        for idx in range(len(input_names)):
            this_shape = graph.get_tensor_by_name(input_names[idx]).shape
            this_dtype = input_dtypes[idx]
            if this_dtype == np.uint8:
                input_values.append(
                    np.array(np.random.randint(0, 255, this_shape, this_dtype)))
            elif this_dtype == np.float32:
                input_values.append(
                    np.random.random_sample(this_shape).astype(np.float32))
            elif this_dtype == np.bool_:
                # generate random integer from [0, 2)
                input_values.append(
                    np.array(np.random.randint(2, size=this_shape), this_dtype))
            elif this_dtype == np.int32:
                input_values.append(
                    np.array(np.random.randint(0, 99, this_shape, "int32")))
            elif this_dtype == np.int64:
                input_values.append(
                    np.array(np.random.randint(0, 99, this_shape, "int64")))

        # get output values by running
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as sess:
            output_values = sess.run(
                output_names, feed_dict=dict(zip(input_names, input_values)))

    elif extension == ".tflite":
        # load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(filename)
        interpreter.allocate_tensors()

        # get list of tensors details for input/output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # identify input namess and output names
        input_names = [d['name'] for d in input_details]
        output_names = [d['name'] for d in output_details]

        # identify input dtypes and output dtypes
        input_dtypes = [d['dtype'] for d in input_details]
        output_dtypes = [d['dtype'] for d in output_details]

        # gen random input values and set tensor
        for idx in range(len(input_details)):
            this_shape = input_details[idx]['shape']
            this_dtype = input_details[idx]['dtype']
            if this_dtype == np.uint8:
                input_values.append(
                    np.array(np.random.randint(0, 255, this_shape, this_dtype)))
            elif this_dtype == np.float32:
                input_values.append(
                    np.random.random_sample(this_shape).astype(np.float32))
            elif this_dtype == np.bool_:
                # generate random integer from [0, 2)
                input_values.append(
                    np.array(np.random.randint(2, size=this_shape), this_dtype))
            elif this_dtype == np.int32:
                input_values.append(
                    np.array(np.random.randint(0, 99, this_shape, this_dtype)))
            elif this_dtype == np.int64:
                input_values.append(
                    np.array(np.random.randint(0, 99, this_shape, this_dtype)))
            interpreter.set_tensor(input_details[idx]['index'], input_values[idx])

        # get output values by running
        interpreter.invoke()
        for idx in range(len(output_details)):
            output_values.append(interpreter.get_tensor(output_details[idx]['index']))

    else:
        print("Only .pb and .tflite models are supported.")
        sys.exit(-1)

    # dump input and output in h5
    import h5py
    supported_dtypes = ("float32", "uint8", "bool", "int32", "int64")
    h5dtypes = {
        "float32": ">f4",
        "uint8": "u1",
        "bool": "u1",
        "int32": "int32",
        "int64": "int64"
    }
    with h5py.File(out_dir + "input.h5", 'w') as hf:
        name_grp = hf.create_group("name")
        val_grp = hf.create_group("value")
        for idx, t in enumerate(input_names):
            dtype = tf.compat.v1.as_dtype(input_dtypes[idx])
            if not dtype.name in supported_dtypes:
                print("ERR: Supported input types are {}".format(supported_dtypes))
                sys.exit(-1)
            val_grp.create_dataset(
                str(idx), data=input_values[idx], dtype=h5dtypes[dtype.name])
            name_grp.attrs[str(idx)] = input_names[idx]

    with h5py.File(out_dir + "expected.h5", 'w') as hf:
        name_grp = hf.create_group("name")
        val_grp = hf.create_group("value")
        for idx, t in enumerate(output_names):
            dtype = tf.compat.v1.as_dtype(output_dtypes[idx])
            if not dtype.name in supported_dtypes:
                print("ERR: Supported output types are {}".format(supported_dtypes))
                sys.exit(-1)
            val_grp.create_dataset(
                str(idx), data=output_values[idx], dtype=h5dtypes[dtype.name])
            name_grp.attrs[str(idx)] = output_names[idx]
