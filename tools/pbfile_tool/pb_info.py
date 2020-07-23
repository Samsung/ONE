#!/usr/bin/python

# Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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
from google.protobuf import text_format
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

import argparse
import os


def splitDirFilenameExt(path):
    # in case of '/tmp/.ssh/my.key.dat'
    # this returns ('/tmp/.ssh', 'my.key', 'dat')
    directory = os.path.split(path)[0]
    ext = os.path.splitext(path)[1][1:]  # remove '.', e.g., '.dat' -> 'dat'
    filename = os.path.splitext(os.path.split(path)[1])[0]

    return (directory, filename, ext)


def importGraphIntoSession(sess, filename):
    # this should be called inside
    # with tf.Session() as sess:
    assert sess
    (_, _, ext) = splitDirFilenameExt(filename)
    if (ext.lower() == 'pb'):
        with tf.gfile.GFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

    elif (ext.lower() == 'pbtxt'):
        with open(filename, 'r') as reader:
            graph_def = tf.GraphDef()
            text_format.Parse(reader.read(), graph_def)
    else:
        print("# Error: unknown extension - " + ext)

    tf.import_graph_def(graph_def)


def print_operation(op, op_count):
    print("")  # new line
    print("OP #{}: {}, name = {}".format(op_count, op.type, op.name))

    print("\tinputs:")
    for input_tensor in op.inputs:
        print("\t\t{} : name = {}".format(input_tensor.shape, input_tensor.name))

    print("\toutputs:")
    for output_tensor in op.outputs:
        print("\t\t{}, name = {}".format(output_tensor.shape, output_tensor.name))

    print("\tattributes:")
    op_def = op.op_def
    for attr_def in op.op_def.attr:
        attr = op.get_attr(attr_def.name)
        # skip Const value
        if op.type == "Const" and attr_def.name == "value":
            print("\t\t{}, name = {}".format("skipping value", attr_def.name))
        else:
            print("\t\t{}, name = {}".format(attr, attr_def.name))
    print("")  # new line


def print_graph_info(pb_path, optype_substring, name_prefix):
    with tf.Session() as sess:
        importGraphIntoSession(sess, pb_path)

        op_seq = 1
        op_count = 1
        graph = sess.graph
        ops = graph.get_operations()
        for op in ops:
            if optype_substring == "*" and (name_prefix == None
                                            or op.name.startswith(name_prefix)):
                print_operation(op, op_seq)
                op_count += 1
            elif op.type.lower().find(optype_substring.lower()) != -1 and (
                    name_prefix == None or op.name.startswith(name_prefix)):
                print_operation(op, op_seq)
                op_count += 1
            else:
                print("skipping {}, name = {}".format(op.type, op.name))
            op_seq += 1

        print("")
        print("Total number of operations : " + str(op_count))
        print("")


def print_summary(pb_path, optype_substring, name_prefix):
    op_map = {}
    op_count = 0
    with tf.Session() as sess:
        importGraphIntoSession(sess, pb_path)

        graph = sess.graph
        ops = graph.get_operations()
        for op in ops:
            process = False
            if optype_substring == "*" and (name_prefix == None
                                            or op.name.startswith(name_prefix)):
                process = True
            elif op.type.lower().find(optype_substring.lower()) != -1 and (
                    name_prefix == None or op.name.startswith(name_prefix)):
                process = True

            if process:
                op_count += 1
                if op_map.get(op.type) == None:
                    op_map[op.type] = 1
                else:
                    op_map[op.type] += 1

        # print op list
        print("")
        for op_type, count in op_map.items():
            print("\t" + op_type + " : \t" + str(count))
        print("")
        print("Total number of operations : " + str(op_count))
        print("Total number of operation types : " + str(len(op_map.keys())))
        print("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prints information inside pb file')

    parser.add_argument("pb_file", help="pb file to read")
    parser.add_argument(
        "op_subst",
        help="substring of operations. only info of these operasions will be printed.")
    parser.add_argument("--summary",
                        help="print summary of operations",
                        action="store_true")
    parser.add_argument("--name_prefix", help="filtered by speficied name prefix")

    args = parser.parse_args()

    if args.summary:
        print_summary(args.pb_file, args.op_subst, args.name_prefix)
    else:
        print_graph_info(args.pb_file, args.op_subst, args.name_prefix)
