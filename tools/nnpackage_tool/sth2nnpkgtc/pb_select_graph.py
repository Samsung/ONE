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

import tensorflow as tf
import tensorflow.tools.graph_transforms as gt
import os
import sys
import argparse


# cmd arguments parsing
def usage():
    script = os.path.basename(os.path.basename(__file__))
    print("Usage: {} path_to_pb".format(script))
    sys.exit(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_def', type=str, help='path to graph_def (pb)')
    parser.add_argument('input_names', type=str, help='input tensor names separated by ,')
    parser.add_argument('output_names',
                        type=str,
                        help='output tensor names separated by ,')
    parser.add_argument('graph_outname',
                        type=str,
                        help='graph_def base name for selected subgraph')
    parser.add_argument('-o',
                        '--output',
                        action='store',
                        dest="out_dir",
                        help="output directory")
    args = parser.parse_args()

    filename = args.graph_def
    input_names = args.input_names.split(",")
    output_names = args.output_names.split(",")
    newfilename = args.graph_outname

    if args.out_dir:
        out_dir = args.out_dir + '/'
    else:
        out_dir = "./"

    # import graph_def (pb)
    graph = tf.compat.v1.get_default_graph()
    graph_def = tf.compat.v1.GraphDef()

    with tf.io.gfile.GFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    transforms = ['remove_nodes(op=Identity, op=CheckNumerics)', 'strip_unused_nodes']

    selected_graph_def = tf.tools.graph_transforms.TransformGraph(
        graph_def, input_names, output_names, transforms)

    tf.io.write_graph(selected_graph_def, out_dir, newfilename + ".pb", as_text=False)
