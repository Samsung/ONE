#!/usr/bin/env python3

# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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


# This function is copied from
# https://github.com/tensorflow/tensorflow/blob/r2.3/tensorflow/examples/label_image/label_image.py#L26
def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")

    return graph


def get_bcq_version(input_path):
    """
    If BCQ metadata exists, BCQ version is in the second element.
    Return -1 when the metadata is not found.
    """
    graph = load_graph(input_path)
    graph_def = graph.as_graph_def()
    for node in graph_def.node:
        if node.op == "Const" and "one_compiler/bcqinfo_one_metadata" in node.name:
            metadata_tensor = tf.make_ndarray(node.attr["value"].tensor)
            return metadata_tensor[1]
    return -1


def get_bcqinfo_output_arrays_v1(input_path, output_arrays):
    """
    This function generates a file which includes output arrays of BCQ v1
    information bundles. Each bundle is consisted with one of candidate
    operations (BCQ may be applied) and BCQ constant nodes related with
    the operation.
    """
    graph = load_graph(input_path)
    ops = graph.get_operations()

    # If there is a constant node named PREFIX_1/bcqinfo_alpha,
    # it is used for applying BCQ to constant node named PREFIX_1.
    # Collected prefixes will be used for connecting
    # bcqinfo nodes and user operations of prefix nodes.
    prefix_set = set()
    has_dequant_weight = False
    for op in ops:
        if op.type == "Const" and "/bcqinfo_" in op.outputs[0].name:
            # Metadata do not have prefix
            if "one_compiler/bcqinfo_one_metadata" in op.outputs[0].name:
                continue

            prefix_index = op.outputs[0].name.index("/bcqinfo_")
            prefix = op.outputs[0].name[:prefix_index]
            prefix_set.add(prefix)

            # Usually, output name of op is like "outputname:0"
            # -2 is for removing ":0"
            infoname = op.outputs[0].name[prefix_index + 1:-2]
            if infoname == "bcqinfo_dequant_weight":
                has_dequant_weight = True

    # the name of metadata node
    ret_output_arrays = ['one_compiler/bcqinfo_one_metadata']

    # given node from user
    ret_output_arrays.append += output_arrays.split(',')

    # all pairs of a constant node and related BCQ information nodes.
    for prefix in prefix_set:
        ret_output_arrays.append(prefix + '/bcqinfo_do_w_x')
        ret_output_arrays.append(prefix + '/bcqinfo_alpha')
        ret_output_arrays.append(prefix + '/bcqinfo_packed_binary_code')
        ret_output_arrays.append(prefix + '/bcqinfo_number_of_clusters')
        ret_output_arrays.append(prefix + '/bcqinfo_size_of_clusters')
        ret_output_arrays.append(prefix + '/bcqinfo_qbits_of_clusters')
        ret_output_arrays.append(prefix)
        if has_dequant_weight:
            ret_output_arrays.append(prefix + '/bcqinfo_dequant_weight')

    return ret_output_arrays


def get_bcq_output_arrays(input_path, output_arrays):
    """Returns BCQ output arrays that the model from input_path has"""
    program_version = 1
    model_version = get_bcq_version(input_path)

    if model_version == 1:
        return get_bcqinfo_output_arrays_v1(input_path, output_arrays)
    elif model_version == -1:
        return output_arrays.split(',')
    else:
        err_msg = "BCQ version of the model(v{}) ".format(model_version)
        err_msg += "is higher than "
        err_msg += "the version supported by this program(v{})".format(program_version)
        raise SystemExit(err_msg)
