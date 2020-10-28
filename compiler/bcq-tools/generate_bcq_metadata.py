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

import numpy as np
import tensorflow as tf

import argparse
import sys

ONE_START_MAGICNUM = int(-2e9 + 27)
ONE_END_MAGICNUM = int(2e9 - 27)


def _get_parser():
    """
    Returns an ArgumentParser for generating BCQ metadata.
    """
    parser = argparse.ArgumentParser(
        description=("Command line tool to generate metadata of BCQ nodes"))

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
    parser.add_argument(
        "-O",
        "--output_arrays",
        type=str,
        help="Original model output arrays",
        required=True)

    return parser


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


def generate_metadata_header(original_graph, bcq_version, output_arrays):
    # Generating metadata starts
    metadata_values = np.array([ONE_START_MAGICNUM])

    # Append BCQ version
    metadata_values = np.append(metadata_values, bcq_version)

    # Append original output count
    output_cnt = output_arrays.count(',') + 1
    metadata_values = np.append(metadata_values, output_cnt)

    return metadata_values


def generate_bcq_metadata_v1(flags):
    """
    BCQv1 contains following metadata.
        - The number of each BCQ information set
    """

    is_valid = True
    allowed_info_names = [
        "bcqinfo_do_w_x", "bcqinfo_alpha", "bcqinfo_packed_binary_code",
        "bcqinfo_number_of_clusters", "bcqinfo_size_of_clusters",
        "bcqinfo_qbits_of_clusters", "bcqinfo_dequant_weight"
    ]

    original_graph = load_graph(flags.input_path)
    original_graph_def = original_graph.as_graph_def()

    prefix_infonames_dict = {}

    for node in original_graph_def.node:
        if node.op == "Const" and "/bcqinfo_" in node.name:
            prefix_index = node.name.index("/bcqinfo_")
            prefix = node.name[:prefix_index]
            infoname = node.name[prefix_index + 1:]

            if infoname not in allowed_info_names:
                is_valid = False
                break

            if prefix not in prefix_infonames_dict:
                prefix_infonames_dict[prefix] = set()

            prefix_infonames_dict[prefix].add(infoname)

    # All the number of BCQ information should be same
    num_of_bcqinfo = -1
    for key in prefix_infonames_dict:
        infonames = prefix_infonames_dict[key]
        if num_of_bcqinfo == -1:
            num_of_bcqinfo = len(infonames)
        elif num_of_bcqinfo != len(infonames):
            is_valid = False

    # The number of BCQv1 information should be 6 or 7
    if num_of_bcqinfo != 6 and num_of_bcqinfo != 7:
        is_valid = False

    # If BCQ information is invalid, return original model
    if is_valid == False:
        return original_graph_def

    new_graph_def = tf.compat.v1.GraphDef()
    for node in original_graph_def.node:
        new_node = new_graph_def.node.add()
        new_node.CopyFrom(node)

    # Generate metadata header
    metadata_values = generate_metadata_header(original_graph, 1, flags.output_arrays)

    # Append metadata of BCQv1
    metadata_values = np.append(metadata_values, num_of_bcqinfo + 1)

    # Finish generating metadata
    metadata_values = np.append(metadata_values, ONE_END_MAGICNUM)

    # Generate metadata tensor
    metadata_tensor = tf.make_tensor_proto(metadata_values, tf.int32)

    new_node = new_graph_def.node.add()
    new_node.op = "Const"
    new_node.name = "one_compiler/bcqinfo_one_metadata"
    new_node.attr["dtype"].CopyFrom(
        tf.core.framework.attr_value_pb2.AttrValue(type=tf.int32.as_datatype_enum))
    new_node.attr["value"].tensor.CopyFrom(metadata_tensor)
    return new_graph_def


def determine_bcq_version(flags):
    """
    CAUTION : For now, BCQ has only one version and thus always returns 1 when BCQ
    information nodes are included. If new BCQ version is introduced,
    this function must be updated accordingly.

    When BCQ information does not exist, -1 is returned.
    """
    bcq_version = -1

    original_graph = load_graph(flags.input_path)
    original_graph_def = original_graph.as_graph_def()

    for node in original_graph_def.node:
        if node.op == "Const" and "/bcqinfo_" in node.name:
            bcq_version = 1
            break

    return bcq_version


def generate_bcq_metadata(flags):
    """
    Basic format of metadata is as following.
        - Magic number indicating start
        - Version of BCQ Format
        - The number of original outputs
        - Metadata based on each BCQ format
        - Magic number indicating end
    """
    program_version = 1
    model_version = determine_bcq_version(flags)

    if model_version == 1:
        result_graph_def = generate_bcq_metadata_v1(flags)
    elif model_version == -1:
        # When there is no BCQ information, do nothing
        result_graph_def = load_graph(flags.input_path)
    else:
        err_msg = "BCQ version of the model(v{}) ".format(model_version)
        err_msg += "is higher than "
        err_msg += "the version supported by this program(v{})".format(program_version)
        raise SystemExit(err_msg)

    tf.io.write_graph(result_graph_def, '.', flags.output_path, False)


def main():
    # Parse argument.
    parser = _get_parser()
    flags = parser.parse_known_args(args=sys.argv[1:])

    # Generate a new pb file, which BCQ metadata is included.
    generate_bcq_metadata(flags[0])


if __name__ == "__main__":
    main()
