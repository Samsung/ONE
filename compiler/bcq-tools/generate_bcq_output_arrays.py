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


def _get_parser():
    """
    Returns an ArgumentParser for generating output_arrays.
    """
    parser = argparse.ArgumentParser(
        description=("Command line tool to generated output_arrays of BCQ nodes"))

    # Input and output path.
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help="Full filepath of the input file.",
        required=True)
    parser.add_argument(
        "-m",
        "--metadata_path",
        type=str,
        help="Full filepath for the file that provides metadata.",
        required=True)
    parser.add_argument(
        "-A",
        "--output_arrays_path",
        type=str,
        help="Full filepath for the file that provides output arrays",
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


def find_bcq_version(flags):
    """
    If BCQ metadata exists, BCQ version is in the second element.
    Return -1 when the metadata is not found.
    """
    graph = load_graph(flags.input_path)
    graph_def = graph.as_graph_def()
    for node in graph_def.node:
        if node.op == "Const" and "one_compiler/bcqinfo_one_metadata" in node.name:
            metadata_tensor = tf.make_ndarray(node.attr["value"].tensor)
            return metadata_tensor[1]
    return -1


def print_bcqinfo_output_arrays_v1(flags):
    """
    This function generates a file which includes output arrays of BCQ v1
    information bundles. Each bundle is consisted with one of candidate
    operations (BCQ may be applied) and BCQ constant nodes related with
    the operation.
    """
    graph = load_graph(flags.input_path)
    graph_def = graph.as_graph_def()
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

    # Ideal situation is that the user nodes of BCQ applicable constant nodes
    # are BCQ applicable operations such as MatMul, GatherV2, etc.
    # However, operations which do not change original values such as
    # Ideneity or Transpose can exist between them. In view of TensorFlow Lite,
    # real user nodes of BCQ applicable constant nodes must be found first.
    # This work is done by BFS search with queue.

    prefix_node_dict = {}  # key : prefix / value : list of candidates
    matmul_node_prefix_dict = {}  # key : Name of MatMul node / value : prefix

    queue_prefix = list(prefix_set)
    queue_nodename = [queue_prefix[idx] + ":0" for idx in range(len(queue_prefix))]

    while len(queue_prefix) > 0:
        prefix = queue_prefix.pop(0)
        nodename = queue_nodename.pop(0)
        if prefix not in prefix_node_dict.keys():
            prefix_node_dict[prefix] = []

        # Usually, output name of op is like "outputname:0"
        # -2 is for removing ":0"
        for op in ops:
            if op.type == "MatMul" and (op.inputs[0].name == nodename
                                        or op.inputs[1].name == nodename):
                prefix_node_dict[prefix].append(op.outputs[0].name[:-2])
                matmul_node_prefix_dict[op.outputs[0].name[:-2]] = prefix
            elif op.type == "Einsum" and (op.inputs[0].name == nodename
                                          or op.inputs[1].name == nodename):
                prefix_node_dict[prefix].append(op.outputs[0].name[:-2])
            elif op.type == "GatherV2" and op.inputs[0].name == nodename:
                prefix_node_dict[prefix].append(op.outputs[0].name[:-2])
            elif len(op.outputs) == 1:
                for i in range(len(op.inputs)):
                    if op.inputs[i].name == nodename:
                        queue_prefix.append(prefix)
                        queue_nodename.append(op.outputs[0].name)
                        break

    # When TensorFlow model is converted to TensorFlow Lite model,
    # more than one operation can be fused as one.
    # For example, MatMul + BiasAdd + ReLU in TensorFlow can be fused as
    # one FullyConnected in TensorFlow Lite.
    # It means that even real user nodes of BCQ applicable constant nodes
    # in TensorFlow are found, they may be real user nodes in TensorFlow Lite.
    # Therefore additional candidates of real user nodes should be found either.
    # Finding additional candidates is done by BFS search with queue.

    fuseop_prefix_dict = {}  # key : Candidate operation / Value : prefix

    # These ops can be candidate. However other candidates may exists after these ops.
    mark_type = ["Add", "AddV2", "BiasAdd", "Reshape", "Transpose"]

    # These ops can be candidate. And no more candidates will be found after these ops.
    mark_and_stop_type = ["Relu", "Relu6", "Tanh"]

    # These ops cannot be candidates but other candidates may exists after these ops.
    # NOTE : Some of following ops may be removed from the list but not sure for now.
    pass_type = [
        "BatchToSpaceND", "Cast", "DepthToSpace", "ExpandDims", "ResizeBilinear",
        "ResizeNearestNeighbor", "ScatterNd", "SpaceToBatchND", "SpaceToDepth", "Squeeze",
        "Identity", "Pack", "Unpack", "Stack"
    ]

    queue_prefix = list(matmul_node_prefix_dict.values())
    queue_nodename = [matmul + ":0" for matmul in matmul_node_prefix_dict.keys()]

    visited_nodes = set(queue_nodename)
    while len(queue_prefix) > 0:
        prefix = queue_prefix.pop(0)
        nodename = queue_nodename.pop(0)

        # Usually, output name of op is like "outputname:0"
        # -2 is for removing ":0"
        for op in ops:
            for i in range(len(op.inputs)):
                if nodename == op.inputs[i].name:
                    if op.type in mark_type:
                        if op.outputs[0].name[:-2] not in fuseop_prefix_dict.keys():
                            fuseop_prefix_dict[op.outputs[0].name[:-2]] = set()
                        fuseop_prefix_dict[op.outputs[0].name[:-2]].add(prefix)
                        if op.outputs[0].name not in visited_nodes:
                            queue_prefix.append(prefix)
                            queue_nodename.append(op.outputs[0].name)
                            visited_nodes.add(op.outputs[0].name)
                    elif op.type in mark_and_stop_type:
                        if op.outputs[0].name[:-2] not in fuseop_prefix_dict.keys():
                            fuseop_prefix_dict[op.outputs[0].name[:-2]] = set()
                        fuseop_prefix_dict[op.outputs[0].name[:-2]].add(prefix)
                    elif op.type in pass_type and op.outputs[0].name not in visited_nodes:
                        queue_prefix.append(prefix)
                        queue_nodename.append(op.outputs[0].name)
                        visited_nodes.add(op.outputs[0].name)

    # Write the name of metadata node
    with open(flags.metadata_path, 'w') as f_metadata:
        f_metadata.write("one_compiler/bcqinfo_one_metadata,")

    # Write all pairs of candidate operations and related BCQ information nodes.
    with open(flags.output_arrays_path, 'w') as f_arrays:
        for prefix in prefix_set:
            for fusable_op in prefix_node_dict[prefix]:
                f_arrays.write("," + prefix + "/bcqinfo_do_w_x")
                f_arrays.write("," + prefix + "/bcqinfo_alpha")
                f_arrays.write("," + prefix + "/bcqinfo_packed_binary_code")
                f_arrays.write("," + prefix + "/bcqinfo_number_of_clusters")
                f_arrays.write("," + prefix + "/bcqinfo_size_of_clusters")
                f_arrays.write("," + prefix + "/bcqinfo_qbits_of_clusters")
                f_arrays.write("," + fusable_op)
                if has_dequant_weight:
                    f_arrays.write("," + prefix + "/bcqinfo_dequant_weight")
        for fuseop in fuseop_prefix_dict.keys():
            if len(fuseop_prefix_dict[fuseop]) == 1:
                prefix = fuseop_prefix_dict[fuseop].pop()
                f_arrays.write("," + prefix + "/bcqinfo_do_w_x")
                f_arrays.write("," + prefix + "/bcqinfo_alpha")
                f_arrays.write("," + prefix + "/bcqinfo_packed_binary_code")
                f_arrays.write("," + prefix + "/bcqinfo_number_of_clusters")
                f_arrays.write("," + prefix + "/bcqinfo_size_of_clusters")
                f_arrays.write("," + prefix + "/bcqinfo_qbits_of_clusters")
                f_arrays.write("," + fuseop)
                if has_dequant_weight:
                    f_arrays.write("," + prefix + "/bcqinfo_dequant_weight")


def print_bcq_output_arrays(flags):
    program_version = 1
    model_version = find_bcq_version(flags)

    if model_version == 1:
        print_bcqinfo_output_arrays_v1(flags)
    elif model_version == -1:
        # When BCQ information not found, print nothing.
        f_metadata = open(flags.metadata_path, 'w')
        f_arrays = open(flags.output_arrays_path, 'w')
        f_metadata.close()
        f_arrays.close()
    else:
        err_msg = "BCQ version of the model(v{}) ".format(model_version)
        err_msg += "is higher than "
        err_msg += "the version supported by this program(v{})".format(program_version)
        raise SystemExit(err_msg)


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
    graph_def = graph.as_graph_def()
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

    # Ideal situation is that the user nodes of BCQ applicable constant nodes
    # are BCQ applicable operations such as MatMul, GatherV2, etc.
    # However, operations which do not change original values such as
    # Ideneity or Transpose can exist between them. In view of TensorFlow Lite,
    # real user nodes of BCQ applicable constant nodes must be found first.
    # This work is done by BFS search with queue.

    prefix_node_dict = {}  # key : prefix / value : list of candidates
    matmul_node_prefix_dict = {}  # key : Name of MatMul node / value : prefix

    queue_prefix = list(prefix_set)
    queue_nodename = [queue_prefix[idx] + ":0" for idx in range(len(queue_prefix))]

    while len(queue_prefix) > 0:
        prefix = queue_prefix.pop(0)
        nodename = queue_nodename.pop(0)
        if prefix not in prefix_node_dict.keys():
            prefix_node_dict[prefix] = []

        # Usually, output name of op is like "outputname:0"
        # -2 is for removing ":0"
        for op in ops:
            if op.type == "MatMul" and (op.inputs[0].name == nodename
                                        or op.inputs[1].name == nodename):
                prefix_node_dict[prefix].append(op.outputs[0].name[:-2])
                matmul_node_prefix_dict[op.outputs[0].name[:-2]] = prefix
            elif op.type == "Einsum" and (op.inputs[0].name == nodename
                                          or op.inputs[1].name == nodename):
                prefix_node_dict[prefix].append(op.outputs[0].name[:-2])
            elif op.type == "GatherV2" and op.inputs[0].name == nodename:
                prefix_node_dict[prefix].append(op.outputs[0].name[:-2])
            elif len(op.outputs) == 1:
                for i in range(len(op.inputs)):
                    if op.inputs[i].name == nodename:
                        queue_prefix.append(prefix)
                        queue_nodename.append(op.outputs[0].name)
                        break

    # When TensorFlow model is converted to TensorFlow Lite model,
    # more than one operation can be fused as one.
    # For example, MatMul + BiasAdd + ReLU in TensorFlow can be fused as
    # one FullyConnected in TensorFlow Lite.
    # It means that even real user nodes of BCQ applicable constant nodes
    # in TensorFlow are found, they may be real user nodes in TensorFlow Lite.
    # Therefore additional candidates of real user nodes should be found either.
    # Finding additional candidates is done by BFS search with queue.

    fuseop_prefix_dict = {}  # key : Candidate operation / Value : prefix

    # These ops can be candidate. However other candidates may exists after these ops.
    mark_type = ["Add", "AddV2", "BiasAdd", "Reshape", "Transpose"]

    # These ops can be candidate. And no more candidates will be found after these ops.
    mark_and_stop_type = ["Relu", "Relu6", "Tanh"]

    # These ops cannot be candidates but other candidates may exists after these ops.
    # NOTE : Some of following ops may be removed from the list but not sure for now.
    pass_type = [
        "BatchToSpaceND", "Cast", "DepthToSpace", "ExpandDims", "ResizeBilinear",
        "ResizeNearestNeighbor", "ScatterNd", "SpaceToBatchND", "SpaceToDepth", "Squeeze",
        "Identity", "Pack", "Unpack", "Stack"
    ]

    queue_prefix = list(matmul_node_prefix_dict.values())
    queue_nodename = [matmul + ":0" for matmul in matmul_node_prefix_dict.keys()]

    visited_nodes = set(queue_nodename)
    while len(queue_prefix) > 0:
        prefix = queue_prefix.pop(0)
        nodename = queue_nodename.pop(0)

        # Usually, output name of op is like "outputname:0"
        # -2 is for removing ":0"
        for op in ops:
            for i in range(len(op.inputs)):
                if nodename == op.inputs[i].name:
                    if op.type in mark_type:
                        if op.outputs[0].name[:-2] not in fuseop_prefix_dict.keys():
                            fuseop_prefix_dict[op.outputs[0].name[:-2]] = set()
                        fuseop_prefix_dict[op.outputs[0].name[:-2]].add(prefix)
                        if op.outputs[0].name not in visited_nodes:
                            queue_prefix.append(prefix)
                            queue_nodename.append(op.outputs[0].name)
                            visited_nodes.add(op.outputs[0].name)
                    elif op.type in mark_and_stop_type:
                        if op.outputs[0].name[:-2] not in fuseop_prefix_dict.keys():
                            fuseop_prefix_dict[op.outputs[0].name[:-2]] = set()
                        fuseop_prefix_dict[op.outputs[0].name[:-2]].add(prefix)
                    elif op.type in pass_type and op.outputs[0].name not in visited_nodes:
                        queue_prefix.append(prefix)
                        queue_nodename.append(op.outputs[0].name)
                        visited_nodes.add(op.outputs[0].name)

    # the name of metadata node
    ret_output_arrays = ['one_compiler/bcqinfo_one_metadata']

    # given node from user
    ret_output_arrays.append(output_arrays)

    # all pairs of candidate operations and related BCQ information nodes
    for prefix in prefix_set:
        for fusable_op in prefix_node_dict[prefix]:
            ret_output_arrays.append(prefix + '/bcqinfo_do_w_x')
            ret_output_arrays.append(prefix + '/bcqinfo_alpha')
            ret_output_arrays.append(prefix + '/bcqinfo_packed_binary_code')
            ret_output_arrays.append(prefix + '/bcqinfo_number_of_clusters')
            ret_output_arrays.append(prefix + '/bcqinfo_size_of_clusters')
            ret_output_arrays.append(prefix + '/bcqinfo_qbits_of_clusters')
            ret_output_arrays.append(fusable_op)
            if has_dequant_weight:
                ret_output_arrays.append(prefix + '/bcqinfo_dequant_weight')
    for fuseop in fuseop_prefix_dict.keys():
        if len(fuseop_prefix_dict[fuseop]) == 1:
            prefix = fuseop_prefix_dict[fuseop].pop()
            ret_output_arrays.append(prefix + '/bcqinfo_do_w_x')
            ret_output_arrays.append(prefix + '/bcqinfo_alpha')
            ret_output_arrays.append(prefix + '/bcqinfo_packed_binary_code')
            ret_output_arrays.append(prefix + '/bcqinfo_number_of_clusters')
            ret_output_arrays.append(prefix + '/bcqinfo_size_of_clusters')
            ret_output_arrays.append(prefix + '/bcqinfo_qbits_of_clusters')
            ret_output_arrays.append(fuseop)
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
        return None
    else:
        err_msg = "BCQ version of the model(v{}) ".format(model_version)
        err_msg += "is higher than "
        err_msg += "the version supported by this program(v{})".format(program_version)
        raise SystemExit(err_msg)


def main():
    # Parse argument.
    parser = _get_parser()
    flags = parser.parse_known_args(args=sys.argv[1:])

    print_bcq_output_arrays(flags[0])


if __name__ == "__main__":
    main()
