#!/usr/bin/env python3

# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

import onnx
import os
import sys

from onnx import AttributeProto, TensorProto
from onnx import numpy_helper
from onnx import helper


def _data_type_str(data_type):
    return TensorProto.DataType.Name(data_type)


def _get_attribute_value(attr):
    if attr.type == AttributeProto.TENSOR:
        return "{}, {}".format(
            _data_type_str(attr.t.data_type), numpy_helper.to_array(attr.t))
    if attr.type == AttributeProto.GRAPH:
        # TODO revise when graph node is available
        return "<graph>"
    if attr.type == AttributeProto.TENSORS:
        # TODO revise to see contents
        return "<tensors>..."
    if attr.type == AttributeProto.GRAPHS:
        # TODO revise when graph node is available
        return "<graphs>..."
    return helper.get_attribute_value(attr)


def _dump_header(onnx_model):
    print("[General] -----------------------------")
    print("IR version =", onnx_model.ir_version)
    print("Producer   =", onnx_model.producer_name, onnx_model.producer_version)
    print("")


def _dump_operators(onnx_model):
    opcodes_dict = dict()
    for node in onnx_model.graph.node:
        if node.op_type in opcodes_dict:
            opcodes_dict[node.op_type] = opcodes_dict[node.op_type] + 1
        else:
            opcodes_dict[node.op_type] = 1

    print("[Operators] ---------------------------")
    for opcode_key in opcodes_dict:
        print("{:>5} {}".format(opcodes_dict[opcode_key], opcode_key))

    print("")


def _dump_initializers(onnx_model):
    print("[Initializers] ------------------------")
    for initializer in onnx_model.graph.initializer:
        init_name = '"{}"'.format(initializer.name)
        dtstr = _data_type_str(initializer.data_type)
        print('{:<15} {} {}'.format(init_name, dtstr, initializer.dims))

    print("")


def _dump_nodes(onnx_model):
    print("[Nodes] -------------------------------")

    for node in onnx_model.graph.node:
        print('{0}("{1}")'.format(node.op_type, node.name))

        attribute = ''
        for attr in node.attribute:
            if attribute != '':
                attribute += ', '
            attribute += "{}: {}".format(attr.name, _get_attribute_value(attr))

        if attribute != '':
            print('    A {0}'.format(attribute))

        for inp in node.input:
            print('    I "{0}"'.format(inp))
        for out in node.output:
            print('    O "{0}"'.format(out))

    print("")


def _dump_inputoutputs(onnx_model):
    print("[Graph Input/Output]-------------------")
    for mod_input in onnx_model.graph.input:
        io_name = '"{}"'.format(mod_input.name)
        dtstr = _data_type_str(mod_input.type.tensor_type.elem_type)
        shape = mod_input.type.tensor_type.shape
        input_shape = [dim.dim_value for dim in shape.dim]
        print('    I: {:<15} {} {}'.format(io_name, dtstr, input_shape))

    for mod_output in onnx_model.graph.output:
        io_name = '"{}"'.format(mod_output.name)
        dtstr = _data_type_str(mod_output.type.tensor_type.elem_type)
        shape = mod_output.type.tensor_type.shape
        output_shape = [dim.dim_value for dim in shape.dim]
        print('    O: {:<15} {} {}'.format(io_name, dtstr, output_shape))

    print("")


def _dump_graph(onnx_model):
    _dump_header(onnx_model)
    _dump_operators(onnx_model)
    _dump_initializers(onnx_model)
    _dump_nodes(onnx_model)
    _dump_inputoutputs(onnx_model)


def _help_exit(cmd_name):
    print('Dump ONNX model file Graph')
    print('Usage: {0} [onnx_path]'.format(cmd_name))
    print('')
    exit()


def main():
    if len(sys.argv) < 2:
        _help_exit(os.path.basename(sys.argv[0]))

    onnx_model = onnx.load(sys.argv[1])
    onnx.checker.check_model(onnx_model)

    _dump_graph(onnx_model)


if __name__ == "__main__":
    main()
