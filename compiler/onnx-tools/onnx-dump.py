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
        return "{}, {}".format(_data_type_str(attr.t.data_type),
                               numpy_helper.to_array(attr.t))
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


def _gather_value_infos(onnx_model):
    vis = dict()

    for mod_input in onnx_model.graph.input:
        vis[mod_input.name] = mod_input.type

    for mod_output in onnx_model.graph.output:
        vis[mod_output.name] = mod_output.type

    for vi in onnx_model.graph.value_info:
        vis[vi.name] = vi.type

    return vis


def _type_format(type):
    dtstr = _data_type_str(type.tensor_type.elem_type)
    shape = type.tensor_type.shape
    shape_ar = [dim.dim_value for dim in shape.dim]
    return '{} {}'.format(dtstr, shape_ar)


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
    total_nodes = 0
    for opcode_key in opcodes_dict:
        print("{:>5} {}".format(opcodes_dict[opcode_key], opcode_key))
        total_nodes = total_nodes + opcodes_dict[opcode_key]

    print("----- -----")
    print("{:>5} {}".format(total_nodes, 'Total'))

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

    vis = _gather_value_infos(onnx_model)

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
            inp_vi_str = ''
            if inp in vis:
                inp_vi = vis[inp]
                inp_vi_str = _type_format(inp_vi)
            print('    I "{0}" {1}'.format(inp, inp_vi_str))
        for out in node.output:
            out_vi_str = ''
            if out in vis:
                out_vi = vis[out]
                out_vi_str = _type_format(out_vi)
            print('    O "{0}" {1}'.format(out, out_vi_str))

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

    onnx.checker.check_model(sys.argv[1])
    onnx_model = onnx.load(sys.argv[1])

    _dump_graph(onnx_model)


if __name__ == "__main__":
    main()
