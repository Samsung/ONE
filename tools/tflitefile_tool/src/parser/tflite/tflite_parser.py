#!/usr/bin/env python

# Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

import tflite.Model
from .tflite_subgraph import TFLiteSubgraph
from .tflite_operator import TFLiteOperator, EnumStrMaps
from .tflite_tensor import TFLiteTensor, SetTensorTypeStr


def HasOptionalTensor(tf_subgraph):
    for operator_idx in range(tf_subgraph.OperatorsLength()):
        tf_operator = tf_subgraph.Operators(operator_idx)
        if -1 in tf_operator.InputsAsNumpy():
            return True
        output_tensors = tf_operator.OutputsAsNumpy()
        if -1 in tf_operator.OutputsAsNumpy():
            return True

    return False


class TFLiteSubgraphParser(object):
    def __init__(self, tf_model, subgraph_index):
        self.tf_model = tf_model
        self.tf_subgraph = tf_model.Subgraphs(subgraph_index)
        self.subg = TFLiteSubgraph(subgraph_index, self.tf_subgraph)

        # Tensor type string table
        SetTensorTypeStr()

    def Parse(self):
        if HasOptionalTensor(self.tf_subgraph):
            # Prepare that optional input and output tensors are indicated by -1
            self.subg.tensors_map[-1] = TFLiteTensor(-1, None, None)

        # tensors
        for tensor_idx in range(self.tf_subgraph.TensorsLength()):
            tf_tensor = self.tf_subgraph.Tensors(tensor_idx)
            buffer_idx = tf_tensor.Buffer()
            tf_buffer = self.tf_model.Buffers(buffer_idx)
            t = TFLiteTensor(tensor_idx, tf_tensor, tf_buffer)
            self.subg.tensors_map[tensor_idx] = t

        # operators
        for operator_idx in range(self.tf_subgraph.OperatorsLength()):
            tf_operator = self.tf_subgraph.Operators(operator_idx)
            op_name = self.GetOpcodeStr(tf_operator)
            input_tensors = self.GetTensors(tf_operator.InputsAsNumpy())
            output_tensors = self.GetTensors(tf_operator.OutputsAsNumpy())

            op = TFLiteOperator(operator_idx, tf_operator, input_tensors, output_tensors,
                                op_name)
            self.subg.operators_map[op.index] = op
            self.subg.optypes_map[op.op_name] = op

        self.subg.inputs = self.GetTensors(self.tf_subgraph.InputsAsNumpy())
        self.subg.outputs = self.GetTensors(self.tf_subgraph.OutputsAsNumpy())

        return self.subg

    def GetOpcodeStr(self, tf_operator):
        opcode_list_idx = tf_operator.OpcodeIndex()
        opcode_id = self.tf_model.OperatorCodes(opcode_list_idx).BuiltinCode()
        opcode_str = EnumStrMaps.BuiltinOpcode[opcode_id]
        if opcode_id == 32:
            # Custom operator
            custom_operator = self.tf_model.OperatorCodes(tf_operator.OpcodeIndex())
            custom_op_name = custom_operator.CustomCode().decode('utf-8')
            opcode_str = opcode_str + "(" + custom_op_name + ")"
        return opcode_str

    def GetTensors(self, tf_tensors_index):
        assert len(self.subg.tensors_map.keys()) > 0

        return_list = []
        for tensor_idx in tf_tensors_index:
            return_list.append(self.subg.tensors_map[tensor_idx])
        return return_list


class TFLiteParser(object):
    def __init__(self, model_file):
        self.model_file = model_file

    def Parse(self):
        # Generate Model: top structure of tflite model file
        buf = self.model_file.read()
        buf = bytearray(buf)
        tf_model = tflite.Model.Model.GetRootAsModel(buf, 0)

        # Model file can have many models
        subg_list = []
        for subgraph_index in range(tf_model.SubgraphsLength()):
            # Parse Subgraphs
            subg_parser = TFLiteSubgraphParser(tf_model, subgraph_index)
            subg = subg_parser.Parse()
            subg_list.append(subg)

        return subg_list
