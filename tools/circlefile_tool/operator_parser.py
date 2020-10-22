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

import tflite.Model
import tflite.SubGraph
import tflite.Operator
import tflite.OperatorCode
import tflite.BuiltinOperator
from operator_wrapping import Operator, EnumStrMaps
from tensor_wrapping import Tensor, SetTensorTypeStr
from operation import Operation


class OperatorParser(object):
    def __init__(self, tf_model, tf_subgraph):
        self.tf_model = tf_model
        self.tf_subgraph = tf_subgraph
        self.operators_in_list = list()
        self.operators_per_type = dict()
        # Tensor type string table
        SetTensorTypeStr()

    def Parse(self):
        for operator_idx in range(self.tf_subgraph.OperatorsLength()):
            tf_operator = self.tf_subgraph.Operators(operator_idx)
            opcode_str = self.GetOpcodeStr(tf_operator)
            input_tensors = self.GetInputTensors(tf_operator)
            output_tensors = self.GetOutputTensors(tf_operator)

            op = Operator(operator_idx, tf_operator, input_tensors, output_tensors,
                          opcode_str)
            self.AppendOperator(op)

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

    def GetInputTensors(self, tf_operator):
        operator_inputs = tf_operator.InputsAsNumpy()
        return self.GetTensors(operator_inputs)

    def GetOutputTensors(self, tf_operator):
        operator_outputs = tf_operator.OutputsAsNumpy()
        return self.GetTensors(operator_outputs)

    def GetTensors(self, tf_tensors_index):
        return_list = list()
        for tensor_idx in tf_tensors_index:
            # in case of optional input, tensor_idx == -1
            if (tensor_idx < 0):
                return_list.append(Tensor(tensor_idx, None, None))
                continue
            tf_tensor = self.tf_subgraph.Tensors(tensor_idx)
            buffer_idx = tf_tensor.Buffer()
            tf_buffer = self.tf_model.Buffers(buffer_idx)
            return_list.append(Tensor(tensor_idx, tf_tensor, tf_buffer))
        return return_list

    def GetAllTensors(self):
        return_list = list()
        for tensor_idx in range(self.tf_subgraph.TensorsLength()):
            if (tensor_idx < 0):
                return_list.append(Tensor(tensor_idx, 0, 0))
                continue
            tf_tensor = self.tf_subgraph.Tensors(tensor_idx)
            buffer_idx = tf_tensor.Buffer()
            tf_buffer = self.tf_model.Buffers(buffer_idx)
            return_list.append(Tensor(tensor_idx, tf_tensor, tf_buffer))
        return return_list

    def AppendOperator(self, operator):
        self.operators_in_list.append(operator)

        opcode_str = operator.opcode_str
        if opcode_str not in self.operators_per_type:
            self.operators_per_type[opcode_str] = list()
        self.operators_per_type[opcode_str].append(operator)
