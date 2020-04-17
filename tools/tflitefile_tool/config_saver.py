#!/usr/bin/python

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

from operator_wrapping import Operator
from tensor_printer import TensorPrinter
from option_printer import OptionPrinter


class ConfigSaver(object):
    def __init__(self, file_name, operator):
        self.file_name = file_name
        self.operator = operator
        # Set self.verbose to 1 level to print more information
        self.verbose = 1
        self.op_idx = operator.operator_idx
        self.op_name = operator.opcode_str

        self.f = open(file_name, 'at')

    def __del__(self):
        self.f.close()

    def SaveInfo(self):
        self.f.write("[{}]\n".format(self.op_idx))
        if (self.op_name == 'CONV_2D'):
            self.SaveConv2DInputs()
        else:
            self.SaveInputs()

        self.SaveOutputs()

        self.SaveAttributes()

        self.f.write('\n')

    def SaveConv2DInputs(self):
        if (len(self.operator.inputs) != 3):
            raise AssertionError('Conv2D input count should be 3')

        inputs = self.operator.inputs[0]
        weights = self.operator.inputs[1]
        bias = self.operator.inputs[2]

        self.f.write("input: {}\n".format(
            TensorPrinter(self.verbose, inputs).GetShapeString()))
        self.f.write("input_type: {}\n".format(inputs.type_name))
        self.f.write("weights: {}\n".format(
            TensorPrinter(self.verbose, weights).GetShapeString()))
        self.f.write("weights_type: {}\n".format(weights.type_name))
        self.f.write("bias: {}\n".format(
            TensorPrinter(self.verbose, bias).GetShapeString()))
        self.f.write("bias_type: {}\n".format(bias.type_name))

    def SaveInputs(self):
        total = len(self.operator.inputs)
        self.f.write("input_counts: {}\n".format(total))
        for idx in range(total):
            tensor = self.operator.inputs[idx]
            input_shape_str = TensorPrinter(self.verbose, tensor).GetShapeString()
            self.f.write("input{}: {}\n".format(idx, input_shape_str))
            self.f.write("input{}_type: {}\n".format(idx, tensor.type_name))

    def SaveOutputs(self):
        total = len(self.operator.outputs)
        self.f.write("output_counts: {}\n".format(total))
        for idx in range(total):
            tensor = self.operator.outputs[idx]
            output_shape_str = TensorPrinter(self.verbose, tensor).GetShapeString()
            self.f.write("output{}: {}\n".format(idx, output_shape_str))
            self.f.write("output{}_type: {}\n".format(idx, tensor.type_name))

    def SaveFilter(self):
        self.f.write("filter_w: {}\n".format(self.operator.options.FilterWidth()))
        self.f.write("filter_h: {}\n".format(self.operator.options.FilterHeight()))

    def SaveStride(self):
        self.f.write("stride_w: {}\n".format(self.operator.options.StrideW()))
        self.f.write("stride_h: {}\n".format(self.operator.options.StrideH()))

    def SaveDilation(self):
        self.f.write("dilation_w: {}\n".format(self.operator.options.DilationWFactor()))
        self.f.write("dilation_h: {}\n".format(self.operator.options.DilationHFactor()))

    def SavePadding(self):
        if self.operator.options.Padding() == 0:
            self.f.write("padding: SAME\n")
        elif self.operator.options.Padding() == 1:
            self.f.write("padding: VALID\n")

    def SaveFusedAct(self):
        if self.operator.fused_activation is not "NONE":
            self.f.write("fused_act: {}\n".format(self.operator.fused_activation))

    def SaveAttributes(self):
        # operator option
        # Some operations does not have option. In such case no option is printed
        option_str = OptionPrinter(self.verbose, self.op_name,
                                   self.operator.options).GetOptionString()
        if self.op_name == 'AVERAGE_POOL_2D' or self.op_name == 'MAX_POOL_2D':
            self.SaveFilter()
            self.SaveStride()
            self.SavePadding()
        elif self.op_name == 'CONV_2D':
            self.SaveStride()
            self.SaveDilation()
            self.SavePadding()
        elif self.op_name == 'TRANSPOSE_CONV':
            self.SaveStride()
            self.SavePadding()
        elif self.op_name == 'DEPTHWISE_CONV_2D':
            self.SaveStride()
            self.SaveDilation()
            self.SavePadding()
            self.f.write("depthmultiplier: {}\n".format(
                self.operator.options.DepthMultiplier()))

        self.SaveFusedAct()
