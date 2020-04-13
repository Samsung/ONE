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

import tflite.Conv2DOptions
import tflite.Pool2DOptions
import tflite.BuiltinOptions
import tflite.Tensor
from tensor_wrapping import Tensor
import math


# NOTICE
# - an internal class. do not import outside this file.
# - REF: https://stackoverflow.com/questions/551038/private-implementation-class-in-python
class _OperationComputeMethod(object):
    '''
    NOTE: How to count operations of convolution(and also pooling)?

    If we know operations of output's one element, we can calculate total output's operations.
    For example, consider output Shape[3,3]
    [ e11 e12 e13 ]
    [ e21 e22 e23 ]
    [ e31 e32 e33 ]
    If we know operations for calculation of e11, we can know total operations of output(e11, e12, ... e33)
    by operations of e11 * 9(total number of elements)

    So we only need to know how to calculate operations of e11.
    For this, just think how to conv operation to the output's element
    If input_channel is 1, we can only think of kernel_size(kernel_w and kernel_h).
    For example, consider input Shape[3,3] and kernel Shape[2,2]
    [ i11 i12 i13 ]   [ k11 k12 ]   [ o11 o12 o13 ]
    [ i21 i22 i23 ] * [ k21 k22 ] = [ o21 o22 o23 ]
    [ i31 i32 i33 ]                 [ o31 o32 o33 ]

    Conv operation: for o11, i11 * k11 + i21 * k21 + i12 * k12 + i22 * k22 = o11
    On above conv operation, mul operations are done at 4 times(== kernel_w * kernel_h)
    and add operations are dont at 3 times(== kernel_w * kernel_h - 1)
    and also, bias will be done and it will be counted on add operations.

    Anyway, we can calculate total operations on this way. This can apply to the way of pooling.
    '''

    def ComputeOperationForConv2D(self, tf_operator, inputs, outputs):
        assert (
            tf_operator.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions()
            .Conv2DOptions)

        # NOTE: Assume that conv2d operator always take 3 tensors as inputs
        #       and both width and height are the same.
        # operator_inputs[]: [input_tensor, weight_tensor, bias_tensor]
        # operator_outputs[]: [output_tensor]
        # tflite's tensor shape: [N,H,W,C]
        input_tensor = inputs[0].tf_tensor
        weight_tensor = inputs[1].tf_tensor
        output_tensor = outputs[0].tf_tensor

        # kernel_ops = (kernel_w * kernel_h * input_channel * 2(multiply and add))
        kernel_ops = (
            weight_tensor.Shape(2) * weight_tensor.Shape(1) * input_tensor.Shape(3))

        # total ops
        #     = batch_size * output_channel * output_width * output_height * kernel_ops
        total_ops = (output_tensor.Shape(0) * output_tensor.Shape(3) *
                     output_tensor.Shape(2) * output_tensor.Shape(1))

        add_instr_num = (total_ops * (kernel_ops + 1))  # bias
        mul_instr_num = (total_ops * (kernel_ops))
        nonlinear_instr_num = 0
        return (add_instr_num, mul_instr_num, nonlinear_instr_num)

    # NOTE: Reference the comment 'NOTE' of ComputeOperationForConv2D

    def ComputeOperationForPooling(self, tf_operator, inputs, outputs):
        assert (
            tf_operator.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions()
            .Pool2DOptions)

        dummy_input_tensor = inputs[0].tf_tensor
        output_tensor = outputs[0].tf_tensor

        pool2d_options = tflite.Pool2DOptions.Pool2DOptions()
        pool2d_options.Init(tf_operator.BuiltinOptions().Bytes,
                            tf_operator.BuiltinOptions().Pos)

        # kernel_ops = kernel_w * kernel_h
        kernel_ops = (pool2d_options.FilterWidth() * pool2d_options.FilterHeight())

        # total ops
        #     = batch_size * output_channel * output_width * output_height *
        #       kernel_ops(kernel_w * kernel_h)
        total_ops = (output_tensor.Shape(0) * output_tensor.Shape(3) *
                     output_tensor.Shape(2) * output_tensor.Shape(1))

        add_instr_num = (total_ops * kernel_ops - 1)
        mul_instr_num = (total_ops * kernel_ops)
        nonlinear_instr_num = 0
        return (add_instr_num, mul_instr_num, nonlinear_instr_num)

    def ComputeOperationForSoftmax(self, tf_operator, inputs, outputs):
        assert (
            tf_operator.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions()
            .SoftmaxOptions)

        input_tensor = inputs[0].tf_tensor

        dummy_batch_size = input_tensor.Shape(0)
        input_dim = input_tensor.Shape(1)

        # Softmax(x_i) = exp(x_i) / sum of exp(x)
        add_instr_num = input_dim - 1  # sum of exp(x)
        mul_instr_num = input_dim  # /
        nonlinear_instr_num = input_dim + input_dim  # sum of exp(x) and exp(x_i)
        return (add_instr_num, mul_instr_num, nonlinear_instr_num)

    def ComputeOperationForFullyConnected(self, tf_operator, inputs, outputs):
        assert (
            tf_operator.BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions()
            .FullyConnectedOptions)

        # NOTE: Assume that fully_connected operator always take 3 tensors as inputs
        #       and its X tensor's shape is [1, 1, 1, input_dim] with
        #       its output Y [1, output_dim]
        input_tensor = inputs[0].tf_tensor
        output_tensor = outputs[0].tf_tensor

        # ops_per_element
        #     = input_dim(multiplication) + input_dim-1(addition) + 1(bias)
        # total_ops
        #     = ops_per_elem * output_dim

        add_instr_num = (input_tensor.Shape(3) * output_tensor.Shape(1))
        mul_instr_num = (input_tensor.Shape(3) * output_tensor.Shape(1))
        nonlinear_instr_num = 0
        return (add_instr_num, mul_instr_num, nonlinear_instr_num)

    def ComputeOperationForNothing(self, tf_operator, inputs, outputs):
        add_instr_num = 0
        mul_instr_num = 0
        nonlinear_instr_num = 0
        return (add_instr_num, mul_instr_num, nonlinear_instr_num)

    def NYI_ComputeOperation(self, tf_operator, inputs, outputs):
        pass

    operation_to_method_map = {
        # Inceptionv3
        "CONV_2D": ComputeOperationForConv2D,
        "AVERAGE_POOL_2D": ComputeOperationForPooling,
        "MAX_POOL_2D": ComputeOperationForPooling,
        "SOFTMAX": ComputeOperationForSoftmax,
        "FULLY_CONNECTED": ComputeOperationForFullyConnected,
        "CONCATENATION": ComputeOperationForNothing,
        # Extension
        "TOPK_V2": NYI_ComputeOperation,
        "SUB": NYI_ComputeOperation,
        "STRIDED_SLICE": NYI_ComputeOperation,
        "RESHAPE": NYI_ComputeOperation,
        "GATHER": NYI_ComputeOperation,
        "RESIZE_BILINEAR": NYI_ComputeOperation,
        "CAST": NYI_ComputeOperation,
        "ADD": NYI_ComputeOperation,
        "MUL": NYI_ComputeOperation,
        "DIV": NYI_ComputeOperation,
        "CUSTOM(TensorFlowMax)": NYI_ComputeOperation,
        "CUSTOM": NYI_ComputeOperation,
    }


class Operation(object):
    def __init__(self, tf_operator, operator_str, inputs, outputs):
        self.tf_operator = tf_operator
        self.operator_str = operator_str
        self.inputs = inputs
        self.outputs = outputs
        self.add_instr_num = 0
        self.mul_instr_num = 0
        self.nonlinear_instr_num = 0
        self.can_compute = True
        self.Compute()

    def Compute(self):
        comp_map = _OperationComputeMethod().operation_to_method_map
        if not self.operator_str in comp_map.keys():
            self.can_compute = False
            return

        method = comp_map[self.operator_str]
        if method.__name__ == _OperationComputeMethod().NYI_ComputeOperation.__name__:
            self.can_compute = False
            return

        self.add_instr_num, self.mul_instr_num, self.nonlinear_instr_num = method(
            _OperationComputeMethod(), self.tf_operator, self.inputs, self.outputs)

    def TotalInstrNum(self):
        return (self.add_instr_num + self.mul_instr_num + self.nonlinear_instr_num)
