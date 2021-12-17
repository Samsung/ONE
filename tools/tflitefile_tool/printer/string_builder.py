#!/usr/bin/python

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

import numpy as np

SYMBOLS = ['B', 'K', 'M', 'G', 'T']


def ConvertBytesToHuman(n):
    n = int(n)
    if n < 0:
        return 0

    format_str = "%(val)3.1f%(symb)s"
    prefix = {}
    for i, s in enumerate(SYMBOLS[1:]):
        prefix[s] = 1 << (i + 1) * 10

    for symbol in reversed(SYMBOLS[1:]):
        if n >= prefix[symbol]:
            v = float(n) / prefix[symbol]
            return format_str % dict(symb=symbol, val=v)

    return format_str % dict(symb=SYMBOLS[0], val=n)


TENSOR_SYMBOL = "%"
OPERATOR_SYMBOL = "#"
BUFFER_SYMBOL = "&"


def GetStringTensorIndex(tensors):
    return_string = []
    return_string.append("[")
    for idx in range(len(tensors)):
        if idx != 0:
            return_string.append(", ")
        return_string.append(TENSOR_SYMBOL + str(tensors[idx].index))
    return_string.append("]")
    return "".join(return_string)


def GetStringShape(tensor):
    shape_len = len(tensor.shape)
    if shape_len == 0:
        return "Scalar"
    return_string = []
    return_string.append("[")
    for shape_idx in range(shape_len):
        if (shape_idx != 0):
            return_string.append(", ")
        return_string.append(str(tensor.shape[shape_idx]))
    return_string.append("]")
    return "".join(return_string)


def GetStringTensor(tensor):
    info = ""
    if tensor.index < 0:
        info = "{:5}".format(TENSOR_SYMBOL + str(tensor.index))
    else:
        shape_str = GetStringShape(tensor)
        type_name = tensor.type_name
        shape_name = tensor.tensor_name
        memory_size = ConvertBytesToHuman(tensor.memory_size)

        buffer_str = "("
        if tensor.buffer is not None:
            buffer_str += "{:5}: ".format(BUFFER_SYMBOL + str(tensor.buffer_index))
            # if too big, just skip it.
            if tensor.buffer.size > 4:
                buffer_str += "".join(['[' for _ in range(tensor.buffer.ndim)])
                buffer_str += " ... "
                buffer_str += "".join([']' for _ in range(tensor.buffer.ndim)])
            else:
                buffer_str += np.array2string(
                    tensor.buffer, precision=3, separator=', ', threshold=4, edgeitems=2)
        else:
            buffer_str += "Empty"
        buffer_str += ")"

        info = "{:5} : buffer {:25} | {:7} | Memory {:6} | Shape {} ({})".format(
            TENSOR_SYMBOL + str(tensor.index), buffer_str, type_name, memory_size,
            shape_str, shape_name)
    return info


class StringBuilder(object):
    def __init__(self, spacious_str="  "):
        self.spacious_str = spacious_str

    def GraphStats(self, stats):
        results = []

        results.append("{:38}: {:4}".format("Number of all operator types",
                                            len(stats.op_counts)))

        # op type stats
        for op_name in sorted(stats.op_counts.keys()):
            occur = stats.op_counts[op_name]
            optype_info_str = "{:38}: {:4}".format(self.spacious_str + op_name, occur)
            results.append(optype_info_str)

        summary_str = "{:38}: {:4}".format("Number of all operators",
                                           sum(stats.op_counts.values()))
        results.append(summary_str)
        results.append('')

        # memory stats
        results.append("Expected TOTAL  memory: {}".format(
            ConvertBytesToHuman(stats.total_memory)))
        results.append("Expected FILLED memory: {}".format(
            ConvertBytesToHuman(stats.filled_memory)))

        return "\n".join(results)

    def Operator(self, operator):
        results = []
        results.append("{:5}: {}".format(OPERATOR_SYMBOL + str(operator.index),
                                         operator.op_name))
        results.append("{}Fused Activation: {}".format(self.spacious_str,
                                                       operator.activation))
        results.append("{}Input Tensors{}".format(self.spacious_str,
                                                  GetStringTensorIndex(operator.inputs)))
        for tensor in operator.inputs:
            results.append(self.Tensor(tensor, self.spacious_str + self.spacious_str))
        results.append("{}Output Tensors{}".format(self.spacious_str,
                                                   GetStringTensorIndex(
                                                       operator.outputs)))
        for tensor in operator.outputs:
            results.append(self.Tensor(tensor, self.spacious_str + self.spacious_str))
        # operator option
        # Some operations does not have option. In such case no option is printed
        if operator.options != None and operator.options != "":
            results.append(self.Option(operator.options, self.spacious_str))
        return "\n".join(results)

    def Tensor(self, tensor, depth_str=""):
        results = []
        results.append("{}{}".format(depth_str, GetStringTensor(tensor)))
        return "".join(results)

    def Option(self, options_str, depth_str=""):
        results = []
        results.append("{}Options".format(depth_str))
        results.append("{}{}{}".format(depth_str, self.spacious_str, options_str))
        return "\n".join(results)
