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


def GetStringTensorIndex(tensors):
    return_string = []
    return_string.append("[")
    for idx in range(len(tensors)):
        if idx != 0:
            return_string.append(", ")
        return_string.append(str(tensors[idx].tensor_idx))
    return_string.append("]")
    return "".join(return_string)


def GetStringShape(tensor):
    if tensor.tf_tensor.ShapeLength() == 0:
        return "Scalar"
    return_string = []
    return_string.append("[")
    for shape_idx in range(tensor.tf_tensor.ShapeLength()):
        if (shape_idx != 0):
            return_string.append(", ")
        # when shape signature is -1, that means unknown dim
        if tensor.tf_tensor.ShapeSignature(shape_idx) != -1:
            return_string.append(str(tensor.tf_tensor.Shape(shape_idx)))
        else:
            return_string.append("-1")
    return_string.append("]")
    return "".join(return_string)


def GetStringTensor(tensor):
    info = ""
    if tensor.tensor_idx < 0:
        info = "Tensor {0:4}".format(tensor.tensor_idx)
    else:
        buffer_idx = tensor.tf_tensor.Buffer()
        buffer_str = "Empty" if buffer_idx == 0 else str(buffer_idx)
        isEmpty = "Filled"
        if (tensor.tf_buffer.DataLength() == 0):
            isEmpty = " Empty"
        shape_str = GetStringShape(tensor)
        type_name = tensor.type_name

        shape_name = ""
        if tensor.tf_tensor.Name() != 0:
            shape_name = tensor.tf_tensor.Name()

        memory_size = ConvertBytesToHuman(tensor.memory_size)

        info = "Tensor {:4} : buffer {:5} | {} | {:7} | Memory {:6} | Shape {} ({})".format(
            tensor.tensor_idx, buffer_str, isEmpty, type_name, memory_size, shape_str,
            shape_name)
    return info


def GetStringPadding(options):
    if options.Padding() == 0:
        return "SAME"
    elif options.Padding() == 1:
        return "VALID"
    else:
        return "** wrong padding value **"


def GetStringOption(op_name, options):
    if (op_name == "AVERAGE_POOL_2D" or op_name == "MAX_POOL_2D"):
        return "{}, {}, {}".format(
            "Filter W:H = {}:{}".format(options.FilterWidth(), options.FilterHeight()),
            "Stride W:H = {}:{}".format(options.StrideW(),
                                        options.StrideH()), "Padding = {}".format(
                                            GetStringPadding(options)))
    elif (op_name == "CONV_2D"):
        return "{}, {}, {}".format(
            "Stride W:H = {}:{}".format(options.StrideW(), options.StrideH()),
            "Dilation W:H = {}:{}".format(options.DilationWFactor(),
                                          options.DilationHFactor()),
            "Padding = {}".format(GetStringPadding(options)))
    elif (op_name == "DEPTHWISE_CONV_2D"):
        # yapf: disable
        return "{}, {}, {}, {}".format(
            "Stride W:H = {}:{}".format(options.StrideW(),
                                                options.StrideH()),
            "Dilation W:H = {}:{}".format(options.DilationWFactor(),
                                            options.DilationHFactor()),
            "Padding = {}".format(GetStringPadding(options)),
            "DepthMultiplier = {}".format(options.DepthMultiplier()))
        # yapf: enable
    elif (op_name == "STRIDED_SLICE"):
        # yapf: disable
        return "{}, {}, {}, {}, {}".format(
            "begin_mask({})".format(options.BeginMask()),
            "end_mask({})".format(options.EndMask()),
            "ellipsis_mask({})".format(options.EllipsisMask()),
            "new_axis_mask({})".format(options.NewAxisMask()),
            "shrink_axis_mask({})".format(options.ShrinkAxisMask()))
        # yapf: enable
    else:
        return None


class StringBuilder(object):
    def __init__(self, verbose):
        self.verbose = verbose

    def GraphStats(self, stats):
        results = []

        results.append("Number of all operator types: {}".format(len(stats.op_counts)))

        # op type stats
        for op_name in sorted(stats.op_counts.keys()):
            occur = stats.op_counts[op_name]
            optype_info_str = "\t{:38}: {:4}".format(op_name, occur)
            results.append(optype_info_str)

        summary_str = "{0:46}: {1:4}".format("Number of all operators",
                                             sum(stats.op_counts.values()))
        results.append(summary_str)
        results.append('\n')

        # memory stats
        results.append("Expected TOTAL  memory: {}".format(
            ConvertBytesToHuman(stats.total_memory)))
        results.append("Expected FILLED memory: {}".format(
            ConvertBytesToHuman(stats.filled_memory)))
        results.append('\n')

        return "\n".join(results)

    def Operator(self, operator):
        if (self.verbose < 1):
            return None

        results = []
        results.append("Operator {}: {}".format(operator.operator_idx,
                                                operator.opcode_str))
        results.append("\tFused Activation: {}".format(operator.fused_activation))
        results.append("\tInput Tensors" + GetStringTensorIndex(operator.inputs))
        for tensor in operator.inputs:
            results.append(self.Tensor(tensor, "\t\t"))
        results.append("\tOutput Tensors" + GetStringTensorIndex(operator.outputs))
        for tensor in operator.outputs:
            results.append(self.Tensor(tensor, "\t\t"))
        # operator option
        # Some operations does not have option. In such case no option is printed
        option_string = self.Option(operator.opcode_str, operator.options, "\t")
        if option_string is not None:
            results.append(option_string)
        return "\n".join(results)

    def Tensor(self, tensor, depth_str=""):
        if (self.verbose < 1):
            return None

        results = []
        if depth_str != "":
            results.append(depth_str)
        results.append(GetStringTensor(tensor))
        return "".join(results)

    def Option(self, op_name, options, tab=""):
        if self.verbose < 1 or options == 0:
            return None

        option_str = GetStringOption(op_name, options)
        if option_str is None:
            return None

        results = [option_str]
        results.append("{}Options".format(tab))
        results.append("{}\t{}".format(tab, option_str))
        return "\n".join(results)
