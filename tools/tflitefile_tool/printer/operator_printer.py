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

from ir.operator_wrapping import Operator
from .tensor_printer import TensorPrinter
from .option_printer import OptionPrinter


def GetStringTensorIndex(tensors):
    return_string = []
    return_string.append("[")
    for idx in range(len(tensors)):
        if idx != 0:
            return_string.append(", ")
        return_string.append(str(tensors[idx].tensor_idx))
    return_string.append("]")
    return "".join(return_string)


# TODO: Extract to a single Printer class like Printer.print(operator)
class OperatorPrinter(object):
    def __init__(self, verbose, operator):
        self.verbose = verbose
        self.operator = operator

    def PrintInfo(self):
        info = self.GetStringInfo()
        if info is not None:
            print(info)

    def GetStringInfo(self):
        if (self.verbose < 1):
            return None

        results = []
        results.append("Operator {}: {}".format(self.operator.operator_idx,
                                                self.operator.opcode_str))
        results.append("\tFused Activation: {}".format(self.operator.fused_activation))
        results.append("\tInput Tensors" + GetStringTensorIndex(self.operator.inputs))
        for tensor in self.operator.inputs:
            results.append(TensorPrinter(self.verbose, tensor).GetStringInfoWONL("\t\t"))
        results.append("\tOutput Tensors" + GetStringTensorIndex(self.operator.outputs))
        for tensor in self.operator.outputs:
            results.append(TensorPrinter(self.verbose, tensor).GetStringInfoWONL("\t\t"))
        # operator option
        # Some operations does not have option. In such case no option is printed
        option_string = OptionPrinter(self.verbose, self.operator.opcode_str,
                                      self.operator.options).GetStringInfoWONL("\t")
        if option_string is not None:
            results.append(option_string)
        return "\n".join(results)
