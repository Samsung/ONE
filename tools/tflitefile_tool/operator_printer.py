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

from operator_wrapping import Operator
from tensor_printer import TensorPrinter
from option_printer import OptionPrinter
from perf_predictor import PerfPredictor


def GetStrTensorIndex(tensors):
    return_string = "["
    for idx in range(len(tensors)):
        if idx != 0:
            return_string += ", "
        return_string += str(tensors[idx].tensor_idx)
    return_string += "]"
    return return_string


class OperatorPrinter(object):
    def __init__(self, verbose, operator):
        self.verbose = verbose
        self.operator = operator

    def PrintInfo(self, perf_predictor=None):
        if (self.verbose < 1):
            return

        op_str = "Operator {0}: {1}".format(self.operator.operator_idx,
                                            self.operator.opcode_str)

        if self.verbose == 2:
            # total instruction num
            instrs = "{:,}".format(self.operator.operation.TotalInstrNum()
                                   ) if self.operator.operation.can_compute else "???"

            # total operation cycles
            cycles = "{:,}".format(
                (perf_predictor.PredictCycles(self.operator.operation))
            ) if self.operator.operation.can_compute and perf_predictor != None else "???"

            op_str = op_str + "(instrs: {0}, cycls: {1})".format(instrs, cycles)

        print(op_str)
        print("\tFused Activation: " + self.operator.fused_activation)
        self.PrintTensors()

    def PrintTensors(self):
        print("\tInput Tensors" + GetStrTensorIndex(self.operator.inputs))
        for tensor in self.operator.inputs:
            TensorPrinter(self.verbose, tensor).PrintInfo("\t\t")
        print("\tOutput Tensors" + GetStrTensorIndex(self.operator.outputs))
        for tensor in self.operator.outputs:
            TensorPrinter(self.verbose, tensor).PrintInfo("\t\t")

        # operator option
        # Some operations does not have option. In such case no option is printed
        OptionPrinter(self.verbose, self.operator.opcode_str,
                      self.operator.options).PrintInfo("\t")
