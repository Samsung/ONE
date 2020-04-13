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

from operator_printer import OperatorPrinter
from tensor_printer import TensorPrinter


class ModelPrinter(object):
    def __init__(self, verbose, op_parser, model_name):
        self.verbose = verbose
        self.op_parser = op_parser
        self.model_name = model_name
        self.print_all_tensor = True
        self.print_tensor_index_list = None
        self.print_all_operator = True
        self.print_operator_index_list = None

    def SetPrintSpecificTensors(self, tensor_indices):
        if len(tensor_indices) != 0:
            self.print_all_tensor = False
            self.print_tensor_index_list = tensor_indices

    def SetPrintSpecificOperators(self, operator_indices):
        if len(operator_indices) != 0:
            self.print_all_operator = False
            self.print_operator_index_list = operator_indices

    def PrintInfo(self):
        if self.print_all_tensor == True and self.print_all_operator == True:
            self.PrintModelInfo()
            self.PrintAllOperatorsInList()
            self.PrintAllTypesInfo()
            self.PrintTotalMemory()

        if self.print_all_tensor == False:
            print('')
            self.PrintSpecificTensors()

        if self.print_all_operator == False:
            print('')
            self.PrintSpecificOperators()

    def PrintModelInfo(self):
        print("[" + self.model_name + "]\n")
        if self.verbose > 0:
            model_inputs = self.op_parser.tf_subgraph.InputsAsNumpy()
            model_outputs = self.op_parser.tf_subgraph.OutputsAsNumpy()
            print(self.model_name + " input tensors: " + str(model_inputs))
            print(self.model_name + " output tensors: " + str(model_outputs))
        print('')

    def PrintAllOperatorsInList(self):
        if (self.verbose < 1):
            return

        for operator in self.op_parser.operators_in_list:
            printer = OperatorPrinter(self.verbose, operator)
            printer.PrintInfo(self.op_parser.perf_predictor)
            print('')

        print('')

    def PrintAllTypesInfo(self):
        print("Number of all operator types: {0}".format(
            len(self.op_parser.operators_per_type)))

        # number of instructions of all operator types to print if verbose level is 2
        total_instrs = 0

        # (a string of the operator type, a list of operators which are the same operator type)
        for type_str, oper_list in self.op_parser.operators_per_type.items():
            # number of occurrence of this operator type
            occur = len(oper_list)

            optype_info_str = "\t{type_str:38}: {occur:4}".format(
                type_str=type_str, occur=occur)

            if self.verbose == 2:
                # this operator type can be computed?
                can_compute = oper_list[0].operation.can_compute

                # total number of instructions of the same operator types
                if can_compute:
                    instrs = sum(
                        operator.operation.TotalInstrNum() for operator in oper_list)
                    total_instrs = total_instrs + instrs
                    instrs = "{:,}".format(instrs)
                else:
                    instrs = "???"

                optype_info_str = optype_info_str + " \t (instrs: {instrs})".format(
                    instrs=instrs)

            print(optype_info_str)

        summary_str = "{0:46}: {1:4}".format("Number of all operators",
                                             len(self.op_parser.operators_in_list))
        if self.verbose == 2:
            total_instrs = "{:,}".format(total_instrs)
            summary_str = summary_str + " \t (total instrs: {0})".format(total_instrs)

        print(summary_str)
        print('')

    def PrintSpecificTensors(self):
        for tensor in self.op_parser.GetAllTensors():
            if tensor.tensor_idx in self.print_tensor_index_list:
                printer = TensorPrinter(self.verbose, tensor)
                printer.PrintInfo()
                print('')
        print('')

    def PrintSpecificOperators(self):
        for operator in self.op_parser.operators_in_list:
            if operator.operator_idx in self.print_operator_index_list:
                printer = OperatorPrinter(self.verbose, operator)
                printer.PrintInfo(self.op_parser.perf_predictor)
                print('')

        print('')

    def PrintTotalMemory(self):
        total_memory = 0
        filled_memory = 0  # only memory for constant
        for tensor in self.op_parser.GetAllTensors():
            if tensor.tf_buffer.DataLength() != 0:
                filled_memory += tensor.memory_size
            total_memory += tensor.memory_size

        from tensor_printer import ConvertBytesToHuman
        print("Expected TOTAL  memory: {0}".format(ConvertBytesToHuman(total_memory)))
        print("Expected FILLED memory: {0}".format(ConvertBytesToHuman(filled_memory)))
        print('')
