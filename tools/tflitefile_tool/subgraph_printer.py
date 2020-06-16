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
import graph_stats


class SubgraphPrinter(object):
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
            graph_stats.PrintGraphStats(
                graph_stats.CalcGraphStats(self.op_parser), self.verbose)

        if self.print_all_tensor == False:
            print('')
            self.PrintSpecificTensors(self.print_tensor_index_list)
            print('')

        if self.print_all_operator == False:
            print('')
            self.PrintSpecificOperators(self.print_operator_index_list)
            print('')

    def PrintModelInfo(self):
        print("[" + self.model_name + "]\n")
        if self.verbose > 0:
            model_inputs = self.op_parser.tf_subgraph.InputsAsNumpy()
            model_outputs = self.op_parser.tf_subgraph.OutputsAsNumpy()
            print(self.model_name + " input tensors: " + str(model_inputs))
            self.PrintSpecificTensors(model_inputs, "\t")
            print(self.model_name + " output tensors: " + str(model_outputs))
            self.PrintSpecificTensors(model_outputs, "\t")
        print('')

    def PrintAllOperatorsInList(self):
        if (self.verbose < 1):
            return

        for operator in self.op_parser.operators_in_list:
            printer = OperatorPrinter(self.verbose, operator)
            printer.PrintInfo()
            print('')

        print('')

    def PrintSpecificTensors(self, print_tensor_index_list, depth_str=""):
        for tensor in self.op_parser.GetTensors(print_tensor_index_list):
            printer = TensorPrinter(self.verbose, tensor)
            printer.PrintInfo(depth_str)

    def PrintSpecificOperators(self, print_operator_index_list):
        for operator in self.op_parser.operators_in_list:
            if operator.operator_idx in print_operator_index_list:
                printer = OperatorPrinter(self.verbose, operator)
                printer.PrintInfo()
