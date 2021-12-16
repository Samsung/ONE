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

from ir import graph_stats
from .string_builder import StringBuilder


class SubgraphPrinter(object):
    def __init__(self, verbose, subg):
        self.verbose = verbose
        self.subg = subg
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
            self.PrintGraphStats()

        if self.print_all_tensor == False:
            print('')
            self.PrintSpecificTensors(self.print_tensor_index_list)
            print('')

        if self.print_all_operator == False:
            print('')
            self.PrintSpecificOperators(self.print_operator_index_list)
            print('')

    def PrintModelInfo(self):
        print("[" + self.subg.model_name + "]\n")
        if self.verbose > 0:
            model_inputs = []
            for t in self.subg.inputs:
                model_inputs.append(t.index)
            model_outputs = []
            for t in self.subg.outputs:
                model_outputs.append(t.index)
            print(self.subg.model_name + " input tensors: " + str(model_inputs))
            self.PrintSpecificTensors(model_inputs, "\t")
            print(self.subg.model_name + " output tensors: " + str(model_outputs))
            self.PrintSpecificTensors(model_outputs, "\t")
        print('')

    def PrintAllOperatorsInList(self):
        if (self.verbose < 1):
            return

        for index, operator in self.subg.operators_map.items():
            info = StringBuilder().Operator(operator)
            print(info)
            print('')

        print('')

    def PrintSpecificTensors(self, print_tensor_index_list, depth_str=""):
        if (self.verbose < 1):
            return

        for index in print_tensor_index_list:
            tensor = self.subg.tensors_map[index]
            info = StringBuilder().Tensor(tensor, depth_str)
            print(info)

    def PrintSpecificOperators(self, print_operator_index_list):
        for index in print_operator_index_list:
            operator = self.subg.operators_map[index]
            info = StringBuilder().Operator(operator)
            print(info)
            print('')

    def PrintGraphStats(self):
        stats = graph_stats.CalcGraphStats(self.subg)
        info = StringBuilder().GraphStats(stats)
        print(info)
