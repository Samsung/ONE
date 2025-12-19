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
    def __init__(self, verbose, subg, spacious_str="  "):
        self.verbose = verbose
        self.subg = subg
        self.spacious_str = spacious_str
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
            print("[" + self.subg.model_name + "]")
            print('')
            if self.verbose > 0:
                self.PrintModelInfo()
                print('')
                self.PrintOperators()
            if self.verbose == 2:
                self.PrintBuffers()
            self.PrintGraphStats()

        if self.verbose == 0:
            return

        if self.print_all_tensor == False:
            print('')
            self.PrintSpecificTensors(self.print_tensor_index_list)
            print('')

        if self.print_all_operator == False:
            print('')
            self.PrintSpecificOperators(self.print_operator_index_list)
            print('')

    def PrintModelInfo(self):
        model_inputs = []
        for t in self.subg.inputs:
            model_inputs.append(t.index)
        model_outputs = []
        for t in self.subg.outputs:
            model_outputs.append(t.index)
        print(self.subg.model_name + " input tensors: " + str(model_inputs))
        self.PrintSpecificTensors(model_inputs, "    ")
        print(self.subg.model_name + " output tensors: " + str(model_outputs))
        self.PrintSpecificTensors(model_outputs, "    ")

    def PrintOperators(self):
        for index, operator in self.subg.operators_map.items():
            info = StringBuilder(self.spacious_str).Operator(operator)
            print(info)
            print()

    def PrintSpecificTensors(self, print_tensor_index_list, depth_str=""):
        for index in print_tensor_index_list:
            tensor = self.subg.tensors_map[index]
            info = StringBuilder(self.spacious_str).Tensor(tensor, depth_str)
            print(info)

    def PrintSpecificOperators(self, print_operator_index_list):
        for index in print_operator_index_list:
            operator = self.subg.operators_map[index]
            info = StringBuilder(self.spacious_str).Operator(operator)
            print(info)

    def PrintGraphStats(self):
        stats = graph_stats.CalcGraphStats(self.subg)
        info = StringBuilder(self.spacious_str).GraphStats(stats)
        print(info)

    def PrintBuffers(self):
        for index, tensor in self.subg.tensors_map.items():
            if tensor.buffer is not None:
                info = StringBuilder(self.spacious_str).Buffer(tensor)
                print(info)
                print()
