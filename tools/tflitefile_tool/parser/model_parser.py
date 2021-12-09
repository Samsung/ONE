#!/usr/bin/env python

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

from .tflite_parser import TFLiteParser
from printer.subgraph_printer import SubgraphPrinter
from printer.graph_stats_printer import PrintGraphStats
from saver.model_saver import ModelSaver


# TODO: Rename it as ModelParser
class TFLiteModelFileParser(object):
    def __init__(self, option):
        self.option = option

    def PrintModel(self, model_name, op_parser):
        printer = SubgraphPrinter(self.option.print_level, op_parser, model_name)

        if self.option.print_all_tensor == False:
            printer.SetPrintSpecificTensors(self.option.print_tensor_index)

        if self.option.print_all_operator == False:
            printer.SetPrintSpecificOperators(self.option.print_operator_index)

        printer.PrintInfo()

    def SaveModel(self, model_name, op_parser):
        saver = ModelSaver(model_name, op_parser)

        if self.option.save_config == True:
            saver.SaveConfigInfo(self.option.save_prefix)

    def main(self):
        parser = TFLiteParser(self.option.model_file)
        parser.parse()

        for model_name, op_parser in parser.subg_list:
            if self.option.save == False:
                # print all of operators or requested objects
                self.PrintModel(model_name, op_parser)
            else:
                # save all of operators in this model
                self.SaveModel(model_name, op_parser)

        print('==== Model Stats ({} Subgraphs) ===='.format(len(parser.subg_list)))
        print('')
        PrintGraphStats(parser.stats, self.option.print_level)
