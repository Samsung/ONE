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

import tflite.Model
import tflite.SubGraph
from ir import graph_stats
from .operator_parser import OperatorParser
from printer.subgraph_printer import SubgraphPrinter
from printer.graph_stats_printer import PrintGraphStats
from saver.model_saver import ModelSaver


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
        # Generate Model: top structure of tflite model file
        buf = self.option.model_file.read()
        buf = bytearray(buf)
        tf_model = tflite.Model.Model.GetRootAsModel(buf, 0)

        stats = graph_stats.GraphStats()
        # Model file can have many models
        for subgraph_index in range(tf_model.SubgraphsLength()):
            tf_subgraph = tf_model.Subgraphs(subgraph_index)
            model_name = "#{0} {1}".format(subgraph_index, tf_subgraph.Name())
            # 0th subgraph is main subgraph
            if (subgraph_index == 0):
                model_name += " (MAIN)"

            # Parse Operators
            op_parser = OperatorParser(tf_model, tf_subgraph)
            op_parser.Parse()

            stats += graph_stats.CalcGraphStats(op_parser)

            if self.option.save == False:
                # print all of operators or requested objects
                self.PrintModel(model_name, op_parser)
            else:
                # save all of operators in this model
                self.SaveModel(model_name, op_parser)

        print('==== Model Stats ({} Subgraphs) ===='.format(tf_model.SubgraphsLength()))
        print('')
        PrintGraphStats(stats, self.option.print_level)
