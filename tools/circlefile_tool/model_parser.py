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

import os
import sys
import numpy
import flatbuffers
import tflite.Model
import tflite.SubGraph
import argparse
import graph_stats
from operator_parser import OperatorParser
from subgraph_printer import SubgraphPrinter
from model_saver import ModelSaver


class TFLiteModelFileParser(object):
    def __init__(self, args):
        # Read flatbuffer file descriptor using argument
        self.tflite_file = args.input_file

        # Set print level (0 ~ 1)
        self.print_level = args.verbose
        if (args.verbose > 1):
            self.print_level = 1
        if (args.verbose < 0):
            self.print_level = 0

        # Set tensor index list to print information
        self.print_all_tensor = True
        if (args.tensor != None):
            if (len(args.tensor) != 0):
                self.print_all_tensor = False
                self.print_tensor_index = []
                for tensor_index in args.tensor:
                    self.print_tensor_index.append(int(tensor_index))

        # Set operator index list to print information
        self.print_all_operator = True
        if (args.operator != None):
            if (len(args.operator) != 0):
                self.print_all_operator = False
                self.print_operator_index = []
                for operator_index in args.operator:
                    self.print_operator_index.append(int(operator_index))

        # Set config option
        self.save = False
        if args.config:
            self.save = True
            self.save_config = True

        if self.save == True:
            self.save_prefix = args.prefix

    def PrintModel(self, model_name, op_parser):
        printer = SubgraphPrinter(self.print_level, op_parser, model_name)

        if self.print_all_tensor == False:
            printer.SetPrintSpecificTensors(self.print_tensor_index)

        if self.print_all_operator == False:
            printer.SetPrintSpecificOperators(self.print_operator_index)

        printer.PrintInfo()

    def SaveModel(self, model_name, op_parser):
        saver = ModelSaver(model_name, op_parser)

        if self.save_config == True:
            saver.SaveConfigInfo(self.save_prefix)

    def main(self):
        # Generate Model: top structure of tflite model file
        buf = self.tflite_file.read()
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

            if self.save == False:
                # print all of operators or requested objects
                self.PrintModel(model_name, op_parser)
            else:
                # save all of operators in this model
                self.SaveModel(model_name, op_parser)

        print('==== Model Stats ({} Subgraphs) ===='.format(tf_model.SubgraphsLength()))
        print('')
        graph_stats.PrintGraphStats(stats, self.print_level)


if __name__ == '__main__':
    # Define argument and read
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "input_file", type=argparse.FileType('rb'), help="tflite file to read")
    arg_parser.add_argument(
        '-v', '--verbose', type=int, default=1, help="set print level (0~1, default: 1)")
    arg_parser.add_argument(
        '-t', '--tensor', nargs='*', help="tensor ID to print information (default: all)")
    arg_parser.add_argument(
        '-o',
        '--operator',
        nargs='*',
        help="operator ID to print information (default: all)")
    arg_parser.add_argument(
        '-c',
        '--config',
        action='store_true',
        help="Save the configuration file per operator")
    arg_parser.add_argument(
        '-p', '--prefix', help="file prefix to be saved (with -c/--config option)")
    args = arg_parser.parse_args()

    # Call main function
    TFLiteModelFileParser(args).main()
