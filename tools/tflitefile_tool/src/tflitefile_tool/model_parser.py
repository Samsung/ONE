#!/usr/bin/env python

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
'''
Why is this file named as `model_parser.py` which is same to `parser/model_parser.py`?
- Until now, users have used by the path `tools/tflitefile_tool/model_parser.py`.
- Let's change the name to the proper name like `main.py` after the task for revision is done.
'''

import argparse
from parser.model_parser import ModelParser
from printer.subgraph_printer import SubgraphPrinter
from saver.model_saver import ModelSaver


class MainOption(object):
    def __init__(self, args):
        self.model_file = args.input_file

        # Set print level (0 ~ 2)
        self.print_level = args.verbose
        if (args.verbose > 2):
            self.print_level = 2
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


def PrintSubgraph(option, subg):
    printer = SubgraphPrinter(option.print_level, subg)

    if option.print_all_tensor == False:
        printer.SetPrintSpecificTensors(option.print_tensor_index)

    if option.print_all_operator == False:
        printer.SetPrintSpecificOperators(option.print_operator_index)

    printer.PrintInfo()


def SaveSubgraph(option, subg):
    saver = ModelSaver(subg)

    if option.save_config == True:
        saver.SaveConfigInfo(option.save_prefix)


def main():
    # Define argument and read
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file",
                            type=argparse.FileType('rb'),
                            help="tflite file to read")
    arg_parser.add_argument('-v',
                            '--verbose',
                            type=int,
                            default=1,
                            help="set print level (0~1, default: 1)")
    arg_parser.add_argument('-t',
                            '--tensor',
                            nargs='*',
                            help="tensor ID to print information (default: all)")
    arg_parser.add_argument('-o',
                            '--operator',
                            nargs='*',
                            help="operator ID to print information (default: all)")
    arg_parser.add_argument('-c',
                            '--config',
                            action='store_true',
                            help="Save the configuration file per operator")
    arg_parser.add_argument('-p',
                            '--prefix',
                            help="file prefix to be saved (with -c/--config option)")
    args = arg_parser.parse_args()
    option = MainOption(args)

    subg_list = ModelParser(option.model_file).Parse()

    for subg in subg_list:
        if option.save == False:
            # print all of operators or requested objects
            PrintSubgraph(option, subg)
        else:
            # save all of operators in this model
            SaveSubgraph(option, subg)


if __name__ == '__main__':
    main()
