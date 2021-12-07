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

import argparse
from parser.model_parser import TFLiteModelFileParser
'''
Why is this file named as `model_parser.py` which is same to `parser/model_parser.py`?
- Until now, users have used by the path `tools/tflitefile_tool/model_parser.py`.
- Let's change the name to the proper name like `main.py` after the task for revision is done.
'''

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
