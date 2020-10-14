#!/usr/bin/env python3

# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

import onnx
import os
import sys


def _dump_operators(onnx_model):
    for node in onnx_model.graph.node:
        print(node.op_type)


def _help_exit(cmd_name):
    print('Dump ONNX model file Operators')
    print('Usage: {0} [onnx_path]'.format(cmd_name))
    print('')
    exit()


def main():
    if len(sys.argv) < 2:
        _help_exit(os.path.basename(sys.argv[0]))

    onnx_model = onnx.load(sys.argv[1])
    onnx.checker.check_model(onnx_model)

    _dump_operators(onnx_model)


if __name__ == "__main__":
    main()
