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

import numpy as np
import sys
import json
import struct


def printUsage(progname):
    print("%s <.json>" % (progname))
    print("  This program shows TFLite operations with its input/output shapes.")


if len(sys.argv) < 2:
    printUsage(sys.argv[0])
    exit()

filename = sys.argv[1]
f = open(filename)
j = json.loads(f.read())

tensors = j['subgraphs'][0]['tensors']
operators = j['subgraphs'][0]['operators']
opcodes = j['operator_codes']

for o in operators:
    op_name = "Undefined"
    if 'opcode_index' in o:
        op = opcodes[o['opcode_index']]
        if 'custom_code' in op:
            op_name = op['custom_code']
        elif 'builtin_code' in op:
            op_name = op['builtin_code']
    elif 'builtin_options_type' in o:
        # if we cannot find opcode_index, print option type instead.
        op_name = o['builtin_options_type']
    print("Layer:", op_name)

    print("    Input shapes ---")
    for inp in o['inputs']:
        print("      ", tensors[inp]['shape'])
    print("    Output shapes ---")
    for outp in o['outputs']:
        print("      ", tensors[outp]['shape'])
