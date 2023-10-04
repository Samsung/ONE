#!/usr/bin/env python3

# Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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
import subprocess
import os
import importlib

from test_utils import TestRunner
from q_implant_validator import validate

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--driver', type=str, required=True)
parser.add_argument('--dump', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
driver = args.driver
dump = args.dump
model = args.model

module = importlib.import_module('qparam.' + model)

input_circle = input_dir + '.circle'
output_circle = output_dir + f'/{module._name_}/output.circle'
qparam_dir = output_dir + f'/{module._name_}/qparam.json'
h5_path = output_dir + f'/{module._name_}/output.h5'

if not os.path.exists(input_circle):
    print('fail to load input circle')
    quit(255)

# generate qparam.json and numpys
test_runner = TestRunner(output_dir)

test_runner.register(module._test_case_)

test_runner.run()

if not os.path.exists(qparam_dir):
    print('qparam generate fail')
    quit(255)

# run q-implant
subprocess.run([driver, input_circle, qparam_dir, output_circle], check=True)

if not os.path.exists(output_circle):
    print('output circle generate fail')
    quit(255)

# dump circle to h5
subprocess.run([dump, '--tensors_to_hdf5', h5_path, output_circle], check=True)

if not os.path.exists(h5_path):
    print('h5 dump failed')
    quit(255)

if not validate(h5_path, output_dir + f'/{module._name_}', qparam_dir):
    quit(255)

quit(0)
