#!/usr/bin/env python3
import argparse
import subprocess
import os
import importlib

from test_utils import TestRunner

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--driver', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
driver = args.driver
model = args.model

module = importlib.import_module('import.' + model)

input_circle = input_dir + '.circle'
output_circle = output_dir + f'/{module._name_}/output.circle'
qparam_dir = output_dir + f'/{module._name_}/qparam.json'

if not os.path.exists(input_circle):
    print('fail to load input circle')
    quit(255)

# generate qparam.json and numpys
test_runner = TestRunner(output_dir)

test_runner.register(module._model_)

test_runner.run()

if not os.path.exists(qparam_dir):
    print('qparam generate fail')
    quit(255)

# run q-implant
subprocess.run([driver, input_circle, qparam_dir, output_circle], check=True)

if not os.path.exists(output_circle):
    print('output circle generate fail')
    quit(255)

quit(0)
